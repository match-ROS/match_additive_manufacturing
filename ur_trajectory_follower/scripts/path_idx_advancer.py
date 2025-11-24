#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped, Vector3
from additive_manufacturing_msgs.msg import Vector3Array
from nav_msgs.msg import Path
from std_msgs.msg import Int32

class PathIndexAdvancer:
    def __init__(self):
        rospy.init_node('path_index_advancer', anonymous=True)

        # Example path of (x, y) waypoints
        # In practice, load from params or another topic
        self.path = Path()
        self.current_index = rospy.get_param('~initial_path_index', 1)  # Start at the second waypoint by default
        if self.current_index < 1:
            rospy.logwarn("Initial path index is less than 1. Setting to 1.")
            self.current_index = 1

        # Thresholds (meters) for each metric
        # Adjust them as needed
        self.radius_threshold = rospy.get_param("~radius_threshold", 0.1)
        self.collinear_threshold = rospy.get_param("~collinear_threshold", 0.02)
        self.virtual_line_threshold = rospy.get_param("~virtual_line_threshold", 0.02)
        self.prev_idx_dist = rospy.get_param("~prev_idx_dist", 1) # distance to previous waypoint for virtual line
        self.next_idx_dist = rospy.get_param("~next_idx_dist", 1) # distance to next waypoint for virtual line
        self.use_virtual_bisector = rospy.get_param("~use_virtual_bisector", True)
        rospy.loginfo("Virtual line mode: %s", "bisector" if self.use_virtual_bisector else "segment")
        

        # Publishers: next waypoint index and next waypoint pose
        self.index_pub = rospy.Publisher("/path_index", Int32, queue_size=10, latch=True)
        self.goal_pose_pub = rospy.Publisher("/next_goal", PoseStamped, queue_size=10, latch=True)
        self.normal_pub = rospy.Publisher("/normal_vector", Vector3, queue_size=10, latch=True)
        
        # Choose the metric to use
        metric = rospy.get_param("~metric", "radius")
        if metric not in ["radius", "collinear", "virtual line"]:
            rospy.logerr("Invalid metric name. Please choose 'radius', 'collinear', or 'virtual line'.")
            rospy.signal_shutdown("Invalid metric name.")
            
        if metric == "radius":
            rospy.loginfo("Using 'radius' metric.")
            self.check_condition = self.check_radius_condition
        elif metric == "collinear":
            rospy.loginfo("Using 'collinear' metric.")
            self.check_condition = self.check_collinear_condition
        elif metric == "virtual line":
            rospy.loginfo("Using 'virtual line' metric.")
            self.check_condition = self.check_virtual_line_condition

        # Get the path from the topic /ur_path_transformed
        self.path = rospy.wait_for_message("/ur_path_transformed", Path)
        self.path_length = len(self.path.poses)

        rospy.loginfo(f"Received path with {self.path_length} waypoints")
        self.normals = rospy.wait_for_message("/ur_path_normals", Vector3Array)       
        rospy.loginfo(f"Received normals with {len(self.normals.vectors)} vectors")

        # Create subscriber to your robot's current pose
        self.pose_sub = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)

        self.publish_current_goal()
        
    
    def pose_callback(self, pose_msg):
        """
        Callback that checks the current robot pose against the current waypoint.
        Advances the index if the defined metric is satisfied.
        """
        robot_x = pose_msg.pose.position.x
        robot_y = pose_msg.pose.position.y

        # If we are already at the final waypoint, do nothing further
        if self.current_index >= self.path_length - 1:
            rospy.loginfo_throttle(10, "Already at the final waypoint!")
            self.publish_current_goal()
            return

        # Check metrics to decide if we advance to the next waypoint

        # 1. Check metric
        if self.check_condition(robot_x, robot_y):
            #rospy.loginfo("Advancing waypoint index")
            self.advance_waypoint_index()

            # Publish updated index and goal
            self.publish_current_goal()

    def advance_waypoint_index(self):
        """
        Advances current_index safely, so we don't exceed the last waypoint.
        """
        self.current_index = min(self.current_index + 1, self.path_length - 1)

    def publish_current_goal(self):
        """
        Publishes the current waypoint index and the corresponding goal pose.
        """
        rospy.logwarn_throttle(1,f"Publishing current goal at index {self.current_index}")
        # Publish the waypoint index
        self.index_pub.publish(self.current_index)

        # Publish the goal pose
        self.goal_pose_pub.publish(self.path.poses[self.current_index])

        # Publish the normal vector
        self.normal_pub.publish(self.normals.vectors[self.current_index-1])

    def check_radius_condition(self, x, y):
        """
        Returns True if the (x,y) is within self.radius_threshold of the current waypoint.
        """
        goal_x, goal_y = self.path.poses[self.current_index].pose.position.x, self.path.poses[self.current_index].pose.position.y
        dist = math.hypot(x - goal_x, y - goal_y)
        return dist < self.radius_threshold

    def check_collinear_condition(self, x, y):
        """
        Checks if the robot’s pose is near the current waypoint, but specifically
        along the line segment from the *previous* waypoint to the *current* waypoint.
        
        A common approach:
          1. If current_index == 0, we have no "previous waypoint"; set current_index=1.
          2. Compute the robot's projection onto that line segment.
          3. Measure distance from that projection to the next waypoint.
          4. Return True if below self.collinear_threshold.
        """
        if self.current_index == 0:
            # No previous waypoint
            self.current_index = 1
            return True

        prev_x, prev_y = self.path.poses[self.current_index - 1].pose.position.x, self.path.poses[self.current_index - 1].pose.position.y
        curr_x, curr_y = self.path.poses[self.current_index].pose.position.x, self.path.poses[self.current_index].pose.position.y

        dist_to_line = self.point_proj_on_line_dist(x, y, prev_x, prev_y, curr_x, curr_y)
        return dist_to_line < self.collinear_threshold
    
    def check_virtual_line_condition(self, x, y):
        """
        Checks if the robot has crossed a 'virtual line' that is:
          1) anchored at a point L, which is 'virtual_line_threshold' away from
             the current waypoint along the *incoming* path segment,
          2) oriented by the angle bisector of the current segment direction
             and the next segment direction.
             Segment defined by the previous and next waypoints (prev_idx_dist, next_idx_dist).

        As soon as the robot position is on the 'positive' side of that line,
        True is returned.
        """

        # We need a valid previous and next waypoint for the geometry
        if self.current_index == 0 or self.current_index >= self.path_length - 1:
            rospy.logwarn("Can't define the geometry at the extremes.")
            return False  # can't define the geometry at the extremes

        # Gather points
        prev_idx = max(0, self.current_index - self.prev_idx_dist)
        next_idx = min(self.path_length - 1, self.current_index + self.next_idx_dist)
        px_prev = self.path.poses[prev_idx].pose.position
        px_curr = self.path.poses[self.current_index].pose.position
        px_next = self.path.poses[next_idx].pose.position

        # check for z height change
        z_height_curr = px_curr.z
        z_height_prev = px_prev.z
        if abs(z_height_curr - z_height_prev) > 1e-3:
            return True  # if z height changed significantly, consider it crossed - go to next waypoint

        # Convert them to vectors
        p_prev = (px_prev.x, px_prev.y)
        p_curr = (px_curr.x, px_curr.y)
        p_next = (px_next.x, px_next.y)

        # v1: from prev to curr; v2: from curr to next
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        # Norms
        len_v1 = math.hypot(v1[0], v1[1])
        len_v2 = math.hypot(v2[0], v2[1])
        if len_v1 < 1e-9 or len_v2 < 1e-9:
            return True  # degenerate segment, consider it crossed

        d1 = (v1[0]/len_v1, v1[1]/len_v1)
        d2 = (v2[0]/len_v2, v2[1]/len_v2)

        if self.use_virtual_bisector:
            # The angle bisector of d1 and d2 => b
            bx = d1[0] + d2[0]
            by = d1[1] + d2[1]
            mag_b = math.hypot(bx, by)
            if mag_b < 1e-9:
                rospy.logdebug("Degenerate bisector (segments nearly opposite), falling back to radius check.")
                return self.check_radius_condition(x,y)  # fallback if the bisector is ill-defined
            b = (bx/mag_b, by/mag_b)
        else:
            # Use the current segment direction as line normal
            b = d1

        # 1) Find point L on the current segment that is "virtual_line_threshold"
        #    away from p_curr, going BACK along v1
        L = (
            p_curr[0] - self.virtual_line_threshold * d1[0],
            p_curr[1] - self.virtual_line_threshold * d1[1],
        )

        # 2) We treat b as the normal to the virtual line that passes through L
        #    That line is defined by all x s.t. (x - L) dot b = 0.

        # 3) Evaluate sign for the robot position
        robot_vec = (x - L[0], y - L[1])
        s = robot_vec[0]*b[0] + robot_vec[1]*b[1]

        # rospy.loginfo(f"Virtual line condition: s={s}")
        # rospy.loginfo(f"Virtual line condition: L={L}, b={b}, robot={robot_vec}")
        # rospy.loginfo(f"global nozzle: x={x}, y={y}")

        # We'll consider "crossed" if s >= 0
        crossed = (s >= 0)

        return crossed

    @staticmethod
    def point_proj_on_line_dist(px, py, x1, y1, x2, y2):
        """
        Computes the distance from point (x2, y2) to the projection of (px, py) onto the line segment
        between (x1, y1) and (x2, y2). If the projection falls outside the segment,
        returns the distance to the nearest endpoint.
        """
        # Line segment vector
        seg_x = x2 - x1
        seg_y = y2 - y1
        seg_len_sq = seg_x**2 + seg_y**2

        if seg_len_sq < 1e-9:
            # The segment is effectively a point
            return math.hypot(px - x1, py - y1)

        # Compute projection of point (px, py) onto the line's parameter t
        t = ((px - x1) * seg_x + (py - y1) * seg_y) / seg_len_sq

        # If t < 0, before segment start; if t > 1, after segment end
        t_clamped = max(0.0, min(1.0, t))

        # # Projection point on segment
        # proj_x = x1 + t_clamped * seg_x
        # proj_y = y1 + t_clamped * seg_y

        # # Return distance from (px, py) to this projection
        # return math.hypot(px - proj_x, py - proj_y)
        
        return abs(1 - t_clamped) * seg_len_sq

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = PathIndexAdvancer()
    node.run()
