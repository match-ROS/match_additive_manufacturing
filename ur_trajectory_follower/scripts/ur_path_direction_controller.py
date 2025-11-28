#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32
import numpy as np
from sensor_msgs.msg import JointState

class DirectionController:
    def __init__(self):
        # height controller: fixed dt assumed
        self.nozzle_height_default = rospy.get_param("~nozzle_height_default", 0.1)
        self.nozzle_height_override = 0.0
        self.kp_z = rospy.get_param("~kp_z", 0.0)
        self.ki_z = rospy.get_param("~ki_z", 0.0)
        self.kd_z = rospy.get_param("~kd_z", 0.0)
        rospy.loginfo(f"Z-controller gains: kp={self.kp_z}, ki={self.ki_z}, kd={self.kd_z}")
        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/mur620c/joint_states")
        self.lift_joint_name = rospy.get_param("~lift_joint_name", "right_lift_joint")
        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.95)
        self.integral_z = 0
        self.prev_error_z = 0
        

        self.path: Path = Path()
        self.command_old_twist = Twist()
        self.current_index = 1  # Start at the first waypoint in the path
        self.trajectory_velocity=0
        self.velocity_override=1.0 # in percent
        self.current_lift_height = 0.0
        self.current_pose = None
        self.node_ready = False
        self.ff_mode = rospy.get_param("~ff_mode", 3) # 1: ff_only 2: feedback Control 3: ff for direction feedback of distance
        self.from_index_offset = int(rospy.get_param("~from_index_offset", -1))
        self.goal_index_offset = int(rospy.get_param("~goal_index_offset", 0))
        
        self.path = rospy.wait_for_message("/path", Path)
        rospy.Subscriber("/path_index", Int32, self.index_callback)
        if self.ff_mode > 1:
            rospy.Subscriber("/current_pose", PoseStamped, self.ee_pose_callback)
        rospy.Subscriber("/velocity_override", Float32, self.velocity_override_callback)
        rospy.Subscriber("/nozzle_height_override", Float32, self.nozzle_height_callback)        # lift height
        rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)

        rospy.wait_for_message(self.joint_state_topic, JointState)
        rospy.wait_for_message("/path_index", Int32)
        
        self.pub_ur_velocity_world = rospy.Publisher("/ur_twist_world", Twist, queue_size=10)
        rospy.sleep(0.1)  # allow publisher to set up
        self.node_ready = True
        rospy.loginfo("UR Direction Controller node initialized.")
        self.index_callback(Int32(data=self.current_index))  # initial calculation

    def nozzle_height_callback(self, height_msg: Float32):
        self.nozzle_height_override = height_msg.data

    def joint_state_callback(self, msg: JointState):
        try:
            idx = msg.name.index(self.lift_joint_name)
            self.current_lift_height = msg.position[idx]
        except ValueError:
            pass  # joint not found in this message


    def index_callback(self, index_msg: Int32):
        self.current_index = index_msg.data
        self.get_traj_velocity(self.from_index_offset, self.goal_index_offset)
        if not self.node_ready:
            return
        self.calculate_twist(self.from_index_offset, self.goal_index_offset)

    def velocity_override_callback(self, velocity_msg: Float32):
        self.velocity_override = velocity_msg.data

    def ee_pose_callback(self, pose_msg: PoseStamped):
        self.current_pose = pose_msg
        if not self.node_ready:
            return
        self.calculate_twist(self.from_index_offset, self.goal_index_offset)

    def _clamp_path_index(self, target_index: int) -> int:
        if not self.path.poses:
            return 0
        return max(0, min(target_index, len(self.path.poses) - 1))

    def get_traj_velocity(self, from_offset: int, goal_offset: int):
        """Calculate the velocity of the robot along the trajectory.
        The velocity is calculated as the distance between the last waypoint and the next waypoint divided by the time.
        """
        if not self.path.poses:
            rospy.logwarn("Received empty path; trajectory velocity set to zero.")
            self.trajectory_velocity = 0.0
            return
        last_idx = self._clamp_path_index(self.current_index + from_offset)
        next_idx = self._clamp_path_index(self.current_index + goal_offset)
        if last_idx == next_idx:
            rospy.logwarn("Configured waypoint offsets select identical indices; trajectory velocity set to zero.")
            self.trajectory_velocity = 0.0
            return

        # Get the last waypoint and the next waypoint
        last_waypoint = self.path.poses[last_idx]
        next_waypoint = self.path.poses[next_idx]

        # Compute the absolute velocity from the last waypoint to the next waypoint by dividing the distance by the time
        # The time is the difference between the timestamps of the two waypoints
        distance = ((next_waypoint.pose.position.x - last_waypoint.pose.position.x)**2 + (next_waypoint.pose.position.y - last_waypoint.pose.position.y)**2)**0.5
        dt = (next_waypoint.header.stamp - last_waypoint.header.stamp).to_sec()
        if dt > 0:
            self.trajectory_velocity = distance / dt
        else:
            rospy.logwarn("time difference <=0 encountered in trajectory velocity calculation.")
            self.trajectory_velocity = 0.0

    def get_direction(self, from_offset: int, goal_offset: int):
        """Get the direction from the current pose to the next waypoint in the path.
        Returns:
            direction_xy_norm (np.array): The normalized direction vector in the xy-plane.
            error_z (float): The error in the z-axis.
        """
        # Get direction from current pose and goal pose
        if not self.path.poses:
            return np.array([0, 0]), 0.0
        goal_idx = self._clamp_path_index(self.current_index + goal_offset)
        goal_pose = self.path.poses[goal_idx]
        if self.ff_mode == 1 or self.ff_mode == 3:
            from_idx = self._clamp_path_index(self.current_index + from_offset)
            from_pose = self.path.poses[from_idx]
        else:
            from_pose = self.current_pose
        direction = np.array([goal_pose.pose.position.x - from_pose.pose.position.x,
                              goal_pose.pose.position.y - from_pose.pose.position.y,
                              goal_pose.pose.position.z - from_pose.pose.position.z])
        direction_xy = direction[:2]
        norm_xy = np.linalg.norm(direction_xy)
        if norm_xy < 1e-6:
            return np.array([0, 0]), direction[2]

        if self.ff_mode == 3: # compute distance error along path
            to_goal = np.array([goal_pose.pose.position.x - from_pose.pose.position.x,
                                goal_pose.pose.position.y - from_pose.pose.position.y])
            distance_along_path = np.dot(to_goal, direction_xy / norm_xy)
            direction_xy_norm = ((1+distance_along_path) * direction_xy) / norm_xy
        else:
            direction_xy_norm = direction_xy / norm_xy
        return direction_xy_norm, direction[2]

    def smooth_output(self, control_command: Twist):
        """Smooth the output command using exponential moving average."""
        smoothed_command = Twist()
        smoothed_command.linear.x = (self.output_smoothing_coeff * self.command_old_twist.linear.x +
                                     (1 - self.output_smoothing_coeff) * control_command.linear.x)
        smoothed_command.linear.y = (self.output_smoothing_coeff * self.command_old_twist.linear.y +
                                     (1 - self.output_smoothing_coeff) * control_command.linear.y)
        smoothed_command.linear.z = (self.output_smoothing_coeff * self.command_old_twist.linear.z +
                                     (1 - self.output_smoothing_coeff) * control_command.linear.z)
        self.command_old_twist = smoothed_command

        return smoothed_command


    def calculate_twist(self, from_offset: int, goal_offset: int):
        """Control the direction of the robot to follow the path."""
        if self.ff_mode > 1 and self.current_pose is None:
            rospy.logwarn("No current pose received yet.")
            return
        
        # Compute distance error along the path direction
        direction_xy_norm, error_z = self.get_direction(from_offset, goal_offset)
        goal_idx = self._clamp_path_index(self.current_index + goal_offset)
        goal_pose = self.path.poses[goal_idx]

        if self.ff_mode == 3:
            from_pose = self.current_pose
            # Vector from reference point to goal
            to_goal = np.array([goal_pose.pose.position.x - from_pose.pose.position.x,
                                goal_pose.pose.position.y - from_pose.pose.position.y])
            # Project onto path direction to get distance error along path
            if np.linalg.norm(direction_xy_norm) > 1e-6:
                distance_along_path = np.dot(to_goal, direction_xy_norm)
            else:
                distance_along_path = 0.0
            #rospy.logwarn(f"Distance along path: {distance_along_path}")

        v_xy=direction_xy_norm*self.trajectory_velocity*self.velocity_override
        
        # v_z pid controller (Annahme fester Regeltakt, ohne dt)
        if self.ff_mode == 1 or self.ff_mode == 3:
            error_z += 0.0
        else:
            error_z += self.nozzle_height_default + self.nozzle_height_override 
        v_z=error_z*self.kp_z+self.integral_z*self.ki_z+(error_z-self.prev_error_z)*self.kd_z
        self.integral_z+=error_z
        self.prev_error_z=error_z

        # Create a Twist message to publish the control command (world_frame)
        control_command = Twist()
        control_command.linear.x = v_xy[0]
        control_command.linear.y = v_xy[1]
        control_command.linear.z = v_z
        control_command_smoothed = self.smooth_output(control_command)
        
        self.pub_ur_velocity_world.publish(control_command_smoothed)

if __name__ == "__main__":
    rospy.init_node("ur_direction_controller")
    direction_controller = DirectionController()
    rospy.spin()