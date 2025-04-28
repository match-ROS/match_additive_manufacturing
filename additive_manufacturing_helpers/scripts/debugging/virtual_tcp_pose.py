#!/usr/bin/env python
import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, Twist

class PoseShifter(object):
    def __init__(self):
        # Get parameters (all parameters are under the node's private namespace "~")
        self.twist_topic  = rospy.get_param("~twist_topic", "/mur620a/UR10_r/twist_fb_command")
        self.pose_topic   = rospy.get_param("~pose_topic", "/virtual_tcp_pose")
        self.twist_frame  = rospy.get_param("~twist_frame", "mur620a/base_footprint")
        self.pose_frame   = rospy.get_param("~pose_frame", "mur620a/base_footprint")
        self.update_rate  = rospy.get_param("~update_rate", 50.0)  # Hz
        init_pose_from_tcp = rospy.get_param("~init_pose_from_tcp", False)

        # Internal storage for the latest twist command.
        # Twist commands are assumed to be given in the twist_frame.
        self.current_twist = Twist()

        self.last_time = rospy.Time.now()

        # Set up tf listener for obtaining transforms.
        self.tf_listener = tf.TransformListener()

        ## Get the initial pose
        if init_pose_from_tcp:
            self.tcp_frame = rospy.get_param("~tcp_frame", "mur620a/UR10_r/tcp")
            # Get the initial pose from the transform between the tcp_frame and the pose_frame.
            rospy.loginfo("Waiting for transform from [%s] to [%s].", self.tcp_frame, self.pose_frame)
            self.tf_listener.waitForTransform(self.pose_frame, self.tcp_frame, rospy.Time(0), rospy.Duration(10.0))
            T_tcp_pose = self.tf_listener.lookupTransform(self.pose_frame, self.tcp_frame, rospy.Time(0))
            self.x = T_tcp_pose[0][0]
            self.y = T_tcp_pose[0][1]
            self.z = T_tcp_pose[0][2]
            # Convert the quaternion to Euler angles.
            # Note: we assume that the rotation is only around the z-axis (yaw).
            (_, _, self.theta) = tf.transformations.euler_from_quaternion(T_tcp_pose[1])
        else:
            # Initial pose (in the target pose frame)
            self.x     = rospy.get_param("~initial_pose_x", 0.0)
            self.y     = rospy.get_param("~initial_pose_y", 0.0)
            self.theta = rospy.get_param("~initial_pose_theta", 0.0)

        # Set up the publisher for the PoseStamped message.
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=10)

        # Subscribe to the twist command topic.
        rospy.Subscriber(self.twist_topic, Twist, self.twist_callback)

        # Create a timer to update/integrate the pose at a fixed rate.
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.update_callback)

        rospy.loginfo("PoseShifter node started: twist commands in frame [%s] will update pose in frame [%s].", 
                      self.twist_frame, self.pose_frame)

    def twist_callback(self, msg):
        # Store the most recent twist command.
        self.current_twist = msg

    def update_callback(self, event):
        # Compute time step since last update.

        # Look up the transform from the twist frame (the frame in which the twist is published)
        # to the pose frame (the frame in which we want to integrate and publish the pose).
        # try:
        #     (trans, rot) = self.tf_listener.lookupTransform(self.pose_frame, self.twist_frame, rospy.Time(0))
        #     # Use the rotation (quaternion) to compute the relative yaw.
        #     # This yaw tells us how to rotate a vector from the twist frame into the pose frame.
        #     (_, _, relative_yaw) = tf.transformations.euler_from_quaternion(rot)
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        #     rospy.logwarn("TF lookup failed: %s", str(e))
        #     return

        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        if dt <= 0:
            return
        self.last_time = now
        # --- Transform the Twist Command ---
        # Transform linear velocities from twist_frame to pose_frame.
        # Given a 2D rotation (only yaw) the transformation is:
        #   v'_x = cos(relative_yaw)*v_x - sin(relative_yaw)*v_y
        #   v'_y = sin(relative_yaw)*v_x + cos(relative_yaw)*v_y
        base_vx = self.current_twist.linear.x
        base_vy = self.current_twist.linear.y
        # transformed_vx = math.cos(relative_yaw) * base_vx + math.sin(relative_yaw) * base_vy
        # transformed_vy = -math.sin(relative_yaw) * base_vx + math.cos(relative_yaw) * base_vy
        transformed_vx = base_vx
        transformed_vy = base_vy

        # Transform angular velocity components (if using more than just the z-component).
        # Here we rotate the x and y components; we assume that angular.z (rotation about z) remains unchanged.
        # base_wx = self.current_twist.angular.x
        # base_wy = self.current_twist.angular.y
        base_wz = self.current_twist.angular.z
        # transformed_wx = math.cos(relative_yaw) * base_wx - math.sin(relative_yaw) * base_wy
        # transformed_wy = math.sin(relative_yaw) * base_wx + math.cos(relative_yaw) * base_wy
        transformed_wz = base_wz

        # --- Integrate the Pose ---
        # For planar motion, typically only linear x/y and angular z are integrated.
        # If the twist command were in the robot's own frame, one would normally perform:
        #   dx = v * cos(theta)*dt, dy = v * sin(theta)*dt.
        # Here, we assume that after transforming to the pose_frame these velocities can
        # be directly integrated. (If needed, more sophisticated integration may be applied.)
        self.x     += transformed_vx * dt
        self.y     += transformed_vy * dt
        self.theta += transformed_wz * dt

        # Normalize theta to the range [-pi, pi].
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # --- Publish the Pose ---
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.pose_frame
        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        pose_msg.pose.position.z = 0.0  # Assuming flat ground

        # Convert the yaw angle to a quaternion.
        quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    rospy.init_node("pose_shifter_node", anonymous=False)
    node = PoseShifter()
    rospy.spin()
