#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32


class LaserProfileController(object):
    def __init__(self):
        rospy.init_node("laser_profile_controller", anonymous=True)

        # --- Parameters ---
        self.tcp_pose_topic = rospy.get_param(
            "~tcp_pose_topic",
            "/mur620c/UR10_r/ur_calibrated_pose"
        )
        self.cmd_topic = rospy.get_param(
            "~cmd_topic",
            "/mur620c/UR10_r/twist_controller/command_collision_free"
        )

        self.lateral_error_topic = rospy.get_param(
            "~lateral_error_topic",
            "/laser_profile/lateral_error"
        )
        self.height_error_topic = rospy.get_param(
            "~height_error_topic",
            "/laser_profile/height_error"
        )

        self.k_p = rospy.get_param("~k_p", 0.3)
        self.max_vel = rospy.get_param("~max_vel", 0.15)
        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.95)
        self.control_rate = rospy.get_param("~control_rate", 200.0)

        # Für Höhencheck
        self.target_layer_height = rospy.get_param("~target_layer_height", -10.0)
        self.min_expected_height = rospy.get_param("~min_expected_height", -30.0)

        # --- State ---
        self.ur_tcp_pose = None
        self.lateral_error = None
        self.height_error = None
        self.prev_output = [0.0, 0.0, 0.0]

        # --- Publisher ---
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        # --- Subscribers ---
        rospy.Subscriber(self.tcp_pose_topic, PoseStamped,
                         self.ur_tcp_callback, queue_size=1)
        rospy.Subscriber(self.lateral_error_topic, Float32,
                         self.lateral_error_callback, queue_size=1)
        rospy.Subscriber(self.height_error_topic, Float32,
                         self.height_error_callback, queue_size=1)

        rospy.loginfo("LaserProfileController started.")

        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

    # ------------------------------------------------------------------

    def ur_tcp_callback(self, msg: PoseStamped):
        self.ur_tcp_pose = msg

    def lateral_error_callback(self, msg: Float32):
        self.lateral_error = msg.data

    def height_error_callback(self, msg: Float32):
        self.height_error = msg.data

    def smooth_output(self, new_output):
        smoothed = []
        for i in range(3):
            smoothed_value = (self.output_smoothing_coeff * self.prev_output[i] +
                              (1.0 - self.output_smoothing_coeff) * new_output[i])
            smoothed.append(smoothed_value)
        self.prev_output = smoothed
        return np.array(smoothed)

    # ------------------------------------------------------------------

    def update(self):
        if self.ur_tcp_pose is None:
            rospy.logwarn_throttle(1.0, "No TCP pose received yet.")
            return
        if self.lateral_error is None:
            rospy.logwarn_throttle(1.0, "No lateral_error received yet.")
            return

        # --- Optional: Höhencheck ---
        if self.height_error is not None:
            current_height = self.target_layer_height - self.height_error
            if current_height < self.min_expected_height:
                rospy.logwarn_throttle(
                    5.0,
                    "Profile point too low in controller: %.2f < %.2f",
                    current_height,
                    self.min_expected_height
                )
                twist = Twist()  # stoppen
                self.cmd_pub.publish(twist)
                return

        # --- Lateralgeschwindigkeit in TCP-Frame (Y-Achse) ---
        # lateral_error ist wie vorher avg_deviation in Indexschritten
        v_lat = -self.k_p * self.lateral_error * 0.001  # Index -> m
        v_lat = np.clip(v_lat, -self.max_vel, self.max_vel)

        # v_tcp: nur seitliche Bewegung entlang -Y des TCP
        v_tcp = np.array([0.0, -v_lat, 0.0])
        v_tcp = self.smooth_output(v_tcp)

        # --- TCP -> UR base Rotation ---
        q = self.ur_tcp_pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1 - 2*qx*qx - 2*qy*qy]
        ])

        v_urbase = R.dot(v_tcp)

        twist = Twist()
        twist.linear.x = v_urbase[0]
        twist.linear.y = v_urbase[1]
        twist.linear.z = 0.0

        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    try:
        LaserProfileController()
    except rospy.ROSInterruptException:
        pass
