#!/usr/bin/env python3
import rospy
import numpy as np
from collections import deque

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped


class ProfileOrthogonalController(object):
    def __init__(self):
        rospy.init_node("profile_orthogonal_controller", anonymous=True)

        # --- Parameters ---
        self.profile_topic = rospy.get_param("~profile_topic", "/profiles_float")
        self.tcp_pose_topic = rospy.get_param(
            "~tcp_pose_topic",
            "/mur620c/UR10_r/ur_calibrated_pose"
        )
        self.cmd_topic = rospy.get_param(
            "~cmd_topic",
            #"/laser_profile_offset_cmd_vel"
            "/mur620c/UR10_r/twist_controller/command_collision_free"
        )

        self.window_size = rospy.get_param("~window_size", 10)
        self.k_p = rospy.get_param("~k_p", 0.3)
        self.max_vel = rospy.get_param("~max_vel", 0.15)
        self.min_expected_height = rospy.get_param("~min_expected_height", -30.0)
        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.95)
        self.control_rate = rospy.get_param("~control_rate", 200.0)

        self.deviation_history = deque(maxlen=self.window_size)

        # Last TCP pose
        self.ur_tcp_pose = None
        self.scan_data = None
        self.prev_output = [0.0, 0.0, 0.0]

        # --- Publisher ---
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        # --- Subscribers ---
        rospy.Subscriber(self.profile_topic, Float32MultiArray,
                         self.profile_callback, queue_size=1)

        rospy.Subscriber(self.tcp_pose_topic, PoseStamped,
                         self.ur_tcp_callback, queue_size=1)

        rospy.loginfo("ProfileOrthogonalController (TCP-based) started.")
        
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

    # --- Callbacks ---------------------------------------------------------

    def ur_tcp_callback(self, msg: PoseStamped):
        self.ur_tcp_pose = msg

    def smooth_output(self, new_output):
        smoothed = []
        for i in range(3):
            smoothed_value = (self.output_smoothing_coeff * self.prev_output[i] +
                              (1 - self.output_smoothing_coeff) * new_output[i])
            smoothed.append(smoothed_value)
        self.prev_output = smoothed
        return np.array(smoothed)


    def profile_callback(self, msg: Float32MultiArray):
        self.scan_data = msg.data
       
        
        
        
    def update(self):
        if self.ur_tcp_pose is None:
            rospy.logwarn_throttle(1.0, "No TCP pose received yet.")
            return

        data = np.array(self.scan_data, dtype=np.float32)
        if data.size == 0:
            return

        # --- Find strongest negative point (minimum value) ---
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            rospy.logwarn_throttle(1.0, "No valid laser points")
            return

        valid_values = data[valid_mask]
        valid_indices = np.nonzero(valid_mask)[0]

        min_idx_rel = np.argmax(valid_values)
        min_idx = int(valid_indices[min_idx_rel])

        center_idx = (data.size - 1) / 2.0
        deviation = float(min_idx - center_idx)

        # --- Smooth deviation over N frames ---
        self.deviation_history.append(deviation)
        avg_deviation = float(np.mean(self.deviation_history))

        print(avg_deviation)

        # --- Compute lateral speed in TCP frame (Y axis of TCP) ---
        v_lat = -self.k_p * avg_deviation * 0.001  # Scale factor to convert index to meters
        v_lat = np.clip(v_lat, -self.max_vel, self.max_vel)

        v_tcp = np.array([0.0, -v_lat, 0.0])
        v_tcp = self.smooth_output(v_tcp)

        # --- TCP -> UR base rotation ---
        q = self.ur_tcp_pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1 - 2*qx*qx - 2*qy*qy]
        ])

        # Rotate v_tcp into UR base
        v_urbase = R.dot(v_tcp)

        # --- Publish twist in UR base frame ---
        twist = Twist()
        twist.linear.x = v_urbase[0]
        twist.linear.y = v_urbase[1]
        twist.linear.z = 0.0

        highest_profile_point = data[min_idx]
        if highest_profile_point < self.min_expected_height:
            rospy.logwarn_throttle(
                5.0,
                "Profile point too low: %.2f < %.2f",
                highest_profile_point,
                self.min_expected_height
            )
            twist = Twist()  # Stoppen



        self.cmd_pub.publish(twist)

        rospy.logdebug(
            "min_idx=%d center=%.2f dev=%.2f avg=%.2f v_lat=%.3f v_ur=(%.3f %.3f)",
            min_idx, center_idx, deviation, avg_deviation, v_lat,
            v_urbase[0], v_urbase[1]
        )


if __name__ == "__main__":
    try:
        ProfileOrthogonalController()
    except rospy.ROSInterruptException:
        pass
