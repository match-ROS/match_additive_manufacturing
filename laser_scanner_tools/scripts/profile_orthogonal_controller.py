#!/usr/bin/env python3
import rospy
import numpy as np
from collections import deque

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Pose, PoseStamped
from tf.transformations import euler_from_quaternion


class ProfileOrthogonalController(object):
    def __init__(self):
        rospy.init_node("profile_orthogonal_controller", anonymous=True)

        # Topics / Parameter
        self.profile_topic   = rospy.get_param("~profile_topic", "/profiles_float")
        self.cmd_topic       = rospy.get_param("~cmd_topic", "/laser_profile_offset_cmd_vel")
        self.mir_pose_topic  = rospy.get_param("~mir_pose_topic", "/mur620c/mir_pose_simple")
        self.ur_target_topic = rospy.get_param("~ur_target_topic", "/ur_target_pose")

        self.window_size = rospy.get_param("~window_size", 50)     # Mittelung über N Messungen
        self.k_p        = rospy.get_param("~k_p", 0.06)            # Reglerverstärkung
        self.max_vel    = rospy.get_param("~max_vel", 0.1)        # |v_quer| Begrenzung [m/s]
        self.min_expected_height = rospy.get_param("~min_expected_height", -30.0)  # Minimale erwartete Layerhöhe [m]

        self.deviation_history = deque(maxlen=self.window_size)

        # Letzte Posen
        self.mir_pose = None         # Pose der MiR in map
        self.ur_target_pose = None   # UR-Zielpose in map

        # Publisher / Subscriber
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)

        self.sub_profile = rospy.Subscriber(
            self.profile_topic,
            Float32MultiArray,
            self.profile_callback,
            queue_size=1
        )

        self.sub_mir_pose = rospy.Subscriber(
            self.mir_pose_topic,
            Pose,
            self.mir_pose_callback,
            queue_size=1
        )

        self.sub_ur_target = rospy.Subscriber(
            self.ur_target_topic,
            PoseStamped,
            self.ur_target_callback,
            queue_size=1
        )

        rospy.loginfo("ProfileOrthogonalController (map -> MiR) started.")
        rospy.spin()

    def mir_pose_callback(self, msg: Pose):
        self.mir_pose = msg

    def ur_target_callback(self, msg: PoseStamped):
        self.ur_target_pose = msg

    def profile_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.size == 0:
            return

        if self.ur_target_pose is None:
            rospy.logwarn_throttle(1.0, "No /ur_target_pose received yet.")
            return

        if self.mir_pose is None:
            rospy.logwarn_throttle(1.0, "No /mur620c/mir_pose_simple received yet.")
            return

        # --- Index-Abweichung berechnen ---
        center_idx = (data.size - 1) / 2.0   # inkl. NaNs

        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            rospy.logwarn_throttle(1.0, "No valid points in /profiles_float")
            return

        valid_indices = np.nonzero(valid_mask)[0]
        valid_values  = data[valid_mask]

        min_idx_valid_array = np.argmax(valid_values)
        min_idx = int(valid_indices[min_idx_valid_array])

        deviation = float(min_idx - center_idx)

        self.deviation_history.append(deviation)
        avg_deviation = float(np.mean(self.deviation_history))

        # --- Pfadrichtung aus UR-Target (yaw_path in map) ---
        ur_q = self.ur_target_pose.pose.orientation
        q_path = [ur_q.x, ur_q.y, ur_q.z, ur_q.w]
        _, _, yaw_path = euler_from_quaternion(q_path)

        # Tangente t wäre [cos(yaw_path), sin(yaw_path)]
        # Laterale Richtung (y des Pfades) im map-Frame:
        n_x_map = -np.sin(yaw_path)
        n_y_map =  np.cos(yaw_path)

        # P-Regler auf lateralen Fehler
        v_lat = -self.k_p * avg_deviation * 0.001   # Vorzeichen ggf. anpassen

        # Begrenzen
        v_lat = max(min(v_lat, self.max_vel), -self.max_vel)

        # Lateraler Geschwindigkeitsvektor im map-Frame
        v_x_map = v_lat * n_x_map
        v_y_map = v_lat * n_y_map

        # --- map -> MiR-Frame drehen ---
        mir_q = self.mir_pose.orientation
        q_mir = [mir_q.x, mir_q.y, mir_q.z, mir_q.w]
        _, _, yaw_mir = euler_from_quaternion(q_mir)

        c = np.cos(yaw_mir)
        s = np.sin(yaw_mir)

        # v_mir = R(yaw_mir)^T * v_map
        v_x_mir = c * v_x_map + s * v_y_map
        v_y_mir = -s * v_x_map + c * v_y_map

        

        twist = Twist()
        # Geschwindigkeiten jetzt im MiR-Frame
        twist.linear.x = -v_x_mir 
        twist.linear.y = v_y_mir
        twist.linear.z = 0.0

        highest_profile_point = valid_values[min_idx_valid_array]
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
            "min_idx=%d, center=%.2f, dev=%.2f, avg_dev=%.2f, "
            "yaw_path=%.3f, yaw_mir=%.3f, v_lat=%.3f, "
            "v_map=(%.3f, %.3f), v_mir=(%.3f, %.3f)",
            min_idx, center_idx, deviation, avg_deviation,
            yaw_path, yaw_mir, v_lat,
            v_x_map, v_y_map, v_x_mir, v_y_mir
        )


if __name__ == "__main__":
    try:
        ProfileOrthogonalController()
    except rospy.ROSInterruptException:
        pass
