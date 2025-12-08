#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    euler_from_quaternion
)

class NozzleTcpCalibrator(object):
    def __init__(self):
        self.nozzle_pose = None
        self.tcp_pose = None

        self.transforms_t = []   # list of [x, y, z]
        self.transforms_q = []   # list of [x, y, z, w]

        self.max_samples = rospy.get_param("~max_samples", 100)
        self.time_tol = rospy.get_param("~time_tolerance", 0.02)  # sec

        rospy.Subscriber("/qualisys_map/nozzle/pose",
                         PoseStamped, self.nozzle_cb, queue_size=1)
        rospy.Subscriber("/mur620c/UR10_r/global_tcp_pose",
                         PoseStamped, self.tcp_cb, queue_size=1)

    def nozzle_cb(self, msg):
        self.nozzle_pose = msg
        self.try_add_sample()

    def tcp_cb(self, msg):
        self.tcp_pose = msg
        self.try_add_sample()

    def pose_to_matrix(self, pose):
        q = [pose.orientation.x,
             pose.orientation.y,
             pose.orientation.z,
             pose.orientation.w]
        T = quaternion_matrix(q)
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T

    def try_add_sample(self):
        if self.nozzle_pose is None or self.tcp_pose is None:
            return

        # Zeitabgleich
        dt = (self.nozzle_pose.header.stamp - self.tcp_pose.header.stamp).to_sec()
        if abs(dt) > self.time_tol:
            return

        # Optional: gleicher Frame?
        if self.nozzle_pose.header.frame_id != self.tcp_pose.header.frame_id:
            rospy.logwarn_throttle(5.0,
                "Frame-IDs unterschiedlich: %s vs %s" %
                (self.nozzle_pose.header.frame_id,
                 self.tcp_pose.header.frame_id))
            # Wenn du sicher bist, dass die Frames identisch sind, kannst du das ignorieren.

        # T_world_nozzle und T_world_tcp
        T_w_n = self.pose_to_matrix(self.nozzle_pose.pose)
        T_w_tcp = self.pose_to_matrix(self.tcp_pose.pose)

        # T_nozzle_tcp = T_world_nozzle^-1 * T_world_tcp
        T_n_tcp = np.dot(np.linalg.inv(T_w_n), T_w_tcp)

        t = T_n_tcp[0:3, 3]
        q = quaternion_from_matrix(T_n_tcp)

        self.transforms_t.append(t)
        self.transforms_q.append(q)

        rospy.loginfo("Sample %d / %d aufgenommen" %
                      (len(self.transforms_t), self.max_samples))

        if len(self.transforms_t) >= self.max_samples:
            self.compute_and_print_result()
            rospy.signal_shutdown("Calibration finished")

    def compute_and_print_result(self):
        Ts = np.vstack(self.transforms_t)      # shape (N, 3)
        Qs = np.vstack(self.transforms_q)      # shape (N, 4)

        t_mean = np.mean(Ts, axis=0)

        q_mean = np.mean(Qs, axis=0)
        q_mean = q_mean / np.linalg.norm(q_mean)

        roll, pitch, yaw = euler_from_quaternion(q_mean)  # rad

        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        rospy.loginfo("======== Mittelwert nozzle -> tcp ({} Samples) ========".format(
            self.max_samples))
        rospy.loginfo("Translation [m]: x = %.6f, y = %.6f, z = %.6f",
                      t_mean[0], t_mean[1], t_mean[2])
        rospy.loginfo("Rotation [deg]: roll = %.3f, pitch = %.3f, yaw = %.3f",
                      roll_deg, pitch_deg, yaw_deg)

        # Falls du die Zahlen direkt in eine static_transform_publisher-Zeile kopieren willst:
        rospy.loginfo("static_transform (rad) nozzle->tcp:")
        rospy.loginfo("x y z roll pitch yaw = "
                      "%.6f %.6f %.6f  %.6f %.6f %.6f",
                      t_mean[0], t_mean[1], t_mean[2],
                      roll, pitch, yaw)


if __name__ == "__main__":
    rospy.init_node("nozzle_tcp_calibrator")
    calib = NozzleTcpCalibrator()
    rospy.loginfo("Nozzle-TCP Kalibrier-Node gestartet.")
    rospy.spin()
