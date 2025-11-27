#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Int32, Float32MultiArray, Float32
from copy import deepcopy

class TimeWarpingIndex:
    def __init__(self):
        rospy.init_node("mir_time_warping_index")

        # tunable parameters
        self.rate = 100.0
        self.speed_gain = rospy.get_param("~speed_gain", 0.03)  # how fast dt adapts
        self.max_speed_scale = rospy.get_param("~max_speed_scale", 1.05)
        self.min_speed_scale = rospy.get_param("~min_speed_scale", 0.95)
        self.max_offset_idx = rospy.get_param("~max_offset_idx", 10)  # max index offset allowed
        self.max_mir_distance = rospy.get_param("~max_mir_distance", 0.15)  # 25 cm allowed
        self.global_avg_speed = 0.045  # default value in m/s 
        self.global_avg_speed_override = deepcopy(self.global_avg_speed)

        # data
        self.mir_path = None
        self.timestamps = None
        self.t_orig = None
        self.current_mir_pose = None

        self.ur_index = 0
        self.dt_mir = 0.1 # dT of the original MiR path

        # subs
        rospy.Subscriber("/mir_path_original", Path, self.cb_path)
        rospy.Subscriber("/mir_path_timestamps", Float32MultiArray, self.cb_time)
        rospy.Subscriber("/path_index", Int32, self.cb_ur_index)
        rospy.Subscriber("/velocity_override", Float32, self.velocity_override_cb)

        # wait for initial data
        rospy.loginfo("Waiting for initial data...")
        rospy.wait_for_message("/mir_path_original", Path)
        rospy.wait_for_message("/velocity_override", Float32)


        # internal time
        current_index = rospy.wait_for_message("/path_index", Int32).data
        self.t_mir = current_index * self.dt_mir

        # pub
        self.pub_mod = rospy.Publisher("/path_index_modified", Int32, queue_size=1)

        # wait for index to change for the first time
        last_index = current_index
        while not rospy.is_shutdown():
            current_index = rospy.wait_for_message("/path_index", Int32).data
            if current_index != last_index:
                break
            rospy.sleep(0.01)

        rospy.Timer(rospy.Duration(1.0/self.rate), self.on_timer)

    def cb_path(self, msg):
        self.mir_path = msg

    def cb_time(self, msg):
        self.timestamps = np.array(msg.data, dtype=float)
        self.t_orig = self.timestamps

    def cb_ur_index(self, msg):
        self.ur_index = msg.data
        self.t_ur = self.ur_index * self.dt_mir

    def velocity_override_cb(self, msg):
        self.velocity_override = msg.data
        self.global_avg_speed_override = self.global_avg_speed #* self.velocity_override 

    def compute_dynamic_offset_limit(self):
        """
        Returns the maximum number of indices the MiR may run ahead,
        purely based on allowed geometric distance from the UR reference position.
        """

        if self.mir_path is None:
            return self.max_offset_idx

        pts = self.mir_path.poses
        N = len(pts)
        i0 = self.ur_index

        if i0 >= N - 1:
            return 0

        # Sollposition (aktuelle UR-Pfadposition)
        p0 = pts[i0].pose.position
        max_dist = self.max_mir_distance

        allowed_offset = 0

        # Nur bis max_offset_idx testen
        for k in range(1, self.max_offset_idx + 1):
            idx = i0 + k
            if idx >= N:
                break

            p = pts[idx].pose.position
            dist = np.linalg.norm([p.x - p0.x, p.y - p0.y])

            if dist > max_dist:
                break

            allowed_offset = k

        return allowed_offset


    def estimate_local_speed(self, idx, window=1):
        if self.global_avg_speed_override is None:
            return self.global_avg_speed_override
        
        pts = self.mir_path.poses
        start = max(1, idx - 1)
        end = min(len(pts)-1, idx + window)
        speeds = []
        for i in range(start, end):
            p0 = pts[i-1].pose.position
            p1 = pts[i].pose.position
            dist = np.linalg.norm([p1.x - p0.x, p1.y - p0.y])
            dt = self.timestamps[i] - self.timestamps[i-1]
            if dt <= 0: dt = 1e-3
            speeds.append(dist / dt)
        return np.mean(speeds)

    def on_timer(self, event):
        if self.timestamps is None or self.mir_path is None:
            return

        # ----- 1) Local speed trend -----
        local_speed = self.estimate_local_speed(self.ur_index)
        # ----- 2) Continuous dt_mir adaptation -----

        ratio = local_speed / self.global_avg_speed_override
        error = (1.0 / max(ratio, 1e-6)) - 1.0     # proportional error

        # smooth proportional adaptation of dt_mir
        dt_mir = self.velocity_override * 1.0 / self.rate 
        dt_mir += self.speed_gain  * error * 0.01 

        # ----- 3) Advance MiR’s internal time -----
        self.t_mir += dt_mir

        # clamp within path duration
        t_max = self.t_orig[-1]
        self.t_mir = np.clip(self.t_mir, self.t_ur - 2.0, self.t_ur + 2.0 if self.t_ur < t_max else t_max)

        # ----- 4) Convert t_mir → MiR index -----
        idx_mir = np.interp(self.t_mir, self.t_orig, np.arange(len(self.t_orig)))

        # ----- 5) Limit UR offset -----
        idx_mir = int(round(idx_mir))
        dynamic_limit = self.compute_dynamic_offset_limit()

        if abs(dynamic_limit) > 20:
            rospy.logwarn("Large dynamic limit computed: {}".format(dynamic_limit))

        idx_mir = np.clip(
            idx_mir,
            self.ur_index - dynamic_limit,
            self.ur_index + dynamic_limit
        )

        print("index diff UR-MiR:", self.ur_index - idx_mir, " dt_mir:", dt_mir)

        # ----- 6) Publish -----
        self.pub_mod.publish(Int32(idx_mir))


if __name__ == "__main__":
    TimeWarpingIndex()
    rospy.spin()
