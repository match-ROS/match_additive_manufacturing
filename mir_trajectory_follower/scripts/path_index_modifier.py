#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Int32, Float32MultiArray

class TimeWarpingIndex:
    def __init__(self):
        rospy.init_node("mir_time_warping_index")

        # tunable parameters
        self.rate = 100.0
        self.speed_gain = rospy.get_param("~speed_gain", 0.01)  # how fast dt adapts
        self.max_speed_scale = rospy.get_param("~max_speed_scale", 1.05)
        self.min_speed_scale = rospy.get_param("~min_speed_scale", 0.95)
        self.max_offset_idx = rospy.get_param("~max_offset_idx", 20)
        self.global_avg_speed = 0.07 # default value in m/s 

        # data
        self.mir_path = None
        self.timestamps = None
        self.t_orig = None
        # self.global_avg_speed = None

        self.ur_index = 0
        self.dt_mir = 0.1        # starts with 0.1s per step

        # subs
        rospy.Subscriber("/mir_path_original", Path, self.cb_path)
        rospy.Subscriber("/mir_path_timestamps", Float32MultiArray, self.cb_time)
        rospy.Subscriber("/path_index", Int32, self.cb_ur_index)

        # internal time
        current_index = rospy.wait_for_message("/path_index", Int32).data
        self.t_mir = current_index * self.dt_mir

        # pub
        self.pub_mod = rospy.Publisher("/path_index_modified", Int32, queue_size=1)

        rospy.Timer(rospy.Duration(1.0/self.rate), self.on_timer)

    def cb_path(self, msg):
        self.mir_path = msg

    def cb_time(self, msg):
        self.timestamps = np.array(msg.data, dtype=float)
        self.t_orig = self.timestamps
        # self.global_avg_speed = self.compute_global_avg_speed()

    def cb_ur_index(self, msg):
        self.ur_index = msg.data
        self.t_ur = self.ur_index * self.dt_mir

    def compute_global_avg_speed(self):
        if self.mir_path is None or self.timestamps is None:
            return 0.0

        pts = self.mir_path.poses
        speeds = []
        for i in range(1, len(pts)):
            p0 = pts[i-1].pose.position
            p1 = pts[i].pose.position
            dist = np.linalg.norm([
                p1.x - p0.x,
                p1.y - p0.y,
            ])
            dt = self.timestamps[i] - self.timestamps[i-1]
            if dt <= 0: dt = 1e-3
            speeds.append(dist / dt)
        return np.mean(speeds)

    def estimate_local_speed(self, idx, window=1):
        if self.global_avg_speed is None:
            return self.global_avg_speed
        
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

        ratio = local_speed / self.global_avg_speed
        error = (1.0 / max(ratio, 1e-6)) - 1.0     # proportional error

        # smooth proportional adaptation of dt_mir
        dt_mir = 1.0 / self.rate
        dt_mir += self.speed_gain  * error 
        #print("dt_mir before clamp:", self.dt_mir)

        # clamp speed scaling
        # self.dt_mir = np.clip(
        #     self.dt_mir,
        #     0.1 * self.min_speed_scale,
        #     0.1 * self.max_speed_scale
        # )

        # ----- 3) Advance MiR’s internal time -----
        self.t_mir += dt_mir

        # clamp within path duration
        t_max = self.t_orig[-1]
        self.t_mir = np.clip(self.t_mir, self.t_ur - 2.0, self.t_ur + 2.0 if self.t_ur < t_max else t_max)

        # ----- 4) Convert t_mir → MiR index -----
        idx_mir = np.interp(self.t_mir, self.t_orig, np.arange(len(self.t_orig)))
        #print("t_mir:", self.t_mir, " idx_mir:", idx_mir)
        #print("index ur:", self.ur_index)

        # ----- 5) Limit UR offset -----
        idx_mir = int(round(idx_mir))
        idx_mir = np.clip(idx_mir,
            self.ur_index - self.max_offset_idx,
            self.ur_index + self.max_offset_idx
        )

        print("index diff UR-MiR:", self.ur_index - idx_mir, " dt_mir:", dt_mir)

        # ----- 6) Publish -----
        self.pub_mod.publish(Int32(idx_mir))


if __name__ == "__main__":
    TimeWarpingIndex()
    rospy.spin()
