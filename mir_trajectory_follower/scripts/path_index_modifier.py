#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from std_msgs.msg import Int32, Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np

class PathIndexModifier:
    def __init__(self):
        rospy.init_node("path_index_modifier")

        # Params
        self.max_offset = rospy.get_param("~max_offset", 10)               # UR reach limit
        self.max_offset_change = rospy.get_param("~max_offset_change", 1)  # per step
        self.look_ahead = rospy.get_param("~look_ahead", 5)

        # Data buffers
        self.mir_path = None
        self.timestamps = None
        self.current_offset = 0

        # Subscribers
        rospy.Subscriber("/mir_path_original", Path, self.cb_mir_path)
        rospy.Subscriber("/mir_path_timestamps", Float32MultiArray, self.cb_timestamps)
        rospy.Subscriber("/path_index", Int32, self.cb_path_index)

        # Publisher
        self.pub_idx_mod = rospy.Publisher("/path_index_modified", Int32, queue_size=1)

    # -------- Callbacks ----------
    def cb_mir_path(self, msg):
        self.mir_path = msg

    def cb_timestamps(self, msg):
        self.timestamps = np.array(msg.data)

    def cb_path_index(self, msg):
        if self.mir_path is None or self.timestamps is None:
            return

        ur_idx = msg.data
        self.update_offset(ur_idx)

        mod_idx = int(ur_idx + self.current_offset)
        mod_idx = max(0, min(mod_idx, len(self.timestamps) - 1))

        self.pub_idx_mod.publish(Int32(data=mod_idx))

    # -------- Core Logic ----------
    def update_offset(self, ur_idx):
        """
        Compute local speed of MiR around current index.
        Compare to global average.
        Adjust offset smoothly.
        """

        pts = self.mir_path.poses

        # Validate index window
        if ur_idx < 1 or ur_idx >= len(pts) - 2:
            return

        # global MiR average speed
        global_speeds = self.estimate_speeds(pts, self.timestamps)
        global_avg = np.mean(global_speeds)

        # local window speeds
        start = max(0, ur_idx - 1)
        end = min(len(pts) - 1, ur_idx + self.look_ahead)
        local_avg = np.mean(global_speeds[start:end])

        # offset logic
        if local_avg < 0.8 * global_avg:
            # MiR would be slow → give positive offset
            desired_change = +1
        elif local_avg > 1.2 * global_avg:
            # MiR fast → reduce offset
            desired_change = -1
        else:
            desired_change = 0

        # slope limit (acceleration constraint)
        desired_change = np.clip(desired_change,
                                 -self.max_offset_change,
                                 +self.max_offset_change)

        # apply
        new_offset = self.current_offset + desired_change

        # UR reachability limit
        new_offset = np.clip(new_offset,
                             -self.max_offset,
                             +self.max_offset)

        self.current_offset = int(new_offset)

    # -------- Speed Estimation ----------
    @staticmethod
    def estimate_speeds(poses, timestamps):
        speeds = []
        for i in range(1, len(poses)):
            p0 = poses[i - 1].pose.position
            p1 = poses[i].pose.position
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                dt = 1e-3
            dist = np.sqrt((p1.x - p0.x)**2 + (p1.y - p0.y)**2 + (p1.z - p0.z)**2)
            speeds.append(dist / dt)
        return np.array(speeds)


if __name__ == "__main__":
    PathIndexModifier()
    rospy.spin()
