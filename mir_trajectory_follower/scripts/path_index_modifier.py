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
        self.speed_gain = rospy.get_param("~speed_gain", 0.05)  # how fast dt adapts
        self.max_offset_idx = rospy.get_param("~max_offset_idx", 1)  # max index offset allowed
        self.max_mir_distance = rospy.get_param("~max_mir_distance", 0.15)  # 25 cm allowed
        self.global_avg_speed = 0.051  # default value in m/s 
        self.global_avg_speed_override = deepcopy(self.global_avg_speed)

        # data
        self.mir_path = None
        self.timestamps = None
        self.t_orig = None
        self.current_mir_pose = None

        self.ur_index = 0
        self.dt_mir = 0.1 # dT of the original MiR path
        self.path_index_topic = rospy.get_param("~path_index_topic", "/path_index")
        self.path_index_modified_topic = rospy.get_param("~path_index_modified_topic", "/path_index_modified")
        self.initial_path_index = self._parse_initial_path_index(rospy.get_param("~initial_path_index", -1))

        # subs
            self.mir_path_topic = rospy.get_param("~mir_path_topic", "/mir_path_transformed")
            self.mir_timestamps_topic = rospy.get_param("~mir_path_timestamps_topic", "/mir_path_timestamps")
            rospy.Subscriber(self.mir_path_topic, Path, self.cb_path)
            rospy.Subscriber(self.mir_timestamps_topic, Float32MultiArray, self.cb_time)
        rospy.Subscriber(self.path_index_topic, Int32, self.cb_ur_index)
        rospy.Subscriber("/velocity_override", Float32, self.velocity_override_cb)

        # wait for initial data
        rospy.loginfo("Waiting for initial data...")
            rospy.wait_for_message(self.mir_path_topic, Path)
        rospy.loginfo("Received MiR path.")
        rospy.wait_for_message("/velocity_override", Float32)
        rospy.loginfo("Received velocity override.")
        rospy.wait_for_message(self.mir_timestamps_topic, Float32MultiArray)
        rospy.loginfo("Received MiR timestamps.")

        # compute global average speed
        if self.mir_path is not None:
            self.global_avg_speed = self.compute_global_avg_speed_from_straights(0.1)
            rospy.loginfo(f"Global avg speed (straights only): {self.global_avg_speed:.3f} m/s")


        # internal time
        current_index = self._resolve_initial_path_index()
        self.t_mir = current_index * self.dt_mir

        # pub
        self.pub_mod = rospy.Publisher(self.path_index_modified_topic, Int32, queue_size=1)
        rospy.sleep(0.01) # wait for publisher to connect
        self.pub_mod.publish(Int32(current_index)) # initial publish to start trajectory follower

        # wait for index to change for the first time
        last_index = current_index
        
        while not rospy.is_shutdown():
            if self.ur_index != last_index:
                break
            rospy.sleep(0.01)
        rospy.loginfo("Starting time warping index modifier...")

        rospy.Timer(rospy.Duration(1.0/self.rate), self.on_timer)

    @staticmethod
    def _parse_initial_path_index(raw_value):
        try:
            idx = int(raw_value)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5.0, f"Invalid initial_path_index value '{raw_value}', ignoring.")
            return None
        return idx if idx >= 0 else None

    def _resolve_initial_path_index(self):
        if self.initial_path_index is not None:
            idx = max(0, self.initial_path_index)
            rospy.loginfo(f"Using initial path index {idx} from parameter for path_index_modifier.")
            self.ur_index = idx
            return idx

        # rospy.loginfo(f"Waiting for first path index on {self.path_index_topic}")
        # current_index = rospy.wait_for_message(self.path_index_topic, Int32).data
        current_index = self.initial_path_index if self.initial_path_index is not None else rospy.wait_for_message(self.path_index_topic, Int32).data
        self.ur_index = current_index
        return current_index

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

    def compute_path_curvature(self):
        """
        Returns array of curvature values for each path point.
        Start and end points receive curvature = 0.
        """
        pts = self.mir_path.poses
        N = len(pts)

        curvature = np.zeros(N)

        for i in range(1, N - 1):
            p_prev = np.array([pts[i-1].pose.position.x,
                            pts[i-1].pose.position.y])
            p     = np.array([pts[i].pose.position.x,
                            pts[i].pose.position.y])
            p_next = np.array([pts[i+1].pose.position.x,
                            pts[i+1].pose.position.y])

            a = np.linalg.norm(p - p_prev)
            b = np.linalg.norm(p_next - p)
            c = np.linalg.norm(p_next - p_prev)

            # Degenerate case
            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                curvature[i] = 0.0
                continue

            # Triangle area
            s = 0.5 * (a + b + c)
            A = max(0.0, s*(s-a)*(s-b)*(s-c)) ** 0.5

            # curvature k = 4A / (abc)
            curvature[i] = 4 * A / (a * b * c)

        return curvature

    def compute_global_avg_speed_from_straights(self, curvature_threshold=0.1):
        """
        Computes global average speed using only straight segments.
        curvature_threshold defines what counts as straight.
        """
        if self.mir_path is None or self.timestamps is None:
            return None

        curv = self.compute_path_curvature()
        pts = self.mir_path.poses
        t = self.timestamps

        speeds = []

        for i in range(1, len(pts)):
            if curv[i] > curvature_threshold:
                continue  # skip curves

            p0 = pts[i-1].pose.position
            p1 = pts[i].pose.position
            dist = np.linalg.norm([p1.x - p0.x, p1.y - p0.y])

            dt = t[i] - t[i-1]
            if dt <= 0:
                continue

            speeds.append(dist / dt)

        if len(speeds) == 0:
            # fallback: use full-path average
            return np.mean([
                np.linalg.norm([
                    pts[i].pose.position.x - pts[i-1].pose.position.x,
                    pts[i].pose.position.y - pts[i-1].pose.position.y
                ]) / max(t[i]-t[i-1], 1e-3)
                for i in range(1, len(pts))
            ])

        return np.mean(speeds)

    def estimate_local_speed(self, idx, window=5):
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

        rospy.loginfo_throttle(1.0, f"UR idx: {self.ur_index}, MiR idx: {idx_mir}, dt_mir: {dt_mir:.4f}, local speed: {local_speed:.3f} m/s, dynamic limit: {dynamic_limit}")

        # ----- 6) Publish -----
        self.pub_mod.publish(Int32(idx_mir))


if __name__ == "__main__":
    TimeWarpingIndex()
    rospy.spin()
