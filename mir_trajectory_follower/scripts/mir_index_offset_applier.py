#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray, Int32

class MirIndexOffsetApplier:
    def __init__(self):
        self.ns = rospy.get_param("~namespace", "/mur620c")
        self.offset_topic = rospy.get_param("~offset_topic", f"{self.ns}/mir_index_offset")
        self.ur_index_topic = rospy.get_param("~ur_index_topic", f"{self.ns}/path_index")
        self.out_topic = rospy.get_param("~out_topic", f"{self.ns}/path_index_modified")
        self.start_signal_topic = rospy.get_param("~start_signal_topic", "/start_condition")

        self.gain = float(rospy.get_param("~gain", 0.2))  # start with 10%
        self.rounding = rospy.get_param("~rounding", "round")  # round|floor|ceil

        self.offset_vec = None
        self.last_published = None

        self.pub = rospy.Publisher(self.out_topic, Int32, queue_size=10)

        rospy.Subscriber(self.offset_topic, Float32MultiArray, self._cb_offset, queue_size=1)
        rospy.Subscriber(self.ur_index_topic, Int32, self._cb_ur_index, queue_size=50)
        rospy.Subscriber(self.start_signal_topic, Int32, self._cb_start_signal, queue_size=1)

        rospy.loginfo("MirIndexOffsetApplier up. offset=%s ur_index=%s out=%s gain=%.3f rounding=%s",
                      self.offset_topic, self.ur_index_topic, self.out_topic, self.gain, self.rounding)

    def _cb_offset(self, msg: Float32MultiArray):
        self.offset_vec = list(msg.data)
        rospy.loginfo_throttle(5.0, "Received offset vector, len=%d", len(self.offset_vec))

    def _apply_rounding(self, x: float) -> int:
        if self.rounding == "floor":
            import math
            return int(math.floor(x))
        if self.rounding == "ceil":
            import math
            return int(math.ceil(x))
        # default: round to nearest int
        return int(round(x))

    def _cb_start_signal(self, msg: Int32):
        # Reset last published on start signal
        rospy.loginfo("Received start signal, resetting last published index.")
        self.last_published = None

    def _cb_ur_index(self, msg: Int32):
        ur_idx = int(msg.data)

        # Need offsets first
        if self.offset_vec is None or len(self.offset_vec) == 0:
            rospy.logwarn_throttle(2.0, "No offset vector yet -> publishing raw UR index (%d)", ur_idx)
            self._publish_with_monotonic_guard(ur_idx)
            return

        # Guard: index in range
        if ur_idx < 0 or ur_idx >= len(self.offset_vec):
            rospy.logwarn_throttle(
                2.0,
                "UR index %d out of range [0, %d) -> publishing raw UR index",
                ur_idx, len(self.offset_vec)
            )
            self._publish_with_monotonic_guard(ur_idx)
            return

        offset_middle = (max(self.offset_vec) + min(self.offset_vec)) / 2.0
        raw_offset = float(self.offset_vec[ur_idx + int(offset_middle)]) 
        weighted_offset = self.gain * raw_offset
        candidate = self._apply_rounding(ur_idx + weighted_offset)
        self._publish_with_monotonic_guard(candidate, ur_idx=ur_idx, raw_offset=raw_offset)
        # log index offset
        rospy.loginfo_throttle(0.5,
            "UR index: %d, raw offset: %.4f, gain: %.3f -> weighted offset: %.4f -> MiR index: %d",
            ur_idx, raw_offset, self.gain, weighted_offset, candidate
        )

    def _publish_with_monotonic_guard(self, candidate: int, ur_idx: int = None, raw_offset: float = None):
        # Enforce non-decreasing published index
        if self.last_published is not None and candidate < self.last_published:
            if ur_idx is not None and raw_offset is not None:
                rospy.logwarn(
                    "MiR index would go backwards: last=%d, new=%d (ur=%d, offset=%.4f, gain=%.3f). "
                    "Publishing last again.",
                    self.last_published, candidate, ur_idx, raw_offset, self.gain
                )
            else:
                rospy.logwarn(
                    "MiR index would go backwards: last=%d, new=%d. Publishing last again.",
                    self.last_published, candidate
                )
            candidate = self.last_published   # at least increment by 1

        self.pub.publish(Int32(data=int(candidate)))
        self.last_published = int(candidate)

def main():
    rospy.init_node("mir_index_offset_applier")
    MirIndexOffsetApplier()
    rospy.spin()

if __name__ == "__main__":
    main()
