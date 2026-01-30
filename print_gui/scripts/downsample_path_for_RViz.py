#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path

DOWNSAMPLE = 10

class PathDownsampler:
    def __init__(self):
        self.pub_mir = rospy.Publisher(
            "/mur620c/mir_path_RViz", Path, queue_size=1, latch=True
        )
        self.pub_ur = rospy.Publisher(
            "/mur620c/ur_path_RViz", Path, queue_size=1, latch=True
        )

        rospy.Subscriber(
            "/mur620c/mir_path_transformed", Path, self.cb_mir, queue_size=1
        )
        rospy.Subscriber(
            "/mur620c/ur_path_transformed", Path, self.cb_ur, queue_size=1
        )

    def downsample(self, msg: Path) -> Path:
        out = Path()
        out.header = msg.header
        out.poses = msg.poses[::DOWNSAMPLE]
        return out

    def cb_mir(self, msg: Path):
        self.pub_mir.publish(self.downsample(msg))

    def cb_ur(self, msg: Path):
        self.pub_ur.publish(self.downsample(msg))

if __name__ == "__main__":
    rospy.init_node("path_downsampler_rviz")
    PathDownsampler()
    rospy.spin()
