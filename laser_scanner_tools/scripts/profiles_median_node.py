#!/usr/bin/env python3
"""Subscribe to /profiles (PointCloud2), compute median Z in two frames and publish at 10 Hz.

This node is intended to run on the robot (mur) together with the Keyence driver.
It publishes two Float32 topics: /profiles/median_base and /profiles/median_map.
"""
import rospy
import numpy as np
import tf2_ros
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Float32


def quat_to_R(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)


class ProfilesMedianNode(object):
    def __init__(self):
        rospy.init_node('profiles_median_node', anonymous=False)

        # params
        self.base_frame = rospy.get_param('~base_frame', 'mur620c/UR10_r/base')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self._tf_timeout = rospy.Duration(secs=0, nsecs=200_000_000)

        # TF
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(secs=10))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # cache
        self._last_pts = None
        self._last_frame = None
        self._last_stamp = None

        # publishers
        self._pub_base = rospy.Publisher('/profiles/median_base', Float32, queue_size=1)
        self._pub_map = rospy.Publisher('/profiles/median_map', Float32, queue_size=1)

        # subscriber
        rospy.Subscriber('/profiles', PointCloud2, self._profiles_cb, queue_size=1)

        # timer at 10 Hz
        self._timer = rospy.Timer(rospy.Duration(secs=0, nsecs=100_000_000), self._timer_cb)

        rospy.loginfo('profiles_median_node started: publishing medians at 10 Hz')

    def _profiles_cb(self, msg: PointCloud2):
        try:
            pts = np.asarray(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float64)
        except Exception as ex:
            rospy.logwarn_throttle(5.0, 'profiles_median: failed to read points: %s' % ex)
            return
        if pts.size == 0:
            return
        if pts.shape[0] > 100000:
            pts = pts[::5, :]
        self._last_pts = pts
        self._last_frame = msg.header.frame_id
        self._last_stamp = msg.header.stamp

    def _median_in_frame(self, pts_xyz: np.ndarray, src_frame: str, stamp, target_frame: str):
        if pts_xyz is None or pts_xyz.size == 0:
            return float('nan')
        try:
            tf = self._tf_buffer.lookup_transform(target_frame, src_frame, stamp, self._tf_timeout)
        except Exception as ex:
            rospy.logwarn_throttle(5.0, 'profiles_median: TF %s -> %s unavailable: %s' % (src_frame, target_frame, ex))
            return float('nan')
        t = tf.transform.translation
        r = tf.transform.rotation
        R = quat_to_R(r.x, r.y, r.z, r.w)
        z_vals = (R[2, 0] * pts_xyz[:, 0]) + (R[2, 1] * pts_xyz[:, 1]) + (R[2, 2] * pts_xyz[:, 2]) + t.z
        if z_vals.size == 0:
            return float('nan')
        return float(np.median(z_vals))

    def _timer_cb(self, event):
        pts = self._last_pts
        if pts is None or pts.size == 0:
            return
        src = self._last_frame or ''
        stamp = self._last_stamp
        # ensure frames are strings
        base_frame = str(self.base_frame)
        map_frame = str(self.map_frame)
        med_base = self._median_in_frame(pts, src, stamp, base_frame)
        med_map = self._median_in_frame(pts, src, stamp, map_frame)

        try:
            self._pub_base.publish(float(med_base))
            self._pub_map.publish(float(med_map))
        except Exception:
            pass


if __name__ == '__main__':
    try:
        node = ProfilesMedianNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
