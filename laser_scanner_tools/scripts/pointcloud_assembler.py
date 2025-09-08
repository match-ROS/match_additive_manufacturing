#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2 as pc2
import tf2_ros
import tf2_py as tf2
from tf.transformations import quaternion_matrix

try:
    # laser_geometry is optional; only needed for LaserScan input
    from laser_geometry import laser_geometry as lg
except Exception:
    lg = None


class PointCloudAssembler(object):
    def __init__(self):
        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/scan')
        self.input_type = rospy.get_param('~input_type', 'pointcloud2').strip().lower()  # 'pointcloud2' or 'laserscan'
        self.target_frame = rospy.get_param('~target_frame', 'mur620c/base_footprint')
        self.source_frame_override = rospy.get_param('~source_frame', '')  # optional override
        self.output_topic = rospy.get_param('~output_topic', 'assembled_pointcloud')
        self.max_points = int(rospy.get_param('~max_points', 5000000))

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Accumulator
        self._active = False
        self._points = []  # list of (x,y,z)

        # Publisher (latched so consumers can get the last assembled cloud)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1, latch=True)

        # Services
        self.start_srv = rospy.Service('~start_acquisition', Trigger, self._on_start)
        self.stop_srv = rospy.Service('~stop_acquisition', Trigger, self._on_stop)

        # Subscriber based on input type
        if self.input_type == 'laserscan':
            if lg is None:
                rospy.logerr('input_type=laserscan requires laser_geometry. Please install it or switch to pointcloud2 input.')
                raise rospy.ROSInitException('laser_geometry not available')
            self._projector = lg.LaserProjection()
            self.sub = rospy.Subscriber(self.input_topic, LaserScan, self._scan_cb, queue_size=10)
        elif self.input_type == 'pointcloud2':
            self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self._cloud_cb, queue_size=10)
        else:
            rospy.logerr("Unknown input_type '%s'. Use 'pointcloud2' or 'laserscan'." % self.input_type)
            raise rospy.ROSInitException('invalid input_type')

        rospy.loginfo('PointCloudAssembler ready. Subscribed to %s (%s). Target frame: %s. Output: %s',
                      self.input_topic, self.input_type, self.target_frame, self.output_topic)

    # --- Services ---
    def _on_start(self, _req):
        self._points = []
        self._active = True
        msg = 'Acquisition started; buffers cleared.'
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    def _on_stop(self, _req):
        self._active = False
        count = len(self._points)
        if count == 0:
            msg = 'Acquisition stopped; no points collected.'
            rospy.logwarn(msg)
            return TriggerResponse(success=True, message=msg)

        # Publish assembled cloud
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.target_frame
        cloud = pc2.create_cloud_xyz32(header, self._points)
        self.pub.publish(cloud)
        msg = 'Acquisition stopped; published %d points.' % count
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)

    # --- Callbacks ---
    def _scan_cb(self, scan_msg):
        if not self._active:
            return
        try:
            cloud = self._project_to_cloud(scan_msg)
            self._accumulate_cloud(cloud)
        except Exception as e:
            rospy.logwarn('Failed to process LaserScan: %s', e)

    def _cloud_cb(self, cloud_msg):
        if not self._active:
            return
        try:
            cloud = self._transform_cloud(cloud_msg)
            self._accumulate_cloud(cloud)
        except Exception as e:
            rospy.logwarn('Failed to process PointCloud2: %s', e)

    # --- Helpers ---
    def _project_to_cloud(self, scan_msg):
        # Project LaserScan to a PointCloud2 in scan frame
        cloud_local = self._projector.projectLaser(scan_msg)
        return self._transform_cloud(cloud_local)

    def _transform_cloud(self, cloud_msg):
        # Optionally override source frame
        src_frame = self.source_frame_override.strip() or cloud_msg.header.frame_id
        if not src_frame:
            raise ValueError('Source frame is empty and header.frame_id not set')

        # Build a new header if we override frame
        if src_frame != cloud_msg.header.frame_id:
            # Create a shallow copy with overridden frame
            cloud_msg = PointCloud2(
                header=Header(stamp=cloud_msg.header.stamp, frame_id=src_frame),
                height=cloud_msg.height,
                width=cloud_msg.width,
                fields=cloud_msg.fields,
                is_bigendian=cloud_msg.is_bigendian,
                point_step=cloud_msg.point_step,
                row_step=cloud_msg.row_step,
                data=cloud_msg.data,
                is_dense=cloud_msg.is_dense,
            )

        # Lookup transform and apply
        stamp = cloud_msg.header.stamp
        if stamp.to_sec() == 0.0:
            # If stamp is zero, use latest available
            stamp = rospy.Time(0)
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, src_frame, stamp, rospy.Duration(0.5))
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            raise RuntimeError('TF transform %s -> %s failed: %s' % (src_frame, self.target_frame, e))
        # Manually transform x,y,z only
        return self._transform_cloud_fallback(cloud_msg, tf)

    def _transform_cloud_fallback(self, cloud_msg, tf_msg):
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z

        pts_out = []
        for x, y, z in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            vx = T[0, 0] * x + T[0, 1] * y + T[0, 2] * z + T[0, 3]
            vy = T[1, 0] * x + T[1, 1] * y + T[1, 2] * z + T[1, 3]
            vz = T[2, 0] * x + T[2, 1] * y + T[2, 2] * z + T[2, 3]
            pts_out.append((vx, vy, vz))

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.target_frame
        return pc2.create_cloud_xyz32(header, pts_out)

    def _accumulate_cloud(self, cloud_msg):
        # Read XYZ points and append
        added = 0
        for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            self._points.append((float(p[0]), float(p[1]), float(p[2])))
            added += 1
            if len(self._points) >= self.max_points:
                rospy.logwarn('Reached max_points=%d; further points will be ignored until stop.' % self.max_points)
                self._active = False
                break
        rospy.logdebug('Accumulated %d points (total=%d)', added, len(self._points))


def main():
    rospy.init_node('pointcloud_assembler')
    _ = PointCloudAssembler()
    rospy.spin()


if __name__ == '__main__':
    main()
