#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import math

class VelocityCalculator:
    def __init__(self):
        rospy.init_node('velocity_tcp_publisher', anonymous=True)
        self.use_stamp = rospy.get_param("~use_stamp", True)
        self.min_pose_dt = rospy.get_param("~min_pose_dt", 0.1)  # Minimum time difference to consider for velocity calculation
        self.last_position = None
        self.last_time = None

        # Publisher for calculated velocity (in m/s)
        self.velocity_pub = rospy.Publisher('tcp_velocity', Float32, queue_size=10)
        # Subscriber for global TCP pose
        rospy.Subscriber('global_tcp_pose', PoseStamped, self.pose_callback)
        rospy.loginfo("VelocityCalculator node started, subscribing to 'global_tcp_pose' topic. use_stamp: %s", self.use_stamp)

    def pose_callback(self, msg):
        # Get the appropriate time stamp
        if self.use_stamp:
            current_time = msg.header.stamp.to_sec()
        else:
            current_time = rospy.Time.now().to_sec()

        current_position = msg.pose.position

        if self.last_position is None or self.last_time is None:
            self.last_position = current_position
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        if dt <= 0.0:
            self.last_position = current_position
            self.last_time = current_time
            return
        if dt < self.min_pose_dt:
            return

        # Calculate displacement
        dx = current_position.x - self.last_position.x
        dy = current_position.y - self.last_position.y
        dz = current_position.z - self.last_position.z
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Compute velocity (m/s)
        velocity = distance / dt

        # Publish the velocity
        self.velocity_pub.publish(Float32(velocity))
        rospy.loginfo("Published velocity: %.3f m/s", velocity)

        # Update last position and time
        self.last_position = current_position
        self.last_time = current_time

if __name__ == '__main__':
    try:
        VelocityCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass