#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
import tf.transformations as tft

class URJoyTwist:
    def __init__(self):
        rospy.init_node("ur_joy_twist")

        self.scale_x = rospy.get_param("~scale_x", 0.1)
        self.scale_y = rospy.get_param("~scale_y", 0.1)
        self.alpha   = rospy.get_param("~alpha", 0.1)
        self.timeout = rospy.get_param("~timeout", 1.5)

        self.theta = math.radians(14.0)
        self.Rz = tft.rotation_matrix(self.theta, (0, 0, 1))[:3, :3]

        self.x_filt = 0.0
        self.y_filt = 0.0
        self.last_cmd_time = rospy.Time(0)

        self.pub = rospy.Publisher(
            "/mur620d/UR10_r/twist_controller/command_collision_free",
            Twist,
            queue_size=1
        )

        rospy.Subscriber("/joy", Joy, self.joy_cb)
        rospy.Timer(rospy.Duration(0.01), self.timer_cb)  # 100 Hz

    def joy_cb(self, msg):
        x_raw = self.scale_x * msg.axes[4]
        y_raw = self.scale_y * msg.axes[3]

        self.x_filt = self.alpha * x_raw + (1 - self.alpha) * self.x_filt
        self.y_filt = self.alpha * y_raw + (1 - self.alpha) * self.y_filt

        if x_raw == 0.0:
            self.x_filt = 0.0
        if y_raw == 0.0:
            self.y_filt = 0.0  
    
        self.last_cmd_time = rospy.Time.now()

    def timer_cb(self, _):
        twist = Twist()

        if (rospy.Time.now() - self.last_cmd_time).to_sec() < self.timeout:
            v = self.Rz @ [self.x_filt, self.y_filt, 0.0]
            twist.linear.x = v[0]
            twist.linear.y = v[1]
        else:
            twist.linear.x = 0.0
            twist.linear.y = 0.0

        self.pub.publish(twist)

if __name__ == "__main__":
    URJoyTwist()
    rospy.spin()
