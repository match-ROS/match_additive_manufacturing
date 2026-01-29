#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Pose, Twist
from tf.transformations import euler_from_quaternion

class GoToGoal:
    def __init__(self):
        rospy.init_node("go_to_goal")

        self.pose = None

        # Zielposition (anpassen)
        self.goal_x = rospy.get_param("~goal_x", 51.523826)
        self.goal_y = rospy.get_param("~goal_y", 43.065081)
        self.yaw_goal = rospy.get_param("~yaw_goal", -2.4130057)

        # Reglerparameter
        self.k_lin = 0.4
        self.k_ang = 1.0
        self.max_v = 0.4
        self.max_w = 0.8
        self.goal_tol = 0.05
        self.yaw_tol = 0.05

        rospy.Subscriber(
            "/mur620c/mir_pose_simple",
            Pose,
            self.pose_cb,
            queue_size=1
        )

        self.cmd_pub = rospy.Publisher(
            "/mur620c/cmd_vel",
            Twist,
            queue_size=1
        )

        self.rate = rospy.Rate(20)

    def pose_cb(self, msg):
        self.pose = msg

    def get_yaw(self, q):
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def run(self):
        while not rospy.is_shutdown():
            if self.pose is None:
                self.rate.sleep()
                continue

            x = self.pose.position.x
            y = self.pose.position.y
            yaw = self.get_yaw(self.pose.orientation)

            dx = self.goal_x - x
            dy = self.goal_y - y
            dist = math.hypot(dx, dy)

            cmd = Twist()

            # Phase 1: Position anfahren
            if dist > self.goal_tol:
                target_angle = math.atan2(dy, dx)
                ang_err = math.atan2(
                    math.sin(target_angle - yaw),
                    math.cos(target_angle - yaw)
                )

                cmd.linear.x = min(self.k_lin * dist, self.max_v)
                cmd.angular.z = max(
                    -self.max_w,
                    min(self.k_ang * ang_err, self.max_w)
                )

            # Phase 2: Endorientierung einstellen
            else:
                yaw_err = math.atan2(
                    math.sin(self.yaw_goal - yaw),
                    math.cos(self.yaw_goal - yaw)
                )

                if abs(yaw_err) < self.yaw_tol:
                    self.cmd_pub.publish(Twist())
                    rospy.loginfo_once("Goal position + orientation reached")
                    self.rate.sleep()
                    continue

                cmd.angular.z = max(
                    -self.max_w,
                    min(self.k_ang * yaw_err, self.max_w)
                )

            self.cmd_pub.publish(cmd)
            self.rate.sleep()


if __name__ == "__main__":
    GoToGoal().run()
