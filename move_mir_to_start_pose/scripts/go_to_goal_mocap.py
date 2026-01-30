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
        self.goal_x = rospy.get_param("~goal_x", 48.821396 )
        self.goal_y = rospy.get_param("~goal_y", 41.657644)
        self.yaw_goal = rospy.get_param("~yaw_goal", 2.1024978)

        # Reglerparameter
        self.k_lin = 0.2
        self.k_ang = 0.2
        self.max_v = 0.2
        self.max_w = 0.4
        self.goal_tol = 0.05
        self.yaw_tol = 0.05

        self.state = "ALIGN"
        self.drive_target_dist = None
        self.drive_start_x = None
        self.drive_start_y = None
        self.drive_fraction = 0.8


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

    def traveled_distance(self, x, y):
        return math.hypot(x - self.drive_start_x, y - self.drive_start_y)


    def get_yaw(self, q):
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Shutting down go_to_goal node.") 
        rospy.sleep(1)  
        rospy.signal_shutdown("Goal reached")


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

            # =======================
            # STATE: ALIGN TO TARGET
            # =======================
            if self.state == "ALIGN":
                if dist < self.goal_tol:
                    self.state = "FINAL_ALIGN"
                    continue

                target_angle = math.atan2(dy, dx)
                ang_err = math.atan2(
                    math.sin(target_angle - yaw),
                    math.cos(target_angle - yaw)
                )

                if abs(ang_err) < self.yaw_tol:
                    self.drive_target_dist = self.drive_fraction * dist
                    self.drive_start_x = x
                    self.drive_start_y = y
                    self.state = "DRIVE"
                else:
                    cmd.angular.z = max(
                        -self.max_w,
                        min(self.k_ang * ang_err, self.max_w)
                    )

            # =======================
            # STATE: DRIVE STRAIGHT
            # =======================
            elif self.state == "DRIVE":
                traveled = self.traveled_distance(x, y)

                if traveled >= self.drive_target_dist:
                    self.state = "ALIGN"
                else:
                    cmd.linear.x = self.max_v

            # =======================
            # STATE: FINAL ALIGNMENT
            # =======================
            elif self.state == "FINAL_ALIGN":
                yaw_err = math.atan2(
                    math.sin(self.yaw_goal - yaw),
                    math.cos(self.yaw_goal - yaw)
                )

                if abs(yaw_err) < self.yaw_tol:
                    self.cmd_pub.publish(Twist())
                    rospy.loginfo("Goal position + orientation reached")
                    self.shutdown()
                    return
                else:
                    cmd.angular.z = max(
                        -self.max_w,
                        min(self.k_ang * yaw_err, self.max_w)
                    )

            self.cmd_pub.publish(cmd)
            self.rate.sleep()



if __name__ == "__main__":
    GoToGoal().run()
