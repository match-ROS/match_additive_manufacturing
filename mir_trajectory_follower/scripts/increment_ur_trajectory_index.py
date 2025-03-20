#! /usr/bin/env python3
import rospy
from std_msgs.msg import Int32

# this node increments the trajectory index periodically to fake the UR robot moving along the trajectory

class IncrementUrTrajectoryIndex:
    def __init__(self):
        rospy.init_node('increment_ur_trajectory_index')
        self.trajectory_index_pub = rospy.Publisher("/path_index", Int32, queue_size=1)
        self.rate = rospy.Rate(4)
        self.trajectory_index = 0

    def run(self):
        while not rospy.is_shutdown():
            self.trajectory_index_pub.publish(self.trajectory_index)
            self.trajectory_index += 1
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = IncrementUrTrajectoryIndex()
        node.run()
    except rospy.ROSInterruptException:
        pass