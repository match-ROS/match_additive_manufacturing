#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32

class LayerSpawner:

    def __init__(self):

        rospy.init_node('spawn_n_layers')
        current_layer_topic = rospy.get_param('~current_layer_topic', '/current_layer')
        number_of_layers = rospy.get_param('~number_of_layers', 100)

        for i in range(number_of_layers):
            current_layer_pub = rospy.Publisher(current_layer_topic, Int32, queue_size=1)
            current_layer_pub.publish(Int32(i))
            rospy.sleep(1.0)

        rospy.signal_shutdown("All layers spawned.")


if __name__ == '__main__':
    try:
        LayerSpawner()
    except rospy.ROSInterruptException:
        pass
