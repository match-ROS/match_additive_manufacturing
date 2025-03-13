#!/usr/bin/env python3

import rospy
from helper.ur_helper import Control_ur_helper
import math
from geometry_msgs.msg import TwistStamped

class Control_ur():
    
    def config(self):
        
        self.ur_scanner_angular_offset = rospy.get_param("~ur_scanner_angular_offset", -math.pi)
    
    
    def __init__(self):
        rospy.init_node("control_ur_node")
        Control_ur_helper(self)
        self.twist_debug_publisher = rospy.Publisher("/ur_twist_debug", TwistStamped, queue_size=1)    
        self.config()
        
    
    
    def main(self):
        pass