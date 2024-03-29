#! /usr/bin/env python3
import rospy
import math
import numpy as np
import tf
from mirX import mirX
from mirY import mirY

from toolX import toolX
from toolY import toolY
from toolZ import toolZ

from timeStamp import timeStamp

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

mir_x= mirX()
mir_y= mirY()

wall_x = toolX()
wall_y = toolY()
wall_z = toolZ()

time_stamps = timeStamp()

#print(mir_y)

rospy.init_node("parse_path_node")
mir_pub = rospy.Publisher("/mir_path", Path, queue_size= 1, latch=True)
ur_pub = rospy.Publisher("/ur_path", Path, queue_size= 1, latch=True)


mir_path = Path()
mir_path_point = PoseStamped()
mir_path.header.frame_id = "map"
mir_path.header.stamp = rospy.Time.now()
ur_path = Path()
ur_path_point = PoseStamped()
ur_path.header.frame_id = "map"
ur_path.header.stamp = rospy.Time.now()


mir_path.poses = [PoseStamped() for i in range(len(mir_x)-1)] 
ur_path.poses = [PoseStamped() for i in range(len(wall_x)-1)] 

# reverse mir path
# for i in range(0,len(mir_x)):
#     mir_path.poses[i].pose.position.x = mir_x[len(mir_x)-i-1]
#     mir_path.poses[i].pose.position.y = mir_y[len(mir_y)-i-1]

#     orientation = math.atan2(mir_y[len(mir_y)-i-2]-mir_y[len(mir_y)-i-1], mir_x[len(mir_x)-i-2]-mir_x[len(mir_x)-i-1])
#     q = tf.transformations.quaternion_from_euler(0, 0, orientation)
#     mir_path.poses[i].pose.orientation.x = q[0]
#     mir_path.poses[i].pose.orientation.y = q[1]
#     mir_path.poses[i].pose.orientation.z = q[2]
#     mir_path.poses[i].pose.orientation.w = q[3]



for i in range(0,len(mir_x)-1): #len(robot0_xhat)
    mir_path_point.pose.position.x = mir_x[i]
    mir_path_point.pose.position.y = mir_y[i]

    if mir_x[i] == 0.0 or mir_y[i] == 0.0:
        print("mir_path_point.pose.position.x = 0.0")

    mir_path_point.header.frame_id = "map"
    mir_path_point.header.stamp = rospy.Time.now()

    # add orientation
    orientation = math.atan2(mir_y[i+1]-mir_y[i], mir_x[i+1]-mir_x[i])
    q = tf.transformations.quaternion_from_euler(0, 0, orientation)
    mir_path_point.pose.orientation.x = q[0]
    mir_path_point.pose.orientation.y = q[1]
    mir_path_point.pose.orientation.z = q[2]
    mir_path_point.pose.orientation.w = q[3]
    
    mir_path.poses[i].pose.position.x = mir_path_point.pose.position.x
    mir_path.poses[i].pose.position.y = mir_path_point.pose.position.y
    mir_path.poses[i].pose.orientation.x = mir_path_point.pose.orientation.x
    mir_path.poses[i].pose.orientation.y = mir_path_point.pose.orientation.y
    mir_path.poses[i].pose.orientation.z = mir_path_point.pose.orientation.z
    mir_path.poses[i].pose.orientation.w = mir_path_point.pose.orientation.w
    
    mir_path.poses[i].header.stamp = rospy.Duration(time_stamps[i])


for i in range(0,len(wall_x)-1): #len(robot0_xhat)
    ur_path_point.pose.position.x = wall_x[i]
    ur_path_point.pose.position.y = wall_y[i]
    ur_path_point.pose.position.z = wall_z[i]
    
    # add orientation
    orientation = math.atan2(wall_y[i+1]-wall_y[i], wall_x[i+1]-wall_x[i])
    q = tf.transformations.quaternion_from_euler(0, 0, orientation)
    
    #  rotate around x so that the gripper is pointing down
    q_rot = tf.transformations.quaternion_from_euler(np.pi, 0, 0)
    q_ur=tf.transformations.quaternion_multiply(q_rot, q)
    q_rot = tf.transformations.quaternion_from_euler(0, 0, np.pi/2)
    q_ur=tf.transformations.quaternion_multiply(q_rot, q_ur)

    ur_path.poses[i].pose.position.x = ur_path_point.pose.position.x
    ur_path.poses[i].pose.position.y = ur_path_point.pose.position.y
    ur_path.poses[i].pose.position.z = ur_path_point.pose.position.z
    ur_path.poses[i].pose.orientation.x = q_ur[0]
    ur_path.poses[i].pose.orientation.y = q_ur[1]
    ur_path.poses[i].pose.orientation.z = q_ur[2]
    ur_path.poses[i].pose.orientation.w = q_ur[3]
    
    ur_path.poses[i].header.stamp = rospy.Duration(time_stamps[i])
    

rospy.sleep(1)
mir_pub.publish(mir_path)
rospy.sleep(1)
ur_pub.publish(ur_path)
rospy.loginfo("paths published")
rospy.sleep(1)


