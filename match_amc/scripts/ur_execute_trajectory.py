#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped, Pose, Twist, PoseWithCovarianceStamped
from tf import transformations
import math
import sys
import moveit_commander
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
import tf
from sensor_msgs.msg import JointState
#import csv

from match_lib.robot_mats.jacobians.jacobian_ur_16_eef import getJacobianUr16_base_link_inertiaUr16_wrist_3_link as getJacobian
from match_lib.match_robots import Joints
class ur_velocity_controller():
    
    
    def __init__(self):
        self.config()   # load parameters
        rospy.Subscriber('ur_trajectory', Path, self.ur_trajectory_cb)
        rospy.Subscriber('tool0_pose', Pose, self.ur_pose_cb)
        self.joint_obj = Joints()
        
        self.target_vel_mir = TwistStamped()
        #rospy.Subscriber('mobile_base_controller/cmd_vel', Twist, self.mir_vel_cb)
        rospy.Subscriber('ground_truth', Odometry, self.mir_pose_cb)
        rospy.Subscriber('tool0_pose', Pose, self.ur_local_pose_cb)
        
        # Subscribers Not in use:
        rospy.Subscriber("ur_path", Path, self.path_cb)
        rospy.Subscriber('mir_trajectory', Path, self.mir_trajectory_cb)


        self.joint_group_vel_pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
        self.test_pub = rospy.Publisher("/joint_group_vel_controller/command_test", Float64MultiArray, queue_size=1)
        self.test_pub2 = rospy.Publisher("/test_publish", Float64, queue_size=1)
        self.pose_broadcaster = tf.TransformBroadcaster()
        
        #Moveit - config
        moveit_commander.roscpp_initialize(sys.argv)
        # self.group = moveit_commander.MoveGroupCommander(group_name, ns=ns, robot_description= ns+"/robot_description",wait_for_servers=5.0)

        #self.reset()
   
    def transf_velocity_world_to_mirbase(self, final_tcp_vel_world):
        """transform velocity in world coordinates to velocity in mir coordinates. (Approx. the same as in UR coordinates)

        Args:
            final_tcp_vel_world (list[float]): [v_x, v_y] velocity of TCP in world coordinates

        Returns:
            list[float]: [v_x_mir, v_y_mir] velocity of TCP in MIR-coordinates
        """
        mir_orientation = transformations.euler_from_quaternion([self.mir_pose.orientation.x, self.mir_pose.orientation.y, self.mir_pose.orientation.z, self.mir_pose.orientation.w])
        mir_rot_about_z = mir_orientation[2]
        
        v_x_mir = -math.cos(mir_rot_about_z) * final_tcp_vel_world[0] - math.sin(math.pi - mir_rot_about_z) * final_tcp_vel_world[1]
        v_y_mir = math.sin(mir_rot_about_z) * final_tcp_vel_world[0] + math.cos(math.pi - mir_rot_about_z) * final_tcp_vel_world[1]
        #wenn die Verdrehung von mir zu urbase von 0,26grad nicht betrachtet wird ist die korrekturgeschwindigkeit im MiR-KS gleich der im UR-KS
        return [v_x_mir, v_y_mir]

    
    def differential_inverse_kinematics_ur(self, velocities):
        """Calculate velocities of joints from cartesian TCP-Velocities

        Args:
            velocities (list[float]): dim6

        Returns:
            Float64MultiArray: joint_group_vel
        """
        joint_group_vel = Float64MultiArray()
        # current_joints = self.joint_states#self.group.get_current_joint_values()
        # # jm = np.array(self.group.get_jacobian_matrix(current_joints))
        # jm = getJacobian(current_joints)
        jm = self.joint_obj.getJacobian()
        jacobian_matrix = np.matrix(jm)
        jacobian_inverse = np.linalg.inv(jacobian_matrix)
        
        cartesian_velocities_matrix = np.matrix.transpose(np.matrix(velocities))
          
        joint_velocities = jacobian_inverse @ cartesian_velocities_matrix
        joint_group_vel.data = (np.asarray(joint_velocities)).flatten()
        
        return joint_group_vel


    def position_controller(self, target_pos_x, target_pos_y,target_pos_z, actual_pos_x, actual_pos_y, actual_pos_z, K_p=0.5):
        """Proportional-controller. Calculates error in x,y and returns controller-output

        Args:
            target_pos_x (float): goal pos
            target_pos_y (float): goal pos
            actual_pos_x (float): current pos
            actual_pos_y (float): current pos
            K_p (float, optional): Proportional Term. Defaults to 0.1.

        Returns:
            (K_p*e_x, K_p*e_y, distance, e_x, e_y)
        """  
        
        e_x = target_pos_x - actual_pos_x   
        e_y = target_pos_y - actual_pos_y
        e_z = target_pos_z - actual_pos_z
        return K_p*e_x, K_p*e_y, K_p*e_z    
        

        
    def run(self):
        """calculates joint velocities by cartesian position error and trajectory velocity. Takes current MiR velocity into account. Trajectory velocity proportional to control rate.
        """
        ### wait for UR to continue ###
        while not rospy.is_shutdown() and not rospy.get_param("/state_machine/follow_trajectory", False):
            rospy.sleep(0.01)
            pass

        rospy.loginfo("Starting position controller")
        rate = rospy.Rate(self.control_rate)
        listener = tf.TransformListener()
        listener.waitForTransform("map", "mur216/UR16/tool0", rospy.Time(), rospy.Duration(4.0))
        index = 0
        while not rospy.is_shutdown() and index < len(self.trajectorie[3]):
            (trans,rot) = listener.lookupTransform('map','mur216/UR16/tool0', rospy.Time(0))
            set_pose_x      = self.trajectorie[0][index]
            set_pose_y      = self.trajectorie[1][index]
            set_pose_z      = self.trajectorie[2][index]
            set_pose_phi    = self.trajectorie[3][index]
            v_target        = self.trajectorie[4][index] * self.control_rate
            w_target        = self.trajectorie[5][index] * self.control_rate
            
            #position controller
            u_x, u_y, u_z = self.position_controller(set_pose_x, set_pose_y, set_pose_z, trans[0], trans[1], trans[2])
            
            
            # influence of mir velocity on tcp velocity
            dist_mir_ur = math.sqrt(pow(self.ur_local_pose.position.x, 2) + pow(self.ur_local_pose.position.y, 2))
            angle_mir_ur = math.atan2(self.ur_local_pose.position.y, self.ur_local_pose.position.x)
            #vel_tcp = self.target_vel_mir.twist.linear.x + self.target_vel_mir.twist.angular.z * dist_mir_ur
            vel_tcp = self.mir_trajectorie[0][index] + self.mir_trajectorie[2][index] * dist_mir_ur
            vel_tcp_x = vel_tcp * math.cos(angle_mir_ur)
            vel_tcp_y = vel_tcp * math.sin(angle_mir_ur)
            
            angle_mir_world = tf.transformations.euler_from_quaternion(rot)
            ur_vel_x = (self.trajectorie[0][index+1] - self.trajectorie[0][index]) * self.control_rate
            ur_vel_y = (self.trajectorie[1][index+1] - self.trajectorie[1][index]) * self.control_rate
            vel_tcp_feed_forward_x = ur_vel_x * math.cos(angle_mir_world[2]) - ur_vel_y * math.sin(angle_mir_world[2])
            vel_tcp_feed_forward_y = ur_vel_x * math.sin(angle_mir_world[2]) + ur_vel_y * math.cos(angle_mir_world[2])
          
            final_tcp_vel_world = [u_x-vel_tcp_x+vel_tcp_feed_forward_x, u_y-vel_tcp_y+vel_tcp_feed_forward_y, u_z]
            final_tcp_vel_mir_base = self.transf_velocity_world_to_mirbase(final_tcp_vel_world)
            
            #TCP velocity in ur_base_link
            tcp_vel_ur = [final_tcp_vel_mir_base[0], final_tcp_vel_mir_base[1], u_z, 0, 0, 0]

            joint_group_vel = self.differential_inverse_kinematics_ur(tcp_vel_ur)
            #rospy.loginfo("joint_group_vel: " + str(joint_group_vel.data)+"\nfor jointstates: "+str(self.joint_obj.q))

            #publish joint velocities
            self.target_pose_broadcaster([set_pose_x,set_pose_y,set_pose_z,set_pose_phi])
            self.joint_group_vel_pub.publish(joint_group_vel)

            
            index += 1

            rate.sleep()
            
    
    def target_pose_broadcaster(self,target_pose):
        frame_id = "tool0_target"
        self.pose_broadcaster.sendTransform((target_pose[0], target_pose[1], target_pose[2]),
                     transformations.quaternion_from_euler(0, 0, target_pose[3]),
                     rospy.Time.now(), frame_id, "map")
    
    
    def ur_trajectory_cb(self,Path):
        """Vorgegebene Trajektorie
        TODO: was passiert bei Neuvorgabe Trajektorie?
            in run(): springt von idx i der alten auf idx i der neuen Trajektorie?
        """
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        trajectory_phi = []
        path_len = len(Path.poses)
        for i in range(0,path_len-1):
            trajectory_x.append(Path.poses[i].pose.position.x)
            trajectory_y.append(Path.poses[i].pose.position.y)
            trajectory_z.append(Path.poses[i].pose.position.z)
            phi = math.atan2(Path.poses[i+1].pose.position.y-Path.poses[i].pose.position.y,Path.poses[i+1].pose.position.x-Path.poses[i].pose.position.x)
            trajectory_phi.append(phi)
        
        trajectory_v = [0.0]
        trajectory_w = [0.0]
        for i in range(1,path_len-2):
            trajectory_v.append(math.sqrt((trajectory_x[i+1]-trajectory_x[i])**2 + (trajectory_y[i+1]-trajectory_y[i])**2 ))
            trajectory_w.append(trajectory_phi[i+1]-trajectory_phi[i])

        self.trajectorie = [trajectory_x, trajectory_y, trajectory_z, trajectory_phi, trajectory_v, trajectory_w]
        rospy.loginfo("ur trajectory received")
                  
 
    def tcp_pose_cb(self, data):
        """Actual tcp pose. Topic not published?
        """
        self.tcp_pose = data
        
    def mir_pose_cb(self, data=Odometry):
        """Nur fuer orientation. Transformationen zwischen v_x_mir und v_x_world.
        """
        self.mir_pose = data.pose.pose
        self.target_vel_mir.twist=data.twist.twist

        
    def ur_pose_cb(self, data):
        """Verwendet in "get_distance_mir_ur() TODO: tcp_pose jetzt hier mit definiert"
        """
        self.ur_pose = data
        self.tcp_pose = data

    def ur_local_pose_cb(self, data:Pose):
        self.ur_local_pose = data


    def path_cb(self, data):
        """Wird nicht verwendet
        """
        self.ur_path = data
    
    def mir_trajectory_cb(self,Path):
        """Nicht gebraucht, da dies vorgegebener Pfad, stattdessen wahreer Pfad

        Args:
            Path (_type_): _description_
        """
        trajectory_x = []
        trajectory_y = []
        trajectory_phi = []
        path_len = len(Path.poses)
        for i in range(0,path_len-1):
            trajectory_x.append(Path.poses[i].pose.position.x)
            trajectory_y.append(Path.poses[i].pose.position.y)
            phi = math.atan2(Path.poses[i+1].pose.position.y-Path.poses[i].pose.position.y,Path.poses[i+1].pose.position.x-Path.poses[i].pose.position.x)
            trajectory_phi.append(phi)
        
        trajectory_v = [0.0]
        trajectory_w = [0.0]
        for i in range(1,path_len-2):
            trajectory_v.append(math.sqrt((trajectory_x[i+1]-trajectory_x[i])**2 + (trajectory_y[i+1]-trajectory_y[i])**2 ))
            trajectory_w.append(trajectory_phi[i+1]-trajectory_phi[i])

        self.mir_trajectorie = [trajectory_v, trajectory_w]
        rospy.loginfo("mir trajectory received")

    def config(self):
        self.control_rate = rospy.get_param('~control_rate', 100)
        
if __name__ == "__main__":
    rospy.init_node("ur_velocity_controller")
    velocity = ur_velocity_controller()
    velocity.run()