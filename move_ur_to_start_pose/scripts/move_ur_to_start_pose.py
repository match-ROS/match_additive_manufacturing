#! /usr/bin/env python3
import rospy
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
import numpy as np
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
import sys
import tf.transformations as tr
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import DisplayTrajectory, RobotState
from moveit_msgs.msg import Constraints, JointConstraint
import math
from tf import transformations as tr
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController, LoadControllerRequest
from math import pi

class MoveManipulatorToTarget:
    def __init__(self):
        rospy.init_node('move_manipulator_to_target', anonymous=True)


        # Initialize parameters
        self.path_topic = rospy.get_param('~path_topic', '/ur_path')
        self.initial_path_index = rospy.get_param('~initial_path_index', 0)
        self.robot_name = rospy.get_param('~robot_name', 'mur620a')
        self.manipulator_base_link = rospy.get_param('~manipulator_base_link', 'UR10_r/base_link')
        self.manipulator_tcp_link = rospy.get_param('~manipulator_tcp_link', 'mur620a/UR10_r/tool0')
        self.planning_group = rospy.get_param('~planning_group', 'UR_arm_r')
        self.UR_prefix = rospy.get_param('~UR_prefix', 'UR10_r')

        param_path = f'/{self.robot_name}/{self.UR_prefix}/ur_calibrated_pose_pub_node/tcp_offset'
        self.tcp_offset = rospy.get_param(param_path, [0.0,0.0,0.0,0.0,0.0,0.0])
        self.spray_distance = rospy.get_param('~spray_distance', 0.15)  # distance from TCP to spray point along z-axis
        # remove [] if present
        if isinstance(self.tcp_offset, str):
            self.tcp_offset = self.tcp_offset.strip('[]').split(',')
        # convert to float
        self.tcp_offset = [float(i) for i in self.tcp_offset]

        # Initialize MoveIt
        roscpp_initialize(sys.argv)
        self.move_group = MoveGroupCommander(self.planning_group, ns=self.robot_name, robot_description=self.robot_name+"/robot_description")
        self.move_group.set_pose_reference_frame(self.manipulator_base_link)
        rospy.loginfo(f"MoveIt MoveGroup for {self.planning_group} initialized.")

        # Initialize the subscriber for the path
        self.path_sub = rospy.Subscriber(self.path_topic, Path, self.path_callback)
        
        # TF listener
        self.tf_listener = tf.TransformListener()

        # initialize the publisher for the target pose
        self.local_target_pose_pub = rospy.Publisher('/ur_local_target_pose', PoseStamped, queue_size=1)
        self.display_trajectory_publisher = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)

        # check if arm controller is loaded
        list_controllers_service = f'/{self.robot_name}/{self.UR_prefix}/controller_manager/list_controllers'
        print(f"Waiting for controller list on topic: {list_controllers_service}")
        try:
            rospy.wait_for_service(list_controllers_service, timeout=5)
            rospy.loginfo("Controller list service is available.")

            controllers_list = rospy.ServiceProxy(list_controllers_service, ListControllers)()
            rospy.loginfo("Controller list retrieved successfully.")
            # get arm_controller state
            arm_controller_state = [controller for controller in controllers_list.controller if controller.name == 'arm_controller']
            print(f"Arm controller state: {arm_controller_state}")
            if not arm_controller_state:
                # load the arm controller
                rospy.logwarn("Arm controller not loaded. Trying to load it.")
                rospy.wait_for_service(f'/{self.robot_name}/{self.UR_prefix}/controller_manager/load_controller')
                try:
                    load_controller_client = rospy.ServiceProxy(f'/{self.robot_name}/{self.UR_prefix}/controller_manager/load_controller', LoadController)
                    load_controller_request = LoadControllerRequest()
                    load_controller_request.name = 'arm_controller'
                    load_controller_client(load_controller_request)
                except rospy.ServiceException as e:
                    rospy.logerr(f"Failed to load arm controller: {e}")
            if arm_controller_state[0].state == 'running':
                rospy.loginfo("Arm controller is running.")
            elif arm_controller_state[0].state == 'stopped' or arm_controller_state[0].state == 'initialized':
                # switch on the arm controller
                rospy.wait_for_service(f'/{self.robot_name}/{self.UR_prefix}/controller_manager/switch_controller')
                try:
                    switch_controller_client = rospy.ServiceProxy(f'/{self.robot_name}/{self.UR_prefix}/controller_manager/switch_controller', SwitchController)
                    switch_controller_request = SwitchControllerRequest()
                    switch_controller_request.start_controllers = ['arm_controller']
                    switch_controller_request.stop_controllers = ['twist_controller']
                    switch_controller_request.strictness = 2  # Best effort
                    switch_controller_client(switch_controller_request)
                except rospy.ServiceException as e:
                    rospy.logerr(f"Failed to start arm controller: {e}")
            else:        
                rospy.logwarn(f"Arm controller is in an unexpected state: {arm_controller_state[0].state}")
        except rospy.ROSException as e:
            rospy.WARN(f"Failed to get controllers list: {e}")
            return
            

    def path_callback(self, path_msg):
        if len(path_msg.poses) == 0:
            rospy.logwarn("Received an empty path!")
            return

        # Get the first TCP pose from the path
        target_tcp_pose = path_msg.poses[self.initial_path_index]
        # the target pose is the pose of the path, we need to compute the actual tcp pose
        target_tcp_pose.pose.position.z += self.tcp_offset[2] + self.spray_distance  # add z offset
        
        # Get the current pose of the manipulator base in the map frame
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform("map", self.robot_name+"/"+self.manipulator_base_link, now, rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", self.robot_name+"/"+self.manipulator_base_link, now)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"TF error: {e}")
            return
        
        # Convert to numpy arrays for easier manipulation
        manipulator_base_position = np.array(trans)
        target_tcp_position = np.array([target_tcp_pose.pose.position.x,
                                        target_tcp_pose.pose.position.y,
                                        target_tcp_pose.pose.position.z])
        
        # Compute the target position in the manipulator’s local frame
        relative_position = target_tcp_position - manipulator_base_position
        
        # rotate the relative position to the local frame of the manipulator
        # get the rotation matrix from the quaternion
        rot_matrix = tr.quaternion_matrix(tr.quaternion_inverse(rot))
        # rotate the relative position
        relative_position = np.dot(rot_matrix[:3, :3], relative_position)

        # relative orientation around the z-axis
        target_tcp_pose = path_msg.poses[self.initial_path_index+1]
        # get path orientation
        path_orientation = tr.euler_from_quaternion([target_tcp_pose.pose.orientation.x,
                                                     target_tcp_pose.pose.orientation.y,
                                                     target_tcp_pose.pose.orientation.z,
                                                     target_tcp_pose.pose.orientation.w])
        mir_orientation = rot[2]
        

        relative_pose = [0.0,0.0,0.0,0.0,0.0,0.0]
        relative_pose[0] = relative_position[0]
        relative_pose[1] = relative_position[1]
        relative_pose[2] = relative_position[2]
        relative_pose[3] = math.pi
        relative_pose[4] = 0.0
        relative_pose[5] = -mir_orientation + path_orientation[2] + self.tcp_offset[5] + pi*0.72 + pi*1.0 # add tcp offset in rotation around z 
        relative_pose[5] = np.arctan2(np.sin(relative_pose[5]), np.cos(relative_pose[5]))  # normalize angle to [-pi, pi]
        
        # Set the target pose for MoveIt
        self.move_group.set_pose_target(relative_pose, end_effector_link=self.manipulator_tcp_link)
        local_target_pose = PoseStamped()
        local_target_pose.header.frame_id = self.manipulator_base_link
        local_target_pose.header.stamp = rospy.Time.now()
        local_target_pose.pose.position.x = relative_position[0]
        local_target_pose.pose.position.y = relative_position[1]
        local_target_pose.pose.position.z = relative_position[2]
        quat = tr.quaternion_from_euler(relative_pose[3], relative_pose[4], relative_pose[5])
        local_target_pose.pose.orientation = Quaternion(*quat)
        self.local_target_pose_pub.publish(local_target_pose)

        constraints = Constraints()
        # Ellbogen oben (z. B. nahe -2.0 rad)
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/shoulder_lift_joint",
            position=-0.5,
            tolerance_above=1.0,
            tolerance_below=1.0,
            weight=1.0
        ))
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/shoulder_pan_joint",
            position=0.0,
            tolerance_above=2.5,
            tolerance_below=0.5,
            weight=1.0
        ))
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/wrist_1_joint",
            position=-2.1,
            tolerance_above=1.1,
            tolerance_below=1.1,
            weight=1.0
        ))
        constraints.joint_constraints.append(JointConstraint(
            joint_name="UR10_r/wrist_2_joint",
            position=-1.5,
            tolerance_above=1.1,
            tolerance_below=1.1,
            weight=1.0
        ))

        self.move_group.set_path_constraints(constraints)
        
        # Plan and execute the motion
        plan_result = self.move_group.plan()

        if isinstance(plan_result, tuple):
            success = plan_result[0]  # Typically the success flag is the first item
            plan_trajectory = plan_result[1]  # The trajectory is usually the second item

            if success:
                plan_to_execute = plan_trajectory
                corrected_joint_target = None
                joint_goal_from_plan = None
                joint_traj = plan_trajectory.joint_trajectory if plan_trajectory else None

                if joint_traj and joint_traj.points:
                    final_positions = joint_traj.points[-1].positions
                    joint_goal_from_plan = dict(zip(joint_traj.joint_names, final_positions))
                else:
                    rospy.logwarn("Received an empty joint trajectory from planner.")

                active_joints = self.move_group.get_active_joints()
                if joint_goal_from_plan and active_joints:
                    last_joint_name = active_joints[-1]
                    if last_joint_name in joint_goal_from_plan:
                        q6 = joint_goal_from_plan[last_joint_name]
                        adjusted_q6 = q6
                        while adjusted_q6 > math.pi:
                            adjusted_q6 -= 2.0 * math.pi
                        while adjusted_q6 < -math.pi:
                            adjusted_q6 += 2.0 * math.pi

                        if abs(adjusted_q6 - q6) > 1e-6:
                            rospy.loginfo(f"Replanning to keep {last_joint_name} within [-pi, pi]: {q6:.3f} -> {adjusted_q6:.3f}.")
                            joint_goal_from_plan[last_joint_name] = adjusted_q6
                            corrected_joint_target = joint_goal_from_plan
                    else:
                        rospy.logwarn(f"Last joint '{last_joint_name}' not present in joint goal from plan.")
                elif not active_joints:
                    rospy.logwarn("Could not get active joints from MoveGroup.")

                if corrected_joint_target:
                    self.move_group.set_joint_value_target(corrected_joint_target)
                    corrected_plan_result = self.move_group.plan()
                    if isinstance(corrected_plan_result, tuple):
                        corrected_success = corrected_plan_result[0]
                        corrected_traj = corrected_plan_result[1]
                        if corrected_success:
                            plan_to_execute = corrected_traj
                            rospy.loginfo("Executing corrected trajectory with wrapped wrist angle.")
                        else:
                            rospy.logwarn("Replanning with corrected wrist joint failed, executing original trajectory.")
                    else:
                        rospy.logwarn("Unexpected corrected plan structure received, executing original trajectory.")

                # Publish the plan to the display path topic
                display_trajectory_publisher = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=10)
                display_trajectory = DisplayTrajectory()
                joint_state = rospy.wait_for_message(f"{self.robot_name}/joint_states", JointState)
                if joint_state is None:
                    rospy.logwarn("Failed to read joint_states for display trajectory; skipping visualization publish.")
                else:
                    robot_state = RobotState()
                    robot_state.joint_state = joint_state
                    display_trajectory.trajectory_start = robot_state

                if plan_to_execute is None:
                    rospy.logerr("Plan to execute missing despite successful planning. Aborting execution.")
                    return

                display_trajectory.trajectory = [plan_to_execute]
                display_trajectory_publisher.publish(display_trajectory)

                # Execute the motion
                self.move_group.execute(plan_to_execute, wait=True)
                rospy.loginfo("Motion executed successfully.")
                rospy.signal_shutdown("Motion executed successfully.")
            else:
                rospy.logwarn(f"Motion planning failed.")
        else:
            rospy.logwarn("Unexpected plan structure received.")



if __name__ == '__main__':
    try:
        manipulator_mover = MoveManipulatorToTarget()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
    finally:
        roscpp_shutdown()
