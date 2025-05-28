#!/usr/bin/env python3

# uses the roslaunch API to enable all UR robots

import rospy
import roslaunch



class EnableAllURs:
    
    def config(self):
        self.robot_names = rospy.get_param('~robot_names', ['mur620a', 'mur620b','mur620c','mur620d'])
        self.UR_prefixes = rospy.get_param('~UR_prefixes', ['UR10_l', 'UR10_r'])
        self.node_name = rospy.get_param('~node_name', 'UR_enable')
        self.launch_pkg = rospy.get_param('~launch_pkg', 'ur_utilities')
        self.target_position_name = rospy.get_param('~target_position_name', 'handling_position')
        self.move_group_names = rospy.get_param('~move_group_names', ['UR_arm_l', 'UR_arm_r'])

    def __init__(self):
        self.config()
        self.start_move_UR_to_home_pose()


    def start_move_UR_to_home_pose(self):
        for robot_name in self.robot_names:
            for UR_prefix in self.UR_prefixes:
                topic = "/" + robot_name + "/" + UR_prefix + "/ur_hardware_interface"
                print("Enabling " + topic)
                namespace = "/" + robot_name + "/" + UR_prefix + "/"
                process = self.launch_ros_node(self.node_name, self.launch_pkg, self.node_name + '.py', namespace, '' , ur_hardware_interface_topic=topic)
                # check if the node is still running
                while process.is_alive() and not rospy.is_shutdown():
                    rospy.sleep(1)

        # shutdown node
        rospy.signal_shutdown('All UR robots moved to start position')


    def launch_ros_node(self,node_name, package_name, node_executable, namespace="/", node_args="", **params):
        # get param names from kwargs
        param_names = params.keys()
        # set params on param server
        print("Setting params for node: " + namespace + node_name)
        for param_name in param_names:
            print("Setting param: " + namespace + node_name + "/" + param_name + " to " + str(params[param_name]))
            rospy.set_param(namespace + node_name + "/" + param_name, params[param_name])

        package = package_name
        executable = node_executable
        name = node_name
        node = roslaunch.core.Node(package=package, node_type=executable, name=name, namespace=namespace,
                                        machine_name=None, args=node_args, output="screen")
        
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)
        return process




if __name__ == '__main__':
    rospy.init_node('enable_all_URs')
    EnableAllURs()
    rospy.spin()





