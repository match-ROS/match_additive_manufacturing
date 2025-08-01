#! /usr/bin/env python3

import rospy
from dynamixel_workbench_msgs.srv import DynamixelCommand
from std_msgs.msg import Int16
import subprocess
from ur_dashboard_msgs.srv import GetSafetyMode

class ServoDriver():
    def __init__(self):
        # Give serial read/write permission:
        # bashCommand = "sudo chmod 666 /dev/ttyUSB2" #rosrun dynamixel_workbench_controllers find_dynamixel /dev/ttyUSB0
        # process.communicate()
        
        rospy.loginfo("Starting servo driver with multiple motors")
        rospy.init_node("servo_driver_node", anonymous=True)
        
        # Load service topics from parameters
        servo_service_topic = rospy.get_param("~servo_service_topic", "/dynamixel_workbench/dynamixel_command")
        ur_safety_service_topic = rospy.get_param("~ur_safety_service_topic", "/mur620c/UR10_r/ur_hardware_interface/dashboard/get_safety_mode")
        
        # Wait for services
        rospy.wait_for_service(servo_service_topic)
        rospy.loginfo("Servo service found on: %s", servo_service_topic)
        rospy.sleep(0.1)
        self.servo_command_service = rospy.ServiceProxy(servo_service_topic, DynamixelCommand)
        
        rospy.wait_for_service(ur_safety_service_topic)
        self.ur_safety_mode_service = rospy.ServiceProxy(ur_safety_service_topic, GetSafetyMode)
        
        # Load motor configurations
        motor_configs = rospy.get_param("~dynamixel_motors", [
            {"name": "left", "id": 1},
            {"name": "right", "id": 2}
        ])
        self.motors = {}       # stores motor info: motor id, etc.
        self.last_command = {} # stores last command per motor

        for config in motor_configs:
            motor = config["name"]
            motor_id = config["id"]
            self.motors[motor] = {"id": motor_id}
            self.last_command[motor] = 0
            topic="servo_target_pos_"+motor
            rospy.Subscriber(topic, Int16, self.generate_callback(motor))
            rospy.loginfo("Subscribed to [%s] for motor '%s' (ID %d)", topic, motor, motor_id)
            
        # Uncomment below to enable periodic safety checks
        # self.safety_loop()
        rospy.spin()
    
    def generate_callback(self, motor):
        # Creates a closure that captures motor name and corresponding motor id
        def callback(position_command):
            rospy.loginfo("Received command [%d] for motor '%s'", position_command.data, motor)
            self.last_command[motor] = position_command.data
            motor_id = self.motors[motor]["id"]
            try:
                res = self.servo_command_service("", motor_id, "Goal_Position", position_command.data)
                rospy.loginfo("Motor '%s' (ID %d) service response: %s", motor, motor_id, res)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed for motor '%s' (ID %d): %s", motor, motor_id, e)
        return callback
    
    def safety_loop(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            try:
                safety_status = self.ur_safety_mode_service()
            except rospy.ServiceException as e:
                rospy.logerr("Safety mode service call failed: %s", e)
                rate.sleep()
                continue
            
            for motor, command in self.last_command.items():
                if safety_status.safety_mode.mode != 1 and command != 0:
                    rospy.logwarn("Safety mode active. Resetting motor '%s' command to 0", motor)
                    zero_command = Int16()
                    zero_command.data = 0
                    motor_id = self.motors[motor]["id"]
                    try:
                        res = self.servo_command_service("", motor_id, "Goal_Position", 0)
                        rospy.loginfo("Motor '%s' (ID %d) reset response: %s", motor, motor_id, res)
                        self.last_command[motor] = 0
                    except rospy.ServiceException as e:
                        rospy.logerr("Reset service call failed for motor '%s' (ID %d): %s", motor, motor_id, e)
            rate.sleep()

if __name__=="__main__":
    ServoDriver()