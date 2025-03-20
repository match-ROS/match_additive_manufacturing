#! /usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion

class PurePursuitRLCorrection:
    def __init__(self):
        rospy.init_node('rl_pure_pursuit_correction', anonymous=True)
        
        # Subscriptions
        rospy.Subscriber('/mir_actual_pose', PoseStamped, self.actual_pose_callback)
        rospy.Subscriber('/mir_target_pose', PoseStamped, self.target_pose_callback)
        rospy.Subscriber('/layer_progress', Float32, self.layer_progress_callback)
        
        # Publisher
        self.cmd_vel_offset_pub = rospy.Publisher('/RL_cmd_vel_offset', Twist, queue_size=10)
        
        # Robot state
        self.actual_pose = None
        self.target_pose = None
        self.layer_progress = 0.0
        
        rospy.loginfo("RL Pure Pursuit Correction Node Started")
        self.run()
    
    def actual_pose_callback(self, msg):
        self.actual_pose = msg
    
    def target_pose_callback(self, msg):
        self.target_pose = msg
    
    def layer_progress_callback(self, msg):
        self.layer_progress = msg.data
    
    def compute_errors(self):
        if self.actual_pose is None or self.target_pose is None:
            return None, None
        
        # Extract positions
        x_r, y_r = self.actual_pose.pose.position.x, self.actual_pose.pose.position.y
        x_t, y_t = self.target_pose.pose.position.x, self.target_pose.pose.position.y
        
        # Compute lateral error (signed distance to trajectory point)
        lateral_error = np.sqrt((x_r - x_t)**2 + (y_r - y_t)**2)
        
        # Compute heading error
        quat_r = self.actual_pose.pose.orientation
        quat_t = self.target_pose.pose.orientation
        _, _, yaw_r = euler_from_quaternion([quat_r.x, quat_r.y, quat_r.z, quat_r.w])
        _, _, yaw_t = euler_from_quaternion([quat_t.x, quat_t.y, quat_t.z, quat_t.w])
        heading_error = yaw_r - yaw_t
        
        return lateral_error, heading_error
    
    def run(self):
        rate = rospy.Rate(10)  # 10 Hz loop rate
        while not rospy.is_shutdown():
            lateral_error, heading_error = self.compute_errors()
            if lateral_error is None or heading_error is None:
                continue
            
            print(f"Lateral Error: {lateral_error}, Heading Error: {heading_error}")

            # TODO: Reinforcement Learning Model Inference Here
            # Placeholder: simple proportional correction (to be replaced with RL agent output)
            cmd_correction = Twist()
            cmd_correction.linear.x = -0.1 * lateral_error
            cmd_correction.angular.z = -0.2 * heading_error
            
            self.cmd_vel_offset_pub.publish(cmd_correction)
            rate.sleep()

if __name__ == '__main__':
    try:
        PurePursuitRLCorrection()
    except rospy.ROSInterruptException:
        pass
