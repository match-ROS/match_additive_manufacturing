#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, TwistStamped

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        # Update the integral term and compute the derivative term
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class PIDTwistController:
    def __init__(self):
        self.stamped = rospy.get_param('~stamped', False)
        # Load PID parameters for each twist component via ROS parameters
        # Linear components
        Kp_linear_x = rospy.get_param('Kp_linear_x', 1.0)
        Ki_linear_x = rospy.get_param('Ki_linear_x', 0.0)
        Kd_linear_x = rospy.get_param('Kd_linear_x', 0.0)
        
        Kp_linear_y = rospy.get_param('Kp_linear_y', 1.0)
        Ki_linear_y = rospy.get_param('Ki_linear_y', 0.0)
        Kd_linear_y = rospy.get_param('Kd_linear_y', 0.0)
        
        Kp_linear_z = rospy.get_param('Kp_linear_z', 1.0)
        Ki_linear_z = rospy.get_param('Ki_linear_z', 0.0)
        Kd_linear_z = rospy.get_param('Kd_linear_z', 0.0)
        
        # Angular components
        Kp_angular_x = rospy.get_param('Kp_angular_x', 1.0)
        Ki_angular_x = rospy.get_param('Ki_angular_x', 0.0)
        Kd_angular_x = rospy.get_param('Kd_angular_x', 0.0)
        
        Kp_angular_y = rospy.get_param('Kp_angular_y', 1.0)
        Ki_angular_y = rospy.get_param('Ki_angular_y', 0.0)
        Kd_angular_y = rospy.get_param('Kd_angular_y', 0.0)
        
        Kp_angular_z = rospy.get_param('Kp_angular_z', 1.0)
        Ki_angular_z = rospy.get_param('Ki_angular_z', 0.0)
        Kd_angular_z = rospy.get_param('Kd_angular_z', 0.0)

        # Create PID controllers for each twist axis
        self.pid_linear_x  = PID(Kp_linear_x, Ki_linear_x, Kd_linear_x)
        self.pid_linear_y  = PID(Kp_linear_y, Ki_linear_y, Kd_linear_y)
        self.pid_linear_z  = PID(Kp_linear_z, Ki_linear_z, Kd_linear_z)
        self.pid_angular_x = PID(Kp_angular_x, Ki_angular_x, Kd_angular_x)
        self.pid_angular_y = PID(Kp_angular_y, Ki_angular_y, Kd_angular_y)
        self.pid_angular_z = PID(Kp_angular_z, Ki_angular_z, Kd_angular_z)
        if self.stamped:
            # Publisher for the output Twist message
            self.pub = rospy.Publisher('output_twist', TwistStamped, queue_size=10)
            # Subscriber to the input Twist (error) topic
            self.sub = rospy.Subscriber('input_twist', TwistStamped, self.twist_callback)
        else:
            # Publisher for the output Twist message
            self.pub = rospy.Publisher('output_twist', Twist, queue_size=10)
            # Subscriber to the input Twist (error) topic
            self.sub = rospy.Subscriber('input_twist', Twist, self.twist_callback)
        
        self.last_time = rospy.Time.now()

    def twist_callback(self, msg):

        if self.stamped:
            twist_in = msg.twist
            # current_time = msg.header.stamp
        else:
            twist_in = msg
        
        current_time = rospy.Time.now()
        
        dt = (current_time - self.last_time).to_sec()
        # if dt == 0:
        #     dt = 0.001  # Avoid division by zero
        self.last_time = current_time

        # Here, the incoming twist message values are treated as the error
        error_linear_x  = twist_in.linear.x
        error_linear_y  = twist_in.linear.y
        error_linear_z  = twist_in.linear.z
        error_angular_x = twist_in.angular.x
        error_angular_y = twist_in.angular.y
        error_angular_z = twist_in.angular.z

        # Apply PID control to each component
        output_linear_x  = self.pid_linear_x.update(error_linear_x, dt)
        output_linear_y  = self.pid_linear_y.update(error_linear_y, dt)
        output_linear_z  = self.pid_linear_z.update(error_linear_z, dt)
        output_angular_x = self.pid_angular_x.update(error_angular_x, dt)
        output_angular_y = self.pid_angular_y.update(error_angular_y, dt)
        output_angular_z = self.pid_angular_z.update(error_angular_z, dt)

        # Create a new Twist message for the output
        new_twist = Twist()
        new_twist.linear.x  = output_linear_x
        new_twist.linear.y  = output_linear_y
        new_twist.linear.z  = output_linear_z
        new_twist.angular.x = output_angular_x
        new_twist.angular.y = output_angular_y
        new_twist.angular.z = output_angular_z

        if self.stamped:
            msg.twist = new_twist
        else:
            msg = new_twist
        self.pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('pid_twist_controller')
    controller = PIDTwistController()
    rospy.loginfo("PID Twist Controller node started.")
    rospy.spin()
