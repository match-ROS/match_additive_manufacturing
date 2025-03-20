#! /usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion

class LayerErrorLogger:
    def __init__(self):
        rospy.init_node('layer_error_logger', anonymous=True)
        
        # Subscriptions
        rospy.Subscriber('/mir_actual_pose', PoseStamped, self.actual_pose_callback)
        rospy.Subscriber('/mir_target_pose', PoseStamped, self.target_pose_callback)
        rospy.Subscriber('/layer_progress', Float32, self.layer_progress_callback)
        
        self.actual_pose = None
        self.target_pose = None
        self.layer_progress = 0.0
        
        # Datenpuffer für 5 Schichten
        self.error_data = []
        self.current_layer_data = []
        self.layer_count = 0
        self.plot_running = False
        
        rospy.loginfo("Layer Error Logger Node Started")
        rospy.spin()
    
    def actual_pose_callback(self, msg):
        self.actual_pose = msg
    
    def target_pose_callback(self, msg):
        self.target_pose = msg
    
    def layer_progress_callback(self, msg):
        self.layer_progress = msg.data
        
        if self.actual_pose is None or self.target_pose is None:
            return
        
        # Fehler berechnen
        lateral_error, heading_error = self.compute_errors()
        
        # Daten sammeln
        self.current_layer_data.append((self.layer_progress, lateral_error, heading_error))
        
        # Falls ein Layer abgeschlossen ist, speichern und neuen Layer starten
        if self.layer_progress >= 1.0:
            self.error_data.append(self.current_layer_data)
            self.current_layer_data = []
            self.layer_count += 1
            rospy.loginfo(f"Schicht {self.layer_count} abgeschlossen")
        
        # Nach 5 Schichten auswerten
        if self.layer_count >= 2 and not self.plot_running:
            self.plot_running = True
            self.plot_errors()
            rospy.signal_shutdown("Logging abgeschlossen")
    
    def compute_errors(self):
        x_r, y_r = self.actual_pose.pose.position.x, self.actual_pose.pose.position.y
        x_t, y_t = self.target_pose.pose.position.x, self.target_pose.pose.position.y
        
        lateral_error = np.sqrt((x_r - x_t)**2 + (y_r - y_t)**2)
        
        quat_r = self.actual_pose.pose.orientation
        quat_t = self.target_pose.pose.orientation
        _, _, yaw_r = euler_from_quaternion([quat_r.x, quat_r.y, quat_r.z, quat_r.w])
        _, _, yaw_t = euler_from_quaternion([quat_t.x, quat_t.y, quat_t.z, quat_t.w])
        heading_error = yaw_r - yaw_t
        
        return lateral_error, heading_error
    
    def plot_errors(self):
        rospy.loginfo("Erstelle Fehler-Plots für die 5 Schichten")
        plt.figure(figsize=(10, 5))
        for i, layer in enumerate(self.error_data):
            progress, lateral, heading = zip(*layer)
            plt.plot(progress, lateral, label=f"Schicht {i+1}", alpha=0.7)
        plt.xlabel("Layer Progress (0-1)")
        plt.ylabel("Lateral Error (m)")
        plt.title("Lateral Error über die Schichten")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        for i, layer in enumerate(self.error_data):
            progress, lateral, heading = zip(*layer)
            plt.plot(progress, heading, label=f"Schicht {i+1}", alpha=0.7)
        plt.xlabel("Layer Progress (0-1)")
        plt.ylabel("Heading Error (rad)")
        plt.title("Heading Error über die Schichten")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    try:
        LayerErrorLogger()
    except rospy.ROSInterruptException:
        pass