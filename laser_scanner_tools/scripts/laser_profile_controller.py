#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32


class LaserProfileController(object):
    def __init__(self):
        rospy.init_node("laser_profile_controller", anonymous=True)

        # --- Parameters ---
        self.tcp_pose_topic = rospy.get_param(
            "~tcp_pose_topic",
            "/mur620c/UR10_r/ur_calibrated_pose"
        )
        self.cmd_topic = rospy.get_param(
            "~cmd_topic",
            "/mur620c/UR10_r/twist_controller/command_collision_free"
        )

        self.lateral_error_topic = rospy.get_param(
            "~lateral_error_topic",
            "/laser_profile/lateral_error"
        )
        self.height_error_topic = rospy.get_param(
            "~height_error_topic",
            "/laser_profile/height_error"
        )
        self.lateral_pitch_topic = rospy.get_param(
            "~lateral_pitch_topic",
            "/profiles_pitch_m"
        )

        # Override-Steuerung
        self.manual_override_topic = rospy.get_param(
            "~manual_override_topic",
            "/velocity_override_manual"
        )
        self.override_out_topic = rospy.get_param(
            "~override_out_topic",
            "/velocity_override"
        )
        # max. Änderung gegenüber manuellem Override (z.B. 0.3 = ±30 Prozentpunkte)
        self.max_override_adjust = rospy.get_param("~max_override_adjust", 0.4)
        # Gain: wie stark der Höhenfehler in Override-Änderung übersetzt wird
        self.height_override_gain = rospy.get_param("~height_override_gain", 0.008)
        # Grenzen für den effektiven Override
        self.override_min = rospy.get_param("~override_min", 0.0)
        self.override_max = rospy.get_param("~override_max", 2.2)

        self.k_p = rospy.get_param("~k_p", 1.2)
        self.k_i = rospy.get_param("~k_i", 0.04)
        self.k_d = rospy.get_param("~k_d", 0.0)
        self.I_max = rospy.get_param("~I_max", 0.1)
        self.lateral_pitch_m = rospy.get_param("~lateral_pitch_m", 0.001)
        self.max_vel = rospy.get_param("~max_vel", 0.15)
        self.output_smoothing_coeff = rospy.get_param("~output_smoothing_coeff", 0.96)
        self.control_rate = rospy.get_param("~control_rate", 200.0)


        # --- State ---
        self.ur_tcp_pose = None
        self.lateral_error = None
        self.height_error = None
        self.prev_output = [0.0, 0.0, 0.0]

        self.manual_override = 1.0
        self.current_override = 1.0

        # --- Init PID state ---
        self.pid_integral = 0.0
        self.prev_error = 0.0



        # --- Publisher ---
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.override_pub = rospy.Publisher(self.override_out_topic, Float32, queue_size=1)

        # --- Subscribers ---
        rospy.Subscriber(self.tcp_pose_topic, PoseStamped,
                         self.ur_tcp_callback, queue_size=1)
        rospy.Subscriber(self.lateral_error_topic, Float32,
                         self.lateral_error_callback, queue_size=1)
        rospy.Subscriber(self.height_error_topic, Float32,
                         self.height_error_callback, queue_size=1)
        rospy.Subscriber(self.lateral_pitch_topic, Float32,
                 self.lateral_pitch_callback, queue_size=1)
        rospy.Subscriber(self.manual_override_topic, Float32,
                         self.manual_override_callback, queue_size=1)

        rospy.loginfo("LaserProfileController started.")

        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

    # ------------------------------------------------------------------

    def ur_tcp_callback(self, msg: PoseStamped):
        self.ur_tcp_pose = msg

    def lateral_error_callback(self, msg: Float32):
        self.lateral_error = msg.data

    def height_error_callback(self, msg: Float32):
        self.height_error = msg.data

    def lateral_pitch_callback(self, msg: Float32):
        if msg.data > 0.0:
            self.lateral_pitch_m = msg.data

    def manual_override_callback(self, msg: Float32):
        self.manual_override = msg.data

    def smooth_output(self, new_output):
        smoothed = []
        for i in range(3):
            smoothed_value = (self.output_smoothing_coeff * self.prev_output[i] +
                              (1.0 - self.output_smoothing_coeff) * new_output[i])
            smoothed.append(smoothed_value)
        self.prev_output = smoothed
        return np.array(smoothed)

    def update_override_from_height(self):
        """
        Nutzt height_error, um den manuellen Override moderat zu verändern.

        height_error < 0  -> zu viel Material -> schneller fahren (Override hoch)
        height_error > 0  -> zu wenig Material -> langsamer fahren (Override runter)
        """
        if self.height_error is None:
            # Noch keine Höheninfo -> einfach manuellen Override durchreichen
            self.current_override = self.manual_override
        else:
            # Delta-Override proportional zum Höhenfehler
            # Vorzeichen: negativer Fehler -> +Delta, positiver Fehler -> -Delta
            raw_delta = -self.height_override_gain * self.height_error
            # Auf ±max_override_adjust begrenzen
            delta = np.clip(raw_delta,
                            -self.max_override_adjust,
                            self.max_override_adjust)
            # Effektiver Override um Delta verschoben
            eff = self.manual_override + delta

            # Auf globale Grenzen clampen
            eff = max(self.override_min, min(self.override_max, eff))
            self.current_override = eff

        # Publish
        self.override_pub.publish(Float32(data=self.current_override))

    # ------------------------------------------------------------------

    def compute_pid(self, error):
        """Einfache PID-Regler-Implementierung."""
        if not hasattr(self, 'pid_integral'):
            self.pid_integral = 0.0
            self.prev_error = 0.0

        # Integralterm
        self.pid_integral += error / self.control_rate
        # Anti-Windup
        self.pid_integral = np.clip(self.pid_integral, -self.I_max, self.I_max)

        # Differentialterm
        derivative = (error - self.prev_error) * self.control_rate

        # PID-Ausgang
        output = (self.k_p * error +
                  self.k_i * self.pid_integral +
                  self.k_d * derivative)

        # Vorbereiten für nächsten Schritt
        self.prev_error = error

        return output

    def update(self):
        if self.ur_tcp_pose is None:
            rospy.logwarn_throttle(1.0, "No TCP pose received yet.")
            return
        if self.lateral_error is None:
            rospy.logwarn_throttle(1.0, "No lateral_error received yet.")
            return

        # Override anhand Höhenfehler updaten
        self.update_override_from_height()


        # --- Lateralgeschwindigkeit in TCP-Frame (Y-Achse) ---
        v_lat = -1.0 *  self.compute_pid(self.lateral_error* self.lateral_pitch_m) 
        v_lat = np.clip(v_lat, -self.max_vel, self.max_vel)

        # v_tcp: nur seitliche Bewegung entlang -Y des TCP
        v_tcp = np.array([0.0, -v_lat, 0.0])
        v_tcp = self.smooth_output(v_tcp)

        # --- TCP -> UR base Rotation ---
        q = self.ur_tcp_pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1 - 2*qx*qx - 2*qy*qy]
        ])

        v_urbase = R.dot(v_tcp)

        twist = Twist()
        twist.linear.x = v_urbase[0]
        twist.linear.y = v_urbase[1]
        twist.linear.z = 0.0

        #self.cmd_pub.publish(twist)


if __name__ == "__main__":
    try:
        LaserProfileController()
    except rospy.ROSInterruptException:
        pass
