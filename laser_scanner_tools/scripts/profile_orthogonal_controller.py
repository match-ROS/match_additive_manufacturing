#!/usr/bin/env python3
import rospy
import numpy as np
from collections import deque

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class ProfileOrthogonalController(object):
    def __init__(self):
        rospy.init_node("profile_orthogonal_controller", anonymous=True)

        # Parameters
        self.profile_topic = rospy.get_param("~profile_topic", "/profiles_float")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/orthogonal_twist")
        self.window_size = rospy.get_param("~window_size", 10)     # Anzahl Messungen zum Mitteln
        self.k_p = rospy.get_param("~k_p", 0.1)                   # Proportionalfaktor
        self.max_vel = rospy.get_param("~max_vel", 0.1)            # [m/s] Begrenzung |vx|

        self.deviation_history = deque(maxlen=self.window_size)

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.sub = rospy.Subscriber(self.profile_topic,
                                    Float32MultiArray,
                                    self.profile_callback,
                                    queue_size=1)

        rospy.loginfo("ProfileOrthogonalController started.")
        rospy.spin()

    def profile_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)

        if data.size == 0:
            return

        # Mittel-Index (inklusive NaNs)
        center_idx = (data.size - 1) / 2.0

        # Nur gültige Punkte (nicht NaN)
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            rospy.logwarn_throttle(1.0, "No valid points in /profiles_float")
            return

        valid_indices = np.nonzero(valid_mask)[0]
        valid_values = data[valid_mask]

        # Index des am stärksten negativen Wertes (Minimum)
        min_idx_valid_array = np.argmax(valid_values)
        min_idx = int(valid_indices[min_idx_valid_array])
        print(f"Min index: {min_idx}")

        # Abweichung vom mittleren Index
        deviation = float(min_idx - center_idx)

        # In Historie einfügen und über letzte N Messungen mitteln
        self.deviation_history.append(deviation)
        avg_deviation = float(np.mean(self.deviation_history))

        # Einfacher P-Regler -> Geschwindigkeit in x
        # Vorzeichen ggf. an dein Koordinatensystem anpassen
        v_x = self.k_p * avg_deviation * 0.001  # in m/s

        # Begrenzen
        v_x = max(min(v_x, self.max_vel), -self.max_vel)

        # Twist publizieren
        twist = Twist()
        twist.linear.x = v_x
        self.cmd_pub.publish(twist)

        # Optional: Debug-Ausgabe
        rospy.logdebug("min_idx=%d, center_idx=%.2f, dev=%.2f, avg_dev=%.2f, vx=%.3f",
                       min_idx, center_idx, deviation, avg_deviation, v_x)


if __name__ == "__main__":
    try:
        ProfileOrthogonalController()
    except rospy.ROSInterruptException:
        pass
