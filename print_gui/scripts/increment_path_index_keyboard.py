#! /usr/bin/env python3

import rospy
import select
import sys
import termios
import tty
from std_msgs.msg import Int32


def _get_key(timeout_s):
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None

    ch1 = sys.stdin.read(1)
    if ch1 != "\x1b":
        return ch1

    if select.select([sys.stdin], [], [], 0.01)[0]:
        ch2 = sys.stdin.read(1)
        if ch2 == "[" and select.select([sys.stdin], [], [], 0.01)[0]:
            ch3 = sys.stdin.read(1)
            return f"\x1b[{ch3}"
        return ch1 + ch2

    return ch1


def increment_path_index_keyboard():
    rospy.init_node("increment_path_index_keyboard", anonymous=True)

    path_index_topic = rospy.get_param("~path_index_topic", "/path_index")
    initial_path_index = rospy.get_param("~initial_path_index", 0)
    delay_s = float(rospy.get_param("~delay", 0.1))
    speed = float(rospy.get_param("~speed", 2.0))

    if delay_s < 0.0:
        rospy.logwarn("Delay is negative. Clamping to 0.0s.")
        delay_s = 0.0
    if speed <= 0.0:
        rospy.logwarn("Speed must be > 0. Using 2.0.")
        speed = 2.0

    fast_delay_s = delay_s / speed if speed != 0.0 else delay_s

    index_pub = rospy.Publisher(path_index_topic, Int32, queue_size=10, latch=True)

    path_index = int(initial_path_index)
    msg = Int32()
    msg.data = path_index
    index_pub.publish(msg)

    rospy.loginfo("Keyboard control: Up Arrow increments index. Space toggles fast delay. Ctrl-C to exit.")

    fast_mode = False

    settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while not rospy.is_shutdown():
            key = _get_key(0.1)
            if key is None:
                continue
            print(f"Key pressed: {repr(key)}")
            if key == "\x1b":
                path_index += 1
                msg = Int32()
                msg.data = path_index
                index_pub.publish(msg)
                rospy.loginfo(f"Published path index: {path_index}")

                rospy.sleep(fast_delay_s if fast_mode else delay_s)
            elif key == " ":
                fast_mode = not fast_mode
                rospy.loginfo(
                    "Fast delay enabled" if fast_mode else "Fast delay disabled"
                )
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    try:
        increment_path_index_keyboard()
    except rospy.ROSInterruptException:
        pass
