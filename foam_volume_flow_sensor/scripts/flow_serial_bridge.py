#!/usr/bin/env python3
"""Bridge Arduino CSV output into ROS FlowSample messages."""
import csv
import math
from typing import Optional

import rospy
from serial import Serial, SerialException

from foam_volume_flow_sensor.msg import FlowSample


class FlowSerialBridge:
    def __init__(self) -> None:
        self.port = rospy.get_param("~port", "/dev/ttyACM0")
        self.baud = int(rospy.get_param("~baud", 115200))
        self.timeout = float(rospy.get_param("~timeout", 0.5))
        self.line_prefix = rospy.get_param("~line_prefix", "")
        self.frame_id = rospy.get_param("~frame_id", "flow_sensor")
        topic = rospy.get_param("~topic", "samples")

        self.publisher = rospy.Publisher(topic, FlowSample, queue_size=10)
        self.serial: Optional[Serial] = None

    def spin(self) -> None:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if not self.serial:
                self._connect()
                continue

            try:
                raw_line = self.serial.readline().decode("ascii", errors="ignore").strip()
            except SerialException as exc:
                rospy.logwarn("Serial error (%s), trying to reconnect", exc)
                self._close_serial()
                rate.sleep()
                continue

            if not raw_line:
                rate.sleep()
                continue

            if self.line_prefix:
                if not raw_line.startswith(self.line_prefix):
                    continue
                raw_line = raw_line[len(self.line_prefix):]

            if raw_line.startswith("time_ms"):
                continue

            try:
                sample = self._parse_line(raw_line)
            except ValueError as exc:
                rospy.logwarn_throttle(5.0, "Failed to parse line '%s': %s", raw_line, exc)
                rate.sleep()
                continue

            self.publisher.publish(sample)
            rate.sleep()

        self._close_serial()

    def _connect(self) -> None:
        if rospy.is_shutdown():
            return
        try:
            self.serial = Serial(self.port, self.baud, timeout=self.timeout)
            rospy.loginfo("Connected to %s @ %d baud", self.port, self.baud)
        except SerialException as exc:
            rospy.logwarn_throttle(5.0, "Unable to open %s: %s", self.port, exc)
            rospy.sleep(1.0)

    def _close_serial(self) -> None:
        if self.serial:
            try:
                self.serial.close()
            except SerialException:
                pass
        self.serial = None

    def _parse_line(self, line: str) -> FlowSample:
        reader = csv.reader([line])
        fields = next(reader)
        if len(fields) < 6:
            raise ValueError("expected at least 6 comma-separated values")

        millis = int(float(fields[0]))
        channel = int(float(fields[1]))
        raw_adc = float(fields[2])
        voltage = float(fields[3])
        current = float(fields[4])
        percent = float(fields[5])
        engineering = math.nan
        if len(fields) >= 7 and fields[6]:
            engineering = float(fields[6])

        msg = FlowSample()
        msg.header.stamp = rospy.Time.from_sec(millis / 1000.0)
        msg.header.frame_id = self.frame_id
        msg.channel = channel
        msg.raw_adc = raw_adc
        msg.voltage_v = voltage
        msg.current_ma = current
        msg.percent = percent
        msg.engineering_value = engineering
        return msg


def main() -> None:
    rospy.init_node("flow_serial_bridge")
    bridge = FlowSerialBridge()
    try:
        bridge.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
