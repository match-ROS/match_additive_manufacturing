#!/usr/bin/env python3
"""ROS node to stream strand-center detections from an OAK-D Pro device."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import depthai as dai
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, String


@dataclass
class StrandCenterDetection:
    u: float
    v: float
    depth_mm: int


def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1280, 720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCamera.ColorOrder.BGR)
    cam.setFps(30)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_l.setResolution(dai.MonoCamera.Properties.SensorResolution.THE_720_P)
    mono_r.setResolution(dai.MonoCamera.Properties.SensorResolution.THE_720_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def find_filament_center(rgb_frame: np.ndarray, row_v: int) -> Tuple[Optional[float], int]:
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    height, _ = binary.shape
    row_v = int(np.clip(row_v, 0, height - 1))
    row = binary[row_v, :]

    white_indices = np.where(row > 0)[0]
    if white_indices.size == 0:
        return None, row_v

    segments = []
    start = white_indices[0]
    prev = white_indices[0]
    for idx in white_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx
    segments.append((start, prev))

    lengths = [end - st + 1 for st, end in segments]
    best_seg = segments[int(np.argmax(lengths))]
    u_center = 0.5 * (best_seg[0] + best_seg[1])

    return u_center, row_v


class StrandCenterNode:
    def __init__(self) -> None:
        row_v_param = rospy.get_param("~row_v", 360)
        if isinstance(row_v_param, (int, float, str)):
            try:
                self.row_v = int(float(row_v_param))
            except (TypeError, ValueError):
                rospy.logwarn("Invalid row_v parameter '%s'. Falling back to 360.", row_v_param)
                self.row_v = 360
        else:
            rospy.logwarn("Unsupported row_v parameter type '%s'. Falling back to 360.", type(row_v_param))
            self.row_v = 360
        self.publish_json = rospy.get_param("~publish_json", False)
        self.publish_debug_log = rospy.get_param("~debug_log", True)
        self.udp_forward_ip = rospy.get_param("~forward_udp_ip", None)
        self.udp_forward_port = rospy.get_param("~forward_udp_port", 5005)

        self.center_pub = rospy.Publisher("strand_center/center", PointStamped, queue_size=10)
        self.depth_pub = rospy.Publisher("strand_center/depth_mm", Float32, queue_size=10)
        self.status_pub = rospy.Publisher("strand_center/status", String, queue_size=10)

        self.pipeline = create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        self.sock = None  # type: Optional[socket.socket]
        if self.udp_forward_ip:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self) -> None:
        if self.sock:
            self.sock.close()
        self.device.close()

    def publish_detection(self, detection: StrandCenterDetection) -> None:
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "oak_d_pro_rgb"
        msg.point.x = detection.u
        msg.point.y = detection.v
        msg.point.z = detection.depth_mm
        self.center_pub.publish(msg)

        self.depth_pub.publish(Float32(float(detection.depth_mm)))

        self.status_pub.publish(String(data="detected"))

        if self.publish_json or self.sock:
            payload = json.dumps({
                "u": detection.u,
                "v": detection.v,
                "depth_mm": detection.depth_mm,
            }).encode("utf-8")
            if self.publish_json:
                self.status_pub.publish(String(data=payload.decode()))
            if self.sock and self.udp_forward_ip:
                self.sock.sendto(payload, (self.udp_forward_ip, self.udp_forward_port))

        if self.publish_debug_log:
            rospy.logdebug("Strand center u=%.2f v=%.2f depth=%d mm", detection.u, detection.v, detection.depth_mm)

    def publish_no_detection(self) -> None:
        self.status_pub.publish(String(data="no_detection"))

    def spin(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            try:
                rgb_in = self.q_rgb.tryGet()
                depth_in = self.q_depth.tryGet()
            except RuntimeError as exc:
                rospy.logerr("DepthAI queues unavailable: %s", exc)
                break

            if rgb_in is None or depth_in is None:
                rate.sleep()
                continue

            rgb_frame = rgb_in.getCvFrame()
            depth_frame = depth_in.getFrame()

            u_center, v_center = find_filament_center(rgb_frame, self.row_v)
            if u_center is None:
                self.publish_no_detection()
                rate.sleep()
                continue

            u_int = int(round(u_center))
            v_int = int(round(v_center))

            if not (0 <= v_int < depth_frame.shape[0] and 0 <= u_int < depth_frame.shape[1]):
                self.publish_no_detection()
                rate.sleep()
                continue

            depth_mm = int(depth_frame[v_int, u_int])
            if depth_mm <= 0:
                self.publish_no_detection()
                rate.sleep()
                continue

            detection = StrandCenterDetection(u=u_center, v=v_center, depth_mm=depth_mm)
            self.publish_detection(detection)

            rate.sleep()


def main() -> None:
    rospy.init_node("oak_strand_center_node", log_level=rospy.INFO)
    node = StrandCenterNode()
    rospy.loginfo("OAK-D strand center node started")
    node.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
