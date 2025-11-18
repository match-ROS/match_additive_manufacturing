import depthai as dai
import numpy as np
import cv2
import json
import socket

# --------- Configurable parameters ----------
ROW_V = 360              # image row to scan for filament (example for 720p)
UDP_IP = "192.168.0.10"  # IP of your ROS host
UDP_PORT = 5005
# -------------------------------------------

def create_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(1280, 720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCamera.ColorOrder.BGR)
    cam.setFps(30)

    monoL = pipeline.create(dai.node.MonoCamera)
    monoR = pipeline.create(dai.node.MonoCamera)
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoL.setResolution(dai.MonoCamera.Properties.SensorResolution.THE_720_P)
    monoR.setResolution(dai.MonoCamera.Properties.SensorResolution.THE_720_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)  # align depth to RGB
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

def find_filament_center(rgb_frame, row_v):
    """Return u_center or None if no filament is found."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # You may need THRESH_BINARY_INV depending on contrast
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    row_v = int(np.clip(row_v, 0, h - 1))
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

def main():
    pipeline = create_pipeline()

    # UDP socket to send results to ROS host
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with dai.Device(pipeline) as device:
        q_rgb   = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            rgb_in   = q_rgb.get()
            depth_in = q_depth.get()

            rgb_frame = rgb_in.getCvFrame()
            depth_frame = depth_in.getFrame()  # uint16 depth, typically in mm

            u_center, v_center = find_filament_center(rgb_frame, ROW_V)
            if u_center is None:
                # No filament found in that row, skip this frame
                continue

            u_int = int(round(u_center))
            v_int = int(round(v_center))

            if (0 <= v_int < depth_frame.shape[0]) and (0 <= u_int < depth_frame.shape[1]):
                depth_mm = int(depth_frame[v_int, u_int])
            else:
                continue

            if depth_mm <= 0:
                # invalid depth
                continue

            # Build small JSON message (pixel + depth)
            msg = {
                "u": float(u_center),
                "v": float(v_center),
                "depth_mm": int(depth_mm)
            }
            data = json.dumps(msg).encode("utf-8")
            sock.sendto(data, (UDP_IP, UDP_PORT))

if __name__ == "__main__":
    main()
