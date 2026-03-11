#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse

import rosbag
import numpy as np
import matplotlib.pyplot as plt


def moving_average(signal, window_size):
    """
    Simple centered moving average filter.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size == 1:
        return signal.copy()

    kernel = np.ones(window_size) / window_size
    # pad to keep same length
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode='edge')
    filtered = np.convolve(padded, kernel, mode='valid')
    return filtered


def read_twist_magnitudes(bag_path, topic_name):
    """
    Read Twist messages from bag and compute xy speed magnitude.
    Returns:
        times_rel: relative time in seconds
        v_xy: xy speed magnitude
    """
    times = []
    v_xy = []

    with rosbag.Bag(bag_path, 'r') as bag:
        t0 = None

        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if t0 is None:
                t0 = t.to_sec()

            ts = t.to_sec() - t0

            # geometry_msgs/Twist
            vx = msg.linear.x
            vy = msg.linear.y

            v_mag_xy = math.sqrt(vx**2 + vy**2)

            times.append(ts)
            v_xy.append(v_mag_xy)

    if len(times) == 0:
        raise RuntimeError(f"No messages found on topic '{topic_name}' in bag '{bag_path}'.")

    return np.array(times), np.array(v_xy)


def save_plot(times, raw_signal, filtered_signal, output_pdf, title):
    """
    Save plot as PDF.
    """
    start_index = 0
    end_index = 700

    plt.figure(figsize=(10, 5))
    plt.plot(times[start_index:end_index], raw_signal[start_index:end_index], label='Raw |v_xy|')
    plt.plot(times[start_index:end_index], filtered_signal[start_index:end_index], label='Filtered |v_xy|', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Velocity magnitude in XY plane [m/s]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Read Twist messages from rosbag, compute xy velocity magnitude, filter it, and save plot as PDF."
    )
    parser.add_argument(
        "--bag",
        type=str,
        #default="record_20260202_113140_MuR.bag",
        default="record_20251209_173205_MuR.bag",
        help="Path to rosbag file"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/ur_twist_direction_world",
        help="Topic containing geometry_msgs/Twist"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Moving average window size (odd number recommended)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ur_twist_direction_world_xy_speed.pdf",
        help="Output PDF filename"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"Error: bag file not found: {args.bag}")
        sys.exit(1)

    times, v_xy = read_twist_magnitudes(args.bag, args.topic)

    # Ensure sensible window size
    window = max(1, min(args.window, len(v_xy)))
    if window % 2 == 0 and window > 1:
        window += 1
        if window > len(v_xy):
            window -= 2
            if window < 1:
                window = 1

    v_xy_filtered = moving_average(v_xy, window)

    save_plot(
        times=times,
        raw_signal=v_xy,
        filtered_signal=v_xy_filtered,
        output_pdf=args.output,
        title=f"XY velocity magnitude from {args.topic}"
    )

    print("Done.")
    print(f"Read {len(v_xy)} messages from topic: {args.topic}")
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()