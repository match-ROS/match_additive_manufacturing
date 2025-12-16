#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import rosbag
import sensor_msgs.point_cloud2 as pc2


def read_pose_z_times(bag_path, topic_pose):
    ts, zs = [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic_pose]):
            t_sec = msg.header.stamp.to_sec() if msg.header.stamp else t.to_sec()
            ts.append(t_sec)
            zs.append(msg.pose.position.z)

    if not ts:
        raise RuntimeError(f"No messages read on '{topic_pose}'")

    ts = np.asarray(ts, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    order = np.argsort(ts)
    return ts[order], zs[order]


def moving_average(x, w):
    if w <= 1:
        return x.copy()
    w = int(w)
    ker = np.ones(w, dtype=np.float64) / w
    xpad = np.pad(x, (w//2, w-1-w//2), mode="edge")
    return np.convolve(xpad, ker, mode="valid")


def find_transition_intervals(ts, zs, smooth_w=51, rate_thresh=0.01,
                             min_dur_s=0.3, merge_gap_s=0.5, pad_s=0.2, expected=4):
    z_s = moving_average(zs, smooth_w)
    dt = np.diff(ts)
    dz = np.diff(z_s)
    dt = np.clip(dt, 1e-6, None)
    zrate = dz / dt  # m/s, length N-1

    mask = zrate > rate_thresh
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError("No transitions found. Lower --rate_thresh or --smooth_w.")

    runs = []
    start = prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = prev = k
    runs.append((start, prev))

    intervals = []
    for a, b in runs:
        i0 = a
        i1 = b + 1  # maps zrate indices to sample indices
        if ts[i1] - ts[i0] >= min_dur_s:
            intervals.append([i0, i1])

    if not intervals:
        raise RuntimeError("Transitions exist but all shorter than --min_dur_s. Lower it.")

    # merge close intervals
    merged = [intervals[0]]
    for i0, i1 in intervals[1:]:
        l0, l1 = merged[-1]
        if ts[i0] - ts[l1] <= merge_gap_s:
            merged[-1][1] = i1
        else:
            merged.append([i0, i1])

    # pad in samples
    median_dt = float(np.median(np.diff(ts)))
    pad_n = int(max(1, pad_s / max(median_dt, 1e-6)))
    padded = []
    for i0, i1 in merged:
        p0 = max(0, i0 - pad_n)
        p1 = min(len(ts) - 1, i1 + pad_n)
        padded.append([p0, p1])

    # keep strongest expected transitions
    if len(padded) != expected:
        if len(padded) < expected:
            raise RuntimeError(
                f"Detected {len(padded)} transitions, expected {expected}. "
                "Tune thresholds."
            )
        strengths = [zs[j] - zs[i] for i, j in padded]
        top = np.argsort(strengths)[-expected:]
        padded = [padded[i] for i in sorted(top)]

    # convert to time intervals [t_start, t_end]
    time_intervals = [(float(ts[i0]), float(ts[i1])) for i0, i1 in padded]
    return time_intervals


def read_profile_heights(bag_path, topic_profile, flank=50):
    t_list, h_list = [], []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic_profile]):
            t_sec = msg.header.stamp.to_sec() if msg.header.stamp else t.to_sec()

            pts = np.array(list(pc2.read_points(msg, field_names=("z",), skip_nans=True)),
                           dtype=np.float64)
            if pts.size == 0:
                continue

            z = pts[:, 0]
            i = int(np.argmax(z))
            lo = max(0, i - flank)
            hi = min(len(z), i + flank + 1)
            h = float(np.mean(z[lo:hi]))

            t_list.append(t_sec)
            h_list.append(h)

    if not t_list:
        raise RuntimeError(f"No usable profiles read on '{topic_profile}'")

    t_arr = np.asarray(t_list, dtype=np.float64)
    h_arr = np.asarray(h_list, dtype=np.float64)
    order = np.argsort(t_arr)
    return t_arr[order], h_arr[order]


def in_any_interval(t, intervals):
    for a, b in intervals:
        if a <= t <= b:
            return True
    return False


def build_layer_time_windows(t_min, t_max, transition_intervals):
    """
    Creates 5 windows excluding transition intervals:
    [t_min..tr0_start), (tr0_end..tr1_start), ..., (tr3_end..t_max)
    """
    trs = sorted(transition_intervals, key=lambda x: x[0])
    cuts = [t_min]
    for a, b in trs:
        cuts.extend([a, b])
    cuts.append(t_max)

    # windows are even-index pairs that are NOT transitions
    windows = []
    for i in range(0, len(cuts)-1, 2):
        w0, w1 = cuts[i], cuts[i+1]
        if w1 > w0:
            windows.append((w0, w1))

    return windows  # should be 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="record_20251212_164357_MuR.bag")
    ap.add_argument("--topic_pose", default="/mur620c/UR10_r/global_tcp_pose")
    ap.add_argument("--topic_profile", default="/profiles")

    ap.add_argument("--flank", type=int, default=50)

    # transition detection
    ap.add_argument("--smooth_w", type=int, default=51)
    ap.add_argument("--rate_thresh", type=float, default=0.01, help="m/s")
    ap.add_argument("--min_dur_s", type=float, default=0.3)
    ap.add_argument("--merge_gap_s", type=float, default=0.5)
    ap.add_argument("--pad_s", type=float, default=0.2)

    ap.add_argument("--out", default="profile_height_over_time_by_layer.pdf")
    args = ap.parse_args()

    if not os.path.exists(args.bag):
        raise FileNotFoundError(os.path.abspath(args.bag))

    # 1) layer transitions from TCP z
    ts_pose, zs_pose = read_pose_z_times(args.bag, args.topic_pose)
    transition_intervals = find_transition_intervals(
        ts_pose, zs_pose,
        smooth_w=args.smooth_w,
        rate_thresh=args.rate_thresh,
        min_dur_s=args.min_dur_s,
        merge_gap_s=args.merge_gap_s,
        pad_s=args.pad_s,
        expected=4
    )

    # 2) profile heights from /profile
    ts_prof, h_prof = read_profile_heights(args.bag, args.topic_profile, flank=args.flank)

    # Use profile time span for plotting and window creation
    t_min, t_max = float(ts_prof[0]), float(ts_prof[-1])

    # Clip transitions to profile time span (robust)
    transition_intervals = [(max(a, t_min), min(b, t_max)) for a, b in transition_intervals if min(b, t_max) > max(a, t_min)]

    layer_windows = build_layer_time_windows(t_min, t_max, transition_intervals)
    if len(layer_windows) != 5:
        print(f"Warning: got {len(layer_windows)} layer windows (expected 5). "
              f"Transitions/windows may need tuning.")

    # 3) remove transition samples and assign to layers
    keep = np.array([not in_any_interval(t, transition_intervals) for t in ts_prof], dtype=bool)
    ts_k = ts_prof[keep]
    h_k = h_prof[keep]

    t0 = t_min
    plt.figure()

    for li, (a, b) in enumerate(layer_windows):
        mask = (ts_k >= a) & (ts_k < b)
        if not np.any(mask):
            continue
        plt.plot(ts_k[mask] - t0, h_k[mask], label=f"Layer {li}")

    plt.xlabel("time [s]")
    plt.ylabel("profile height [m] (mean z around max ± flank)")
    plt.title("Profilhöhe über Zeit (pro Layer, Rampen entfernt)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved: {args.out}")

    png_out = os.path.splitext(args.out)[0] + ".png"
    plt.savefig(png_out, dpi=200)
    print(f"Saved: {png_out}")


if __name__ == "__main__":
    main()
