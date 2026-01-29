#!/usr/bin/env python3
import rosbag
import matplotlib.pyplot as plt
import glob
import os

TOPIC = "/mur620c/cmd_vel"


def read_cmd_vel(bagfile):
    times = []
    values = []

    with rosbag.Bag(bagfile, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[TOPIC]):
            times.append(t.to_sec())
            values.append(msg.linear.x)

    if not times:
        raise RuntimeError(f"No data on {TOPIC} in {bagfile}")

    # normalize time
    t0 = times[0]
    times = [t - t0 for t in times]

    return times, values


def main():
    # find bag files in current directory
    bags = sorted(glob.glob("*.bag"))
    if len(bags) < 2:
        print("Error: need at least two bagfiles in current folder.")
        return

    bag1, bag2 = bags[:2]   # take the first two
    print(f"Using:\n 1: {bag1}\n 2: {bag2}")

    t1, v1 = read_cmd_vel(bag1)
    t2, v2 = read_cmd_vel(bag2)

    plt.figure(figsize=(10,5))
    plt.plot(t1, v1, label=f"{os.path.basename(bag1)}")
    plt.plot(t2, v2, label=f"{os.path.basename(bag2)}")

    plt.title("Comparison of MiR linear velocities")
    plt.xlabel("Time [s]")
    plt.ylabel("linear.x [m/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
