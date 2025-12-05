#!/usr/bin/env python3
import rosbag

bag_a = "record_20251205_135424_MuR.bag"
bag_b = "record_20251205_135424_GUI-PC.bag"
out_bag = "merged.bag"

def read_bag(filename):
    with rosbag.Bag(filename, "r") as bag:
        for topic, msg, t in bag.read_messages():
            yield (t, topic, msg)

# --- beide Bags einlesen und sortieren ---
all_msgs = list(read_bag(bag_a)) + list(read_bag(bag_b))
all_msgs.sort(key=lambda x: x[0])  # sort after timestamp t

# --- neue Bag schreiben ---
with rosbag.Bag(out_bag, "w") as out:
    for t, topic, msg in all_msgs:
        out.write(topic, msg, t)

print(f"Fertig. Ausgabe: {out_bag}")
