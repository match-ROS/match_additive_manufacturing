#!/bin/bash
roslaunch print_sim run_simulation_complete.launch &
sleep 60  # Warte 5 Sekunden
roslaunch ur_trajectory_follower complete_ur_trajectory_follower.launch
