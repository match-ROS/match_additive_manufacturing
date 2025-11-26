from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLineEdit, QHBoxLayout, QPushButton, QLabel, QTableWidget, QCheckBox, QTableWidgetItem, QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox, QTextEdit, QComboBox, QDoubleSpinBox, QDialogButtonBox, QFormLayout, QDialog
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot
from typing import Any, cast
from PyQt5.QtGui import QIcon
from ros_interface import start_status_update, ur_follow_trajectory, open_rviz, launch_drivers, quit_drivers, turn_on_arm_controllers, turn_on_twist_controllers, stop_mir_motion, stop_idx_advancer, stop_ur_motion, stop_all_but_drivers
from ros_interface import enable_all_urs, move_to_home_pose, parse_mir_path, parse_ur_path, move_mir_to_start_pose, move_ur_to_start_pose, mir_follow_trajectory, increment_path_index, target_broadcaster
from ros_interface import ROSInterface
import os
import math
import html
import json


Qt = cast(Any, QtCore.Qt)


class EnterSpinBox(QSpinBox):
    """QSpinBox that emits returnPressed when user presses Enter."""
    returnPressed = pyqtSignal()

    def keyPressEvent(self, event):
        key_return = getattr(Qt, 'Key_Return', None)
        key_enter = getattr(Qt, 'Key_Enter', None)
        if event.key() in [k for k in (key_return, key_enter) if k is not None]:
            self.returnPressed.emit()
        super().keyPressEvent(event)


class ROSGui(QWidget):
    path_idx = pyqtSignal(int)
    medians = pyqtSignal(float, float)
    ros_log_signal = pyqtSignal(str, str, str)  # level, node, text
    
    def __init__(self):
        super().__init__()
        # state
        self.ur_follow_settings = {'idx_metric': 'virtual line', 'threshold': 0.010}
        self.servo_calib = self._load_servo_calibration_defaults()
        # ROS + window
        self.path_idx.connect(self._update_spinbox)
        self.medians.connect(self._update_medians)
        self.ros_interface = ROSInterface(self)
        self.setWindowTitle("Additive Manufacturing GUI")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), '../img/Logo.png')))
        self.setGeometry(100, 100, 3200, 1700)
        main_layout = QHBoxLayout()
        # Left column
        left_layout = QVBoxLayout()
        self.status_label = QLabel("Controller Status: Not Checked")
        self.status_label.setStyleSheet("border: 1px solid black; padding: 5px;")
        left_layout.addWidget(self.status_label)
        # Battery
        self.battery_group = QGroupBox("Battery Status"); self.battery_layout = QVBoxLayout(); self.battery_group.setLayout(self.battery_layout); left_layout.addWidget(self.battery_group); self.battery_labels = {}
        # Selection
        selection_group = QGroupBox("Robot and UR Selection"); selection_layout = QHBoxLayout(); robot_layout = QVBoxLayout()
        self.robots = {n: QCheckBox(n) for n in ["mur620a", "mur620b", "mur620c", "mur620d"]}
        for cb in self.robots.values(): robot_layout.addWidget(cb)
        for name, cb in self.robots.items(): cb.stateChanged.connect(lambda _, r=name: self.ros_interface.check_and_subscribe_battery())
        for robot in self.robots.keys():
            row = QHBoxLayout(); row.addWidget(QLabel(robot)); mir_label = QLabel("MiR: –"); ur_label = QLabel("UR: –"); row.addWidget(mir_label); row.addWidget(ur_label); self.battery_labels[robot] = (mir_label, ur_label); self.battery_layout.addLayout(row)
        ur_layout = QVBoxLayout(); ur_layout.addWidget(QLabel("Select URs:")); self.ur10_l = QCheckBox("UR10_l"); self.ur10_r = QCheckBox("UR10_r"); self.ur10_l.setChecked(False); self.ur10_r.setChecked(True); ur_layout.addWidget(self.ur10_l); ur_layout.addWidget(self.ur10_r)
        selection_layout.addLayout(robot_layout); selection_layout.addLayout(ur_layout); selection_group.setLayout(selection_layout); left_layout.addWidget(selection_group)
        # Override
        override_layout = QVBoxLayout()
        override_label = QLabel("Override (%)")
        self.override_slider = QSlider(Qt.Horizontal)
        self.override_slider.setRange(0, 100)
        self.override_slider.setValue(100)
        self.override_slider.setTickInterval(10)
        self.override_slider.setTickPosition(QSlider.TicksBelow)
        self.override_value_label = QLabel("100%")

        nozzle_label = QLabel("Nozzle Height Override (mm)")
        self.nozzle_override_slider = QSlider(Qt.Horizontal)
        self.nozzle_override_slider.setRange(-50, 50)
        self.nozzle_override_slider.setValue(0)
        self.nozzle_override_slider.setTickInterval(5)
        self.nozzle_override_slider.setTickPosition(QSlider.TicksBelow)
        self.nozzle_override_value_label = QLabel("0.0 mm")

        self.ros_interface.init_override_velocity_slider()
        self.ros_interface.init_nozzle_override_slider()

        override_layout.addWidget(override_label)
        override_layout.addWidget(self.override_slider)
        override_layout.addWidget(self.override_value_label)
        override_layout.addWidget(nozzle_label)
        override_layout.addWidget(self.nozzle_override_slider)
        override_layout.addWidget(self.nozzle_override_value_label)
        left_layout.addLayout(override_layout)
        # Keyence profile medians
        keyence_group = QGroupBox("Keyence Profile Medians")
        keyence_layout = QHBoxLayout()
        self.median_base_label = QLabel("base: —")
        self.median_map_label = QLabel("map: —")
        for w in (self.median_base_label, self.median_map_label):
            w.setStyleSheet("border: 1px solid #999; padding: 4px;")
        keyence_layout.addWidget(self.median_base_label)
        keyence_layout.addWidget(self.median_map_label)
        keyence_group.setLayout(keyence_layout)
        left_layout.addWidget(keyence_group)
        # Setup
        setup_group = QGroupBox("Setup Functions"); setup_layout = QVBoxLayout();
        setup_buttons = {
            "Check Status": lambda: start_status_update(self),
            "Launch Drivers": lambda: launch_drivers(self),
            "Prepare Driver Cleanup Channel": lambda: self.ros_interface.prime_driver_cleanup_sessions(),
            "Launch Keyence Scanner": lambda: self.ros_interface.launch_keyence_scanner(),
            "Launch Flow Sensor Bridge": lambda: self.ros_interface.launch_flow_sensor_bridge(),
            "Start Dynamixel Driver": lambda: self.ros_interface.start_dynamixel_driver(),
            "Stop Dynamixel Driver": lambda: self.ros_interface.stop_dynamixel_driver(),
            "Launch Strand Center Camera": lambda: self.ros_interface.launch_strand_center_app(),
            "Open RVIZ": lambda: open_rviz(self),
            "Start Roscore": lambda: self.ros_interface.start_roscore(),
            "Start Mocap": lambda: self.ros_interface.start_mocap(),
            "Start Sync": lambda: self.ros_interface.start_sync(),
        }
        for text, fn in setup_buttons.items():
            b = QPushButton(text); 
            if text=="Launch Keyence Scanner": self.btn_keyence=b
            if text=="Launch Flow Sensor Bridge":
                self.btn_flow_sensor=b
                b.setContextMenuPolicy(Qt.CustomContextMenu)
                b.customContextMenuRequested.connect(self._handle_flow_sensor_right_click)
            if text=="Launch Drivers":
                self.btn_launch_drivers=b
                b.setContextMenuPolicy(Qt.CustomContextMenu)
                b.customContextMenuRequested.connect(self._handle_launch_drivers_right_click)
            if text == "Start Roscore": self.btn_roscore = b
            elif text == "Start Mocap": self.btn_mocap = b
            elif text == "Start Sync": self.btn_sync = b
            b.clicked.connect(lambda _, f=fn: f()); b.setStyleSheet("background-color: lightgray;"); setup_layout.addWidget(b)
        spray_distance_box = QHBoxLayout(); spray_distance_box.addWidget(QLabel("Spray Distance (m):")); self.spray_distance_spin = QDoubleSpinBox(); self.spray_distance_spin.setRange(0.0, 1.0); self.spray_distance_spin.setDecimals(4); self.spray_distance_spin.setSingleStep(0.001); self.spray_distance_spin.setValue(0.62); spray_distance_box.addWidget(self.spray_distance_spin); left_layout.addLayout(spray_distance_box)

        self.workspace_input = QLineEdit(); default_path = self.get_relative_workspace_path(); self.workspace_input.setText(default_path); self.workspace_input.setPlaceholderText("Enter workspace name"); setup_layout.addWidget(QLabel("Workspace Name:")); setup_layout.addWidget(self.workspace_input); setup_group.setLayout(setup_layout); left_layout.addWidget(setup_group)
        main_layout.addLayout(left_layout)
        # Right column
        right_layout = QVBoxLayout(); controller_group = QGroupBox("Controller Functions"); controller_layout = QVBoxLayout(); controller_buttons = {
            "Enable all URs": lambda: enable_all_urs(self),
            "Turn on Arm Controllers": lambda: turn_on_arm_controllers(self),
            "Turn on Twist Controllers": lambda: turn_on_twist_controllers(self),
            "Move to Home Pose Left": lambda: move_to_home_pose(self, "UR10_l"),
            "Move to Home Pose Right": lambda: move_to_home_pose(self, "UR10_r"),
        }
        for text, fn in controller_buttons.items(): btn = QPushButton(text); btn.clicked.connect(lambda _, f=fn: f()); controller_layout.addWidget(btn)
        controller_group.setLayout(controller_layout); right_layout.addWidget(controller_group)
        prepare_print_group = QGroupBox("Prepare Print Functions"); prepare_print_layout = QVBoxLayout(); prepare_print_buttons = {
            "Parse MiR Path": lambda: parse_mir_path(self),
            "Parse UR Path": lambda: parse_ur_path(self),
            "Move MiR to Start Pose": lambda: move_mir_to_start_pose(self),
            "Move UR to Start Pose": lambda: move_ur_to_start_pose(self),
            "Broadcast Target Poses": lambda: target_broadcaster(self),
            "Start Laser Profile Controller": lambda: self.ros_interface.launch_laser_orthogonal_controller(),
        }
        for text, fn in prepare_print_buttons.items():
            btn = QPushButton(text); btn.clicked.connect(lambda _, f=fn: f()); prepare_print_layout.addWidget(btn)
            if text=="Parse MiR Path": self.btn_parse_mir=btn
            if text=="Parse UR Path": self.btn_parse_ur=btn
            if text=="Broadcast Target Poses": self.btn_target_broadcaster=btn
            if text=="Start Laser Profile Controller": self.btn_laser_ctrl=btn
        prepare_print_group.setLayout(prepare_print_layout); right_layout.addWidget(prepare_print_group)

        print_functions_group = QGroupBox("Print Functions"); print_functions_layout = QVBoxLayout(); print_function_buttons = {
            "MiR follow Trajectory": lambda: mir_follow_trajectory(self),
            "Increment Path Index": lambda: increment_path_index(self),
            "Stop MiR Motion": lambda: stop_mir_motion(self),
            "Stop UR Motion": lambda: stop_ur_motion(self),
            "Stop All (Keep Drivers)": lambda: stop_all_but_drivers(self),
        }
        for text, fn in print_function_buttons.items():
            btn = QPushButton(text)
            if text == "Stop All (Keep Drivers)":
                btn.setStyleSheet("background-color: #ff6666; color: black;")
            btn.clicked.connect(lambda _, f=fn: f())
            print_functions_layout.addWidget(btn)
        ur_btn = QPushButton("UR Follow Trajectory"); ur_btn.clicked.connect(lambda _, f=ur_follow_trajectory: f(self, self.ur_follow_settings)); ur_settings_btn = QPushButton("Settings"); ur_settings_btn.clicked.connect(self.open_ur_settings); ur_settings_btn.setStyleSheet("background-color: lightgray;"); hbox = QHBoxLayout(); hbox.addWidget(ur_btn); hbox.addWidget(ur_settings_btn); print_functions_layout.addLayout(hbox)
        # --- Rosbag recording ---
        self.btn_rosbag_record = QPushButton("Rosbag Record"); self.btn_rosbag_record.setStyleSheet("background-color: lightgray;");  self.btn_rosbag_settings = QPushButton("Settings")
        h_rb = QHBoxLayout(); h_rb.addWidget(self.btn_rosbag_record); h_rb.addWidget(self.btn_rosbag_settings);  print_functions_layout.addLayout(h_rb)
        self.btn_rosbag_record.clicked.connect(lambda: self.ros_interface.toggle_rosbag_record());  self.btn_rosbag_settings.clicked.connect(lambda: self.open_rosbag_settings())
        
        idx_box = QHBoxLayout(); idx_box.addWidget(QLabel("Index:")); self.idx_spin = QSpinBox(); self.idx_spin.setRange(0,10000); self.idx_spin.setValue(0); idx_box.addWidget(self.idx_spin); stop_idx_btn = QPushButton("Stop Index Advancer"); stop_idx_btn.clicked.connect(lambda: stop_idx_advancer(self)); idx_box.addWidget(stop_idx_btn); print_functions_layout.addLayout(idx_box)
        
        # Servo section
        servo_box = QGroupBox("Dynamixel Servo Targets"); servo_outer_layout = QVBoxLayout(); targets_row = QHBoxLayout();
        left_col = QVBoxLayout(); left_col.addWidget(QLabel("Left target (%)")); self.servo_left_slider = QSlider(); self.servo_left_slider.setOrientation(Qt.Horizontal); self.servo_left_slider.setRange(0,100); self.servo_left_slider.setTickInterval(10); self.servo_left_slider.setTickPosition(QSlider.TicksBelow); self.servo_left_spin = EnterSpinBox(); self.servo_left_spin.setRange(-100,200); self.servo_left_spin.setValue(0); self.servo_left_slider.valueChanged.connect(self.servo_left_spin.setValue); self.servo_left_spin.valueChanged.connect(lambda v: 0 <= v <= 100 and self.servo_left_slider.setValue(v)); self.servo_left_spin.returnPressed.connect(self._send_percent_targets); left_col.addWidget(self.servo_left_slider); left_col.addWidget(self.servo_left_spin)
        right_col = QVBoxLayout(); right_col.addWidget(QLabel("Right target (%)")); self.servo_right_slider = QSlider(); self.servo_right_slider.setOrientation(Qt.Horizontal); self.servo_right_slider.setRange(0,100); self.servo_right_slider.setTickInterval(10); self.servo_right_slider.setTickPosition(QSlider.TicksBelow); self.servo_right_spin = EnterSpinBox(); self.servo_right_spin.setRange(-100,200); self.servo_right_spin.setValue(0); self.servo_right_slider.valueChanged.connect(self.servo_right_spin.setValue); self.servo_right_spin.valueChanged.connect(lambda v: 0 <= v <= 100 and self.servo_right_slider.setValue(v)); self.servo_right_spin.returnPressed.connect(self._send_percent_targets); right_col.addWidget(self.servo_right_slider); right_col.addWidget(self.servo_right_spin)
        send_col = QVBoxLayout(); send_btn = QPushButton("Send Targets"); send_btn.clicked.connect(self._send_percent_targets); send_col.addWidget(QLabel(" ")); send_col.addWidget(send_btn)
        targets_row.addLayout(left_col); targets_row.addLayout(right_col); targets_row.addLayout(send_col); servo_outer_layout.addLayout(targets_row)
        zero_row = QHBoxLayout(); zl = QVBoxLayout(); zl.addWidget(QLabel("Left zero (raw)")); self.servo_left_zero_spin = QSpinBox(); self.servo_left_zero_spin.setRange(0,4095); self.servo_left_zero_spin.setValue(self.servo_calib['left']['zero']); zl.addWidget(self.servo_left_zero_spin); zr = QVBoxLayout(); zr.addWidget(QLabel("Right zero (raw)")); self.servo_right_zero_spin = QSpinBox(); self.servo_right_zero_spin.setRange(0,4095); self.servo_right_zero_spin.setValue(self.servo_calib['right']['zero']); zr.addWidget(self.servo_right_zero_spin); zb = QVBoxLayout(); send_zero_btn = QPushButton("Send Zero Position"); send_zero_btn.clicked.connect(self._send_zero_positions); calib_btn = QPushButton("Servo Calibration..."); calib_btn.clicked.connect(self.open_servo_calibration); zb.addWidget(QLabel(" ")); zb.addWidget(send_zero_btn); zb.addWidget(calib_btn); zero_row.addLayout(zl); zero_row.addLayout(zr); zero_row.addLayout(zb); servo_outer_layout.addLayout(zero_row); servo_box.setLayout(servo_outer_layout); print_functions_layout.addWidget(servo_box)

        nozzle_group = QGroupBox("Nozzle position")
        nozzle_layout = QHBoxLayout()
        nozzle_layout.addWidget(QLabel("TCP offset (m):"))
        self.tcp_offset_spins = []
        tcp_labels = ['x', 'y', 'z']
        tcp_defaults = [0.0, 0.0, 0.63409]
        for axis, default in zip(tcp_labels, tcp_defaults):
            col = QVBoxLayout()
            col.addWidget(QLabel(axis.upper()))
            spin = QDoubleSpinBox()
            spin.setRange(-2.0, 2.0)
            spin.setDecimals(5)
            spin.setSingleStep(0.001)
            spin.setValue(default)
            spin.setSuffix(" m")
            col.addWidget(spin)
            nozzle_layout.addLayout(col)
            self.tcp_offset_spins.append(spin)
        # orientation around z (phi)
        phi_col = QVBoxLayout()
        phi_col.addWidget(QLabel("φ (deg)"))
        self.tcp_phi_spin = QDoubleSpinBox()
        self.tcp_phi_spin.setRange(-180.0, 180.0)
        self.tcp_phi_spin.setDecimals(3)
        self.tcp_phi_spin.setSingleStep(0.5)
        self.tcp_phi_spin.setValue(88.0)
        self.tcp_phi_spin.setSuffix(" °")
        phi_col.addWidget(self.tcp_phi_spin)
        nozzle_layout.addLayout(phi_col)
        nozzle_group.setLayout(nozzle_layout)
        print_functions_layout.addWidget(nozzle_group)
        print_functions_group.setLayout(print_functions_layout)
        right_layout.addWidget(print_functions_group)
        main_layout.addLayout(right_layout)

        # --- ROS log console on the far right ---
        log_group = QGroupBox("ROS Messages")
        log_layout = QVBoxLayout()

        self.ros_log_text = QTextEdit()
        self.ros_log_text.setReadOnly(True)
        self.ros_log_text.setLineWrapMode(QTextEdit.NoWrap)

        log_layout.addWidget(self.ros_log_text)

        # --- Log level filter checkboxes ---
        filter_layout = QHBoxLayout()
        self.chk_log_error = QCheckBox("Error")
        self.chk_log_warn = QCheckBox("Warning")
        self.chk_log_info = QCheckBox("Info")
        self.chk_log_debug = QCheckBox("Debug")

        # Default: Error/Warn/Info an, Debug aus
        self.chk_log_error.setChecked(True)
        self.chk_log_warn.setChecked(True)
        self.chk_log_info.setChecked(True)
        self.chk_log_debug.setChecked(False)

        # Clear-Button
        self.btn_log_clear = QPushButton("Clear")
        self.btn_log_clear.clicked.connect(self._clear_ros_log)

        # Filter-Checkboxen triggern Ansicht neu
        for cb in (self.chk_log_error, self.chk_log_warn,
                   self.chk_log_info, self.chk_log_debug):
            cb.stateChanged.connect(self._rebuild_ros_log_view)

        filter_layout.addWidget(self.chk_log_error)
        filter_layout.addWidget(self.chk_log_warn)
        filter_layout.addWidget(self.chk_log_info)
        filter_layout.addWidget(self.chk_log_debug)
        filter_layout.addWidget(self.btn_log_clear)

        log_layout.addLayout(filter_layout)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        self.setLayout(main_layout)

        # connect ROS log signal after widgets exist
        self._ros_log_buffer = []
        self.ros_log_signal.connect(self._append_ros_log)

        
        self.setLayout(main_layout)
        # Timer
        self.status_timer = QTimer(); self.status_timer.timeout.connect(self.ros_interface.update_button_status); self.status_timer.start(2000)

    def _handle_launch_drivers_right_click(self, _pos):
        """Stop driver terminals when the launch button is right-clicked."""
        quit_drivers(self)

    def _handle_flow_sensor_right_click(self, _pos):
        """Stop the flow sensor bridge when its button is right-clicked."""
        self.ros_interface.stop_flow_sensor_bridge()

    def _load_servo_calibration_defaults(self):
        """Load servo calibration defaults from config, falling back to baked values."""
        base_defaults = {
            'left': {'min': 2800, 'zero': 2300, 'max': 3357},
            'right': {'min': 961, 'zero': 1461, 'max': 404},
        }

        def clone_defaults(src):
            return {side: dict(values) for side, values in src.items()}

        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "servo_calibration_defaults.json")
        )

        try:
            with open(config_path, "r", encoding="utf-8") as cfg:
                file_data = json.load(cfg)
        except FileNotFoundError:
            print(f"Servo calibration config not found at {config_path}. Using built-in defaults.")
            return clone_defaults(base_defaults)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Failed to load servo calibration config: {exc}. Using built-in defaults.")
            return clone_defaults(base_defaults)

        if not isinstance(file_data, dict):
            print("Servo calibration config is not a mapping. Using built-in defaults.")
            return clone_defaults(base_defaults)

        merged = clone_defaults(base_defaults)
        for side in ('left', 'right'):
            side_data = file_data.get(side)
            if not isinstance(side_data, dict):
                continue
            for key in ('min', 'zero', 'max'):
                value = side_data.get(key)
                if isinstance(value, (int, float)):
                    merged[side][key] = int(value)

        return merged

    def _percent_to_raw(self, percent: float, which: str) -> int:
        c = self.servo_calib[which]
        span = c['max'] - c['min']
        if span == 0:
            return int(c['min'])
        raw = c['min'] + (percent / 100.0) * span
        return max(0, min(4095, int(round(raw))))

    def _send_percent_targets(self):
        left_p = self.servo_left_spin.value()
        right_p = self.servo_right_spin.value()
        left_raw = self._percent_to_raw(left_p, 'left')
        right_raw = self._percent_to_raw(right_p, 'right')
        self.ros_interface.publish_servo_targets(left_raw, right_raw)

    def _send_zero_positions(self):
        # Use currently stored zero raw values (spin boxes display them)
        self.ros_interface.publish_servo_targets(
            int(self.servo_left_zero_spin.value()),
            int(self.servo_right_zero_spin.value())
        )

    def open_servo_calibration(self):
        dlg = ServoCalibrationDialog(self, self.servo_calib)
        if dlg.exec_() == QDialog.Accepted:
            self.servo_calib = dlg.get_values()
            # update zero spin boxes
            self.servo_left_zero_spin.setValue(self.servo_calib['left']['zero'])
            self.servo_right_zero_spin.setValue(self.servo_calib['right']['zero'])
            # Optionally reset sliders/spins to zero percent
            # self.servo_left_slider.setValue(0); self.servo_left_spin.setValue(0)
            # self.servo_right_slider.setValue(0); self.servo_right_spin.setValue(0)

    @pyqtSlot(int)
    def _update_spinbox(self, idx):
        # guaranteed to run in Qt (GUI) thread
        self.idx_spin.setValue(idx)
    
    @pyqtSlot(float, float)
    def _update_medians(self, med_base: float, med_map: float):
        # guaranteed to run in Qt (GUI) thread
        def fmt(v: float) -> str:
            return "—" if (v != v) or math.isinf(v) else f"{v:.3f} m"
        self.median_base_label.setText(f"base: {fmt(med_base)}")
        self.median_map_label.setText(f"map: {fmt(med_map)}")

    def _ros_log_level_enabled(self, level: str) -> bool:
        level = level.upper()
        if level == "ERROR":
            return self.chk_log_error.isChecked()
        if level in ("WARN", "WARNING"):
            return self.chk_log_warn.isChecked()
        if level == "INFO":
            return self.chk_log_info.isChecked()
        if level == "DEBUG":
            return self.chk_log_debug.isChecked()
        # Unbekannt – behandel wie INFO
        return self.chk_log_info.isChecked()


    @pyqtSlot(str, str, str)
    def _append_ros_log(self, level: str, node: str, text: str):
        """Append one ROS log entry to the buffer and refresh view."""
        if not hasattr(self, "_ros_log_buffer"):
            self._ros_log_buffer = []

        # Buffer enthält Tuples
        self._ros_log_buffer.append((level, node, text))
        self._ros_log_buffer = self._ros_log_buffer[-400:]

        self._rebuild_ros_log_view()


    def _rebuild_ros_log_view(self):
        """Rebuild the log text widget from the buffer, applying filters + colors."""
        html_lines = []

        for level, node, text in self._ros_log_buffer:
            if not self._ros_log_level_enabled(level):
                continue

            line = f"[{level}] {node}: {text}"
            escaped = html.escape(line)

            lvl = level.upper()
            if lvl in ("WARN", "WARNING"):
                escaped = escaped.replace(
                    "[WARN]",
                    '<span style="color:#d9a400; font-weight:bold;">[WARN]</span>'
                )
            elif lvl == "ERROR":
                escaped = escaped.replace(
                    "[ERROR]",
                    '<span style="color:#c00000; font-weight:bold;">[ERROR]</span>'
                )

            html_lines.append(escaped)

        self.ros_log_text.setHtml("<br>".join(html_lines))
        sb = self.ros_log_text.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())


    def _clear_ros_log(self):
        """Clear all buffered log messages and the view."""
        self._ros_log_buffer = []
        self.ros_log_text.clear()
        
    def open_ur_settings(self):
        dlg = URFollowSettingsDialog(self, initial_settings=self.ur_follow_settings)
        if dlg.exec_() == QDialog.Accepted:
            # Retrieve and store new settings
            self.ur_follow_settings = dlg.getValues()
            print("New UR Follow settings:", self.ur_follow_settings)

    def set_idx_metric(self, text):
            self.idx_metric = text

    def get_full_workspace_path(self):
        import os
        current_path = os.path.abspath(__file__)

        while current_path != "/":
            if os.path.basename(current_path) == "src":
                return os.path.dirname(current_path)  # Absoluter Pfad zum Workspace
            current_path = os.path.dirname(current_path)

        return os.path.expanduser("~/catkin_ws")  # Fallback


    def get_relative_workspace_path(self):
        full_path = self.get_full_workspace_path()
        home_path = os.path.expanduser("~")
        if full_path.startswith(home_path):
            return os.path.relpath(full_path, home_path)
        return full_path

    def update_virtual_object_pose(self, pose):
        """Updates the GUI table with the latest virtual object pose."""
        for col in range(6):
            self.table.setItem(8, col, QTableWidgetItem(str(round(pose[col], 4))))
    
    def get_selected_robots(self):
        return [name for name, checkbox in self.robots.items() if checkbox.isChecked()]

    def get_selected_urs(self):
        ur_prefixes = []
        if self.ur10_l.isChecked():
            ur_prefixes.append("UR10_l")
        if self.ur10_r.isChecked():
            ur_prefixes.append("UR10_r")
        return ur_prefixes

    def get_workspace_name(self):
        return self.workspace_input.text().strip()

    def get_override_value(self):
        return self.override_slider.value()

    def is_debug_enabled(self):
        chk = getattr(self, "chk_log_debug", None)
        return bool(chk and chk.isChecked())

    def get_tcp_offset_xyz(self):
        if not hasattr(self, 'tcp_offset_spins'):
            return [0.0, 0.0, 0.0]
        return [spin.value() for spin in self.tcp_offset_spins]

    def get_tcp_offset_sixd(self):
        xyz = self.get_tcp_offset_xyz()
        phi = self.get_tcp_phi_radians()
        return xyz + [0.0, 0.0, phi]

    def get_tcp_phi_radians(self):
        if not hasattr(self, 'tcp_phi_spin'):
            return 0.0
        return math.radians(self.tcp_phi_spin.value())

    def get_spray_distance(self):
        return self.spray_distance_spin.value()

    def open_rosbag_settings(self):
        dlg = RosbagSettingsDialog(self,
            topics=self.ros_interface.rosbag_topics,
            enabled=self.ros_interface.rosbag_enabled)
        if dlg.exec_() == QDialog.Accepted:
            self.ros_interface.rosbag_enabled = dlg.get_enabled()


class URFollowSettingsDialog(QDialog):
    def __init__(self, parent=None, initial_settings=None):
        super().__init__(parent)
        self.setWindowTitle("UR Follow Trajectory Settings")

        # Default settings
        init = initial_settings or {}
        idx_metrics = ["virtual line", "radius", "collinear"]
        idx_metric = init.get('idx_metric', idx_metrics[0])
        threshold = init.get('threshold', 0.010)

        form = QFormLayout()        
        self.dropdown_idx_metric = QComboBox()
        self.dropdown_idx_metric.addItems(idx_metrics)
        self.dropdown_idx_metric.setCurrentText(idx_metric)  # Set default text
        self.dropdown_idx_metric.setStyleSheet("background-color: lightgray;")
        # self.dropdown_idx_metric.currentTextChanged.connect(lambda text: self.set_idx_metric(text))
        form.addRow("Index Metric:", self.dropdown_idx_metric)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.0, 0.2)  # Set range for the spin box
        self.spin_threshold.setSingleStep(0.002)
        self.spin_threshold.setDecimals(3)  # Set number of decimal places
        self.spin_threshold.setValue(threshold)
        self.spin_threshold.setSuffix(" m")
        self.spin_threshold.setStyleSheet("background-color: lightgray;")
        form.addRow("Threshold:", self.spin_threshold)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def getValues(self):
        return {
            'idx_metric': self.dropdown_idx_metric.currentText(),
            'threshold': self.spin_threshold.value(),
        }


class ServoCalibrationDialog(QDialog):
    def __init__(self, parent, calib):
        super().__init__(parent)
        self.setWindowTitle("Servo Calibration")
        self._orig = calib
        self._data = {
            'left': calib['left'].copy(),
            'right': calib['right'].copy()
        }
        form = QFormLayout()
        # Left
        self.left_min = QSpinBox(); self.left_min.setRange(0,4095); self.left_min.setValue(self._data['left']['min'])
        self.left_zero = QSpinBox(); self.left_zero.setRange(0,4095); self.left_zero.setValue(self._data['left']['zero'])
        self.left_max = QSpinBox(); self.left_max.setRange(0,4095); self.left_max.setValue(self._data['left']['max'])
        form.addRow(QLabel("Left Min"), self.left_min)
        form.addRow(QLabel("Left Zero"), self.left_zero)
        form.addRow(QLabel("Left Max"), self.left_max)
        # Right
        self.right_min = QSpinBox(); self.right_min.setRange(0,4095); self.right_min.setValue(self._data['right']['min'])
        self.right_zero = QSpinBox(); self.right_zero.setRange(0,4095); self.right_zero.setValue(self._data['right']['zero'])
        self.right_max = QSpinBox(); self.right_max.setRange(0,4095); self.right_max.setValue(self._data['right']['max'])
        form.addRow(QLabel("Right Min"), self.right_min)
        form.addRow(QLabel("Right Zero"), self.right_zero)
        form.addRow(QLabel("Right Max"), self.right_max)
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _on_accept(self):
        # Basic validation: min <= zero <= max
        if not (self.left_min.value() <= self.left_zero.value() <= self.left_max.value()):
            # silently clamp
            z = min(max(self.left_zero.value(), self.left_min.value()), self.left_max.value())
            self.left_zero.setValue(z)
        if not (self.right_min.value() <= self.right_zero.value() <= self.right_max.value()):
            z = min(max(self.right_zero.value(), self.right_min.value()), self.right_max.value())
            self.right_zero.setValue(z)
        self.accept()

    def get_values(self):
        return {
            'left': {
                'min': self.left_min.value(),
                'zero': self.left_zero.value(),
                'max': self.left_max.value(),
            },
            'right': {
                'min': self.right_min.value(),
                'zero': self.right_zero.value(),
                'max': self.right_max.value(),
            }
        }

class RosbagSettingsDialog(QDialog):
    def __init__(self, parent=None, topics=None, enabled=None):
        super().__init__(parent); self.setWindowTitle("Rosbag Topics")
        self.topics = list(topics) if topics else []
        enabled_map = enabled or {}
        self.enabled = {t: bool(enabled_map.get(t, True)) for t in self.topics}

        self.boxes = {}
        layout = QVBoxLayout()
        for t in self.topics:
            cb = QCheckBox(t); cb.setChecked(self.enabled.get(t, True)); layout.addWidget(cb); self.boxes[t] = cb
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns); self.setLayout(layout)

    def get_enabled(self):
        return {t: cb.isChecked() for t, cb in self.boxes.items()}
