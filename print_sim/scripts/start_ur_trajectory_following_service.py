#!/usr/bin/env python3

import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse


class StartUrTrajectoryFollowingService:
    def __init__(self):
        self.robot_name = str(rospy.get_param("~robot_name", "mur620a")).strip("/")
        self.prefix_ur = str(rospy.get_param("~prefix_ur", "UR10_r/")).strip("/")
        self.start_condition_topic = rospy.get_param("~start_condition_topic", "/start_condition")
        self.service_name = rospy.get_param("~service_name", "/start_trajectory_following")
        self.controller_manager_service = rospy.get_param(
            "~controller_manager_service",
            f"/{self.robot_name}/controller_manager/switch_controller",
        )

        side = "r" if self.prefix_ur.endswith("_r") else "l"
        self.start_controllers = self._as_string_list(
            rospy.get_param("~start_controllers", [f"joint_group_vel_controller_{side}/unsafe"])
        )
        self.stop_controllers = self._as_string_list(
            rospy.get_param("~stop_controllers", [f"{self.prefix_ur}/arm_controller"])
        )
        self.strictness = self._as_int(rospy.get_param("~strictness", 2), default=2)

        self.start_pub = rospy.Publisher(self.start_condition_topic, Bool, queue_size=1, latch=True)
        self.switch_client = rospy.ServiceProxy(self.controller_manager_service, SwitchController)

        rospy.loginfo(f"Waiting for controller switch service: {self.controller_manager_service}")
        self.switch_client.wait_for_service()

        self.service = rospy.Service(self.service_name, Trigger, self._handle_start)
        rospy.loginfo(
            f"Ready. Call service '{self.service_name}' to switch UR controller and start trajectory following."
        )

    @staticmethod
    def _as_string_list(value):
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, tuple):
            return [str(v) for v in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if "," in stripped:
                return [item.strip() for item in stripped.split(",") if item.strip()]
            return [stripped]
        return [str(value)]

    @staticmethod
    def _as_int(value, default=2):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _handle_start(self, _req):
        request = SwitchControllerRequest()
        request.start_controllers = list(self.start_controllers)
        request.stop_controllers = list(self.stop_controllers)
        request.strictness = self.strictness

        try:
            response = self.switch_client(request)
        except rospy.ServiceException as exc:
            msg = f"Controller switch failed: {exc}"
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

        if not response.ok:
            msg = (
                "Controller switch returned ok=false "
                f"(start={self.start_controllers}, stop={self.stop_controllers})"
            )
            rospy.logerr(msg)
            return TriggerResponse(success=False, message=msg)

        self.start_pub.publish(Bool(data=True))
        msg = (
            "Trajectory following started. "
            f"Switched controllers (start={self.start_controllers}, stop={self.stop_controllers}) "
            f"and published True on {self.start_condition_topic}."
        )
        rospy.loginfo(msg)
        return TriggerResponse(success=True, message=msg)


if __name__ == "__main__":
    rospy.init_node("start_ur_trajectory_following_service")
    StartUrTrajectoryFollowingService()
    rospy.spin()
