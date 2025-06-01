#! python3

import sys
import numpy as np
from try_import import *

class HeadControl(Node):
    def __init__(self):
        super().__init__("head_control", namespace = "THMOS")
        self.logger = self.get_logger()

        # parameters
        self.declare_parameter("target_ball_pos", [640, 360])
        self.declare_parameter("converge_to_ball_P", [1, 1])
        self.declare_parameter("converge_to_ball_D", [1, 1])


        # subscriptions and publishers
        self._vision_sub = self.create_subscription(VisionDetections, \
                                "THMOS/vision/detections", \
                                self._vision_cb, 1)
        self._head_pose_pub = self.create_publisher(HeadPose, \
                                "THMOS/hardware/set_head_pose", 1)
        self._head_pose_sub = self.create_publisher(HeadPose, \
                                "THMOS/hardware/set_head_pose", 1)

        self._set_head_pose(np.array([0, 1]))

    def _vision_cb(self, msg: VisionDetections):
        head_pose = self._get_head_pose(msg.header.stamp)
        curr_coord = (  np.array([msg.xmin, msg.ymin]) + \
                        np.array([msg.xmax, msy.ymax]) ) * 0.5
        print(f"current: {curr_coord} head: {head_pose}")
        if not hasattr(self, "_last_coord"):
            deri_coord = np.array([0, 0])
        else:
            stamp_diff = self._last_stamp - msg.header.stamp
            time_diff = stamp_diff.seconds + stamp_diff.nanoseconds * 1e-9;
            deri_coord = (curr_coord - self._last_coord) / time_diff
        pid_P = np.array(self.get_parameter("converge_to_ball_P").value)
        pid_D = np.array(self.get_parameter("converge_to_ball_D").value)
        head_pose += curr_coord * pid_P + deri_coord * pid_D
        print(f" control: {head_pose}")
        self._set_head_pose(head_pose)
        self._last_coord = curr_coord
            
        

    def _head_pose_cb(self, msg: HeadPose):
        self._head_pose_db.append([msg.stamp, np.array([msg.yaw, msg.pitch])])


    def _get_head_pose(self, detection_stamp):
        while not self._head_pose_db:
            if self._head_pose_db[0][0] >= detection_stamp:
                break
            self._head_pose_db.pop()
        return self._head_pose.db[0][1]


    def _set_head_pose(self, target):
        msg = HeadPose()
        msg.header.frame_id = "head_control"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.yaw = float(target[0])
        msg.pitch = float(target[1])
        self._head_pose_pub.publish(msg)

if __name__ == '__main__':
    rclpy.init(args = sys.argv)
    node = HeadControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

