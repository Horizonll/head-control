#!/usr/bin/env python3

import threading, sys
import time
import numpy as np
from try_import import *
import configuration


class HeadControl(Node):
    def __init__(self):
        super().__init__("head_control", namespace="THMOS")
        self.logger = self.get_logger()
        self.logger.info("Head control node initialized")
        self.config = configuration.load_config()

        self.best_ball = None
        self.head_pose = [0, 0]
        self._manual_control = np.array([np.inf, np.inf])

        self._vision_sub = self.create_subscription(
            VisionDetections, "/THMOS/vision/obj_pos", self._vision_cb, 1
        )

        self._head_pose_pub = self.create_publisher(
            JointState, "/THMOS/hardware/set_head_pose", 1
        )

        self._manual_control_sub = self.create_subscription(
            JointState, "/THMOS/head_control/manual_command", self._manual_control_cb, 1
        )
        self.last_ball_time = time.time()
        self.logger.info("Subscribers and publishers initialized")
        self._set_head_pose([0, 0])
        self.ball_coord = [640, 368]
        self.max_yaw = self.config.get("max_yaw")
        self.max_pitch = self.config.get("max_pitch")
        self._look_for_ball_thread = threading.Thread(target=self.head)
        self._look_for_ball_thread.daemon = True
        self._look_for_ball_thread.start()
        self.yaw = [i for i in np.linspace(self.max_yaw[0], self.max_yaw[1], 100)]

    def head(self):
        while True:
            self.run()

    def run(self):
        if self.best_ball is not None:
            self.last_ball_time = time.time()
            self.ball_coord = [
                self.best_ball.xmin
                + (self.best_ball.xmax - self.best_ball.xmin) / 2,
                self.best_ball.ymin
                + (self.best_ball.ymax - self.best_ball.ymin) / 2,
            ]
            if abs(self.ball_coord[0] - 640) > 500:
                self.head_pose[0] -= (self.ball_coord[0] - 640) * 0.00002
            elif abs(self.ball_coord[0] - 640) > 100:
                self.head_pose[0] -= (self.ball_coord[0] - 640) * 0.000018
            if abs(self.ball_coord[1] - 360) > 300:
                self.head_pose[1] += (self.ball_coord[1] - 368) * 0.00002
            elif abs(self.ball_coord[1] - 360) > 100:
                self.head_pose[1] += (self.ball_coord[1] - 368) * 0.000018
            self._set_head_pose(self.head_pose)
        elif time.time() - self.last_ball_time > 1:
            while True:
                for i in self.yaw:
                    t1 = time.time()
                    self._set_head_pose([i, self.max_pitch[1]])
                    while time.time() - t1 < 0.01:
                        if self.best_ball is not None:
                            return
                for i in self.yaw[::-1]:
                    t1 = time.time()
                    self._set_head_pose([i, self.max_pitch[0]])
                    while time.time() - t1 < 0.01:
                        if self.best_ball is not None:
                            return

    def _vision_cb(self, msg: VisionDetections):
        ball_objects = [obj for obj in msg.detected_objects if obj.label == "ball"]
        self.best_ball = (
            max(ball_objects, key=lambda obj: obj.confidence) if ball_objects else None
        )

    def _set_head_pose(self, target):
        """Publish head pose control command"""
        logger = self.logger.get_child("set_head_pose")
        try:
            msg = JointState()
            msg.header.frame_id = "head_control"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["head_yaw_joint", "head_pitch_joint"]
            msg.position = [0.0, 0.0]  # Initialize with zeros
            msg.position[0] = float(
                np.clip(target[0], self.max_yaw[0], self.max_yaw[1])
            )  # 使用范围的最小值和最大值
            msg.position[1] = float(
                np.clip(target[1], self.max_pitch[0], self.max_pitch[1])
            )  # 使用范围的最小值和最大值
            self.head_pose = msg.position
            if not np.isinf(self._manual_control[0]):
                msg.position[0] = self._manual_control[0]
            if not np.isinf(self._manual_control[1]):
                msg.position[1] = self._manual_control[1]
            logger.info(f"FUCK {msg.position[0]} {msg.position[1]}")
            self._head_pose_pub.publish(msg)

            # Output debug information
            logger.info(
                f"Publishing head pose: yaw={msg.position[0]}, pitch={msg.position[1]}"
            )
        except Exception as e:
            logger.error(f"Error publishing head pose command: {str(e)}")

    def _manual_control_cb(self, msg: JointState):
        """Handle manual control commands using HeadPose messages"""

        logger = self.logger.get_child("manual_control_cb")
        try:
            yaw = msg.position[0] if len(msg.position) > 0 else 0.0
            pitch = msg.position[1] if len(msg.position) > 1 else 0.0
            self._manual_control = np.array([yaw, pitch])
            # self._set_head_pose(self._manual_control)
            # logger.info(
            #     f"Received manual control signal - "
            #     + f"Set head pose: Yaw={yaw}, Pitch={pitch}"
            # )
        except Exception as e:
            logger.error(f"Error handling manual control command: {str(e)}")

    def destroy_node(self):
        super().destroy_node()
        self.logger.info("Head control node destroyed")


if __name__ == "__main__":
    rclpy.init(args=sys.argv)
    node = HeadControl()
    rclpy.spin(node)
    try:
        pass
    except KeyboardInterrupt:
        node.logger.info("Node interrupted by user")
    except Exception as e:
        if "node" in locals() and node:
            node.logger.error(f"Error in node execution: {str(e)}")
    finally:
        if "node" in locals() and node:
            node.destroy_node()
        rclpy.shutdown()
