#!/usr/bin/env python3

import threading, math, time, sys, os
import numpy as np
from datetime import datetime
from collections import deque

from try_import import *


class HeadControl(Node):
    def __init__(self):
        super().__init__("head_control", namespace="THMOS")
        self.logger = self.get_logger()
        self.logger.info("Head control node initialized")

        # TODO: upgrade to overridable yaml config
        self.config = {
            "image_size": [1280, 736],
            "ball_target_position": [0.5, 0.5],
            "frequence": 1,
            "max_yaw": [-1, 1],  # clamp to [min, max]
            "max_pitch": [-0.785, 0.785],  # clamp [min, max]
            "converge_to_ball_P": [0.6, 0.6],
            "converge_to_ball_D": [0.05, 0.05],
            "input_filter_alpha": 0.5,  # exp-filter args, higher is smoother
            "EWMA_ball_confidence_factor": 0.9,
            "EWMA_ball_confidence_threshold": 0.5,
            "look_for_ball_point": [
                [[-0.7, 0.0], 0.5],
                [[0.0, 0.0], 0.5],
                [[0.7, 0.0], 0.5],
                [[0.7, 1.0], 0.5],
                [[0.0, 1.0], 0.5],
                [[-0.7, 1.0], 0.5],
            ],  # pitch, yaw, duration
        }
        self.ball_target_coord = np.array(self.config.get("image_size")) * np.array(
            self.config.get("ball_target_position")
        )

        # Initialize data storage with deque for efficient operations
        self._head_pose_db = deque(maxlen=100)  # Use deque instead of list
        self._last_stamp = self.get_clock().now().to_msg()
        self._prev_stamp = self.get_clock().now().to_msg()
        self._last_error = np.array([0, 0])
        self._prev_error = np.array([0, 0])  # Store previous error
        self._manual_control = np.array([np.inf, np.inf])
        self._filtered_ball_coord = None
        self.EWMA_ball_confidence = 0.0

        # Create subscribers and publishers
        self._vision_sub = self.create_subscription(
            VisionDetections, "/THMOS/vision/obj_pos", self._vision_cb, 1
        )

        self._head_pose_pub = self.create_publisher(
            JointState, "/THMOS/hardware/set_head_pose", 1
        )

        self._head_pose_sub = self.create_subscription(
            JointState, "/THMOS/hardware/get_head_pose", self._head_pose_cb, 1
        )

        self._manual_control_sub = self.create_subscription(
            JointState, "/THMOS/head_control/manual_command", self._manual_control_cb, 1
        )

        self.logger.info("Subscribers and publishers initialized")
        self._set_head_pose([0, 0])

        self._look_for_ball_thread = threading.Thread(target=self._look_for_ball_loop)
        self._look_for_ball_thread.daemon = True
        self._look_for_ball_thread.start()

    def _PID_to_ball(self, best_ball: VisionObj, head_pose, timestamp) -> None:
        "PID converge to ball"
        logger = self.logger.get_child("PID_to_ball")

        # verify head pose and ball
        if head_pose is None or best_ball is None:
            return

        # Compute the coordination of the ball in vision
        # with exponential filter for ball coordinate
        alpha = self.config.get("input_filter_alpha")
        ball_coord = 0.5 * (
            np.array([best_ball.xmin, best_ball.ymin])
            + np.array([best_ball.xmax, best_ball.ymax])
        )
        if self._filtered_ball_coord is None:
            self._filtered_ball_coord = ball_coord
        else:
            self._filtered_ball_coord = (
                alpha * self._filtered_ball_coord + (1 - alpha) * ball_coord
            )

        # Compute normalized error
        error = (self._filtered_ball_coord - self.ball_target_coord) / np.array(
            self.config.get("image_size")
        )
        error *= np.array([1, -1])
        # HAVE TO alter the direction

        # Save current detection information
        # and calculate the derivative of error
        self._prev_stamp, self._last_stamp = self._last_stamp, timestamp
        self._prev_error, self._last_error = self._last_error, error
        time_diff = max(self._get_time_diff(self._prev_stamp, self._last_stamp), 1e-9)
        error_d = (self._last_error - self._prev_error) / time_diff

        # PID output as delta (control, not control change)
        pid_P = self.config.get("converge_to_ball_P")
        pid_D = self.config.get("converge_to_ball_D")
        control_delta = -(self._last_error * pid_P + error_d * pid_D)

        # Add up to the headpose
        self._set_head_pose(head_pose + control_delta)

    def _update_ball_confidence(self, best_ball: VisionObj):
        "update ball confidence"
        logger = self.logger.get_child("update_ball_confidence")
        self.EWMA_ball_confidence *= self.config.get("EWMA_ball_confidence_factor")
        if best_ball:
            self.EWMA_ball_confidence += best_ball.confidence

    def _look_for_ball_loop(self):
        "looking for ball if ball is lost"
        logger = self.logger.get_child("look_for_ball_loop")
        point_with_time = self.config.get("look_for_ball_point")
        n = len(point_with_time)
        point = np.fromiter(
            (xy for x in point_with_time for xy in x[0]), dtype=float, count=n * 2
        ).reshape(n, 2)
        gap_time = np.fromiter((x[1] for x in point_with_time), dtype=float, count=n)
        stime = np.cumsum(gap_time)
        loop_index = 0
        while True:
            if self.EWMA_ball_confidence > self.config.get(
                "EWMA_ball_confidence_threshold"
            ):
                time.sleep(1.0)
                continue
            loop_time = math.fmod(time.time_ns() / 1e9, stime[-1])
            while stime[loop_index] < loop_time:
                loop_index += 1
            if stime[loop_index - 1] > loop_time:
                loop_index = 0
            target = point[loop_index]
            self._set_head_pose(target)
            time.sleep(0.05)

    def _vision_cb(self, msg: VisionDetections):
        "call back function of vision"
        # Filter objects labeled "Ball" and find the one with highest confidence
        # and find the best ball with highest confidence
        ball_objects = [obj for obj in msg.detected_objects if obj.label == "ball"]
        best_ball = (
            max(ball_objects, key=lambda obj: obj.confidence) if ball_objects else None
        )

        self._PID_to_ball(
            best_ball=best_ball,
            head_pose=self._get_head_pose(msg.header.stamp),
            timestamp=msg.header.stamp,
        )
        self._update_ball_confidence(best_ball)

    def _head_pose_cb(self, msg: JointState):
        yaw = msg.position[0] if len(msg.position) > 0 else 0.0
        pitch = msg.position[1] if len(msg.position) > 1 else 0.0
        self._head_pose_db.append([msg.header.stamp, np.array([yaw, pitch])])

    def _get_head_pose(self, target_stamp):
        # DO NOT RETURN DEFAULT POSITION. WILL CAUSE INCORRECT CONTROL SIGNAL.
        # Just return None, we'll (and we have to) handle it
        logger = self.logger.get_child("get_head_pose")
        if not self._head_pose_db:
            return None

        # TODO: Return real interpolation after the timestamp in vision
        # represent the time WHEN THE PICTURE WAS TANKEN precisely
        # return self._head_pose_db[0][1]

        # Convert ROS time message to Time object
        target_time = Time.from_msg(target_stamp)

        # Remove outdated data points (optimize boundary condition check)
        while len(self._head_pose_db) > 1:
            next_stamp = self._head_pose_db[1][0]
            next_time = Time.from_msg(next_stamp)
            if next_time <= target_time:
                logger.debug(
                    f"Removing outdated head pose data: {self._head_pose_db[0][0]}"
                )
                self._head_pose_db.popleft()  # Use popleft() for deque to improve efficiency
            else:
                break

        # Linear interpolation to get pose at target time
        if len(self._head_pose_db) > 1:
            t1, pose1 = self._head_pose_db[0]
            t2, pose2 = self._head_pose_db[1]

            # Convert timestamps to Time objects
            t1_time = Time.from_msg(t1)
            t2_time = Time.from_msg(t2)

            # logger.debug(f"Interpolating between: t1={t1}, pose1={pose1}, t2={t2}, pose2={pose2}")

            time_diff_total = self._get_time_diff(t1, t2)
            time_diff_total = max(time_diff_total, 1e-9)  # Avoid division by zero error

            time_diff = self._get_time_diff(t1, target_stamp)
            ratio = min(max(time_diff / time_diff_total, 0.0), 1.0)

            logger.debug(f"Interpolation ratio: {ratio}")

            return pose1 + (pose2 - pose1) * ratio

        # Return latest pose if interpolation not possible
        logger.debug(f"Using latest head pose: {self._head_pose_db[0][1]}")
        return self._head_pose_db[0][1]

    def _get_time_diff(self, stamp1, stamp2):
        """Calculate time difference between two timestamps (seconds)"""
        time1 = Time.from_msg(stamp1)
        time2 = Time.from_msg(stamp2)
        diff = (time2 - time1).nanoseconds * 1e-9  # + (time2 - time1).seconds
        return diff

    def _set_head_pose(self, target):
        """Publish head pose control command"""
        logger = self.logger.get_child("set_head_pose")
        try:
            msg = JointState()
            msg.header.frame_id = "head_control"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["head_yaw_joint", "head_pitch_joint"]
            msg.position = [0.0, 0.0]  # Initialize with zeros
            max_yaw = self.config.get("max_yaw")
            max_pitch = self.config.get("max_pitch")
            msg.position[0] = float(
                np.clip(target[0], max_yaw[0], max_yaw[1])
            )  # 使用范围的最小值和最大值
            msg.position[1] = float(
                np.clip(target[1], max_pitch[0], max_pitch[1])
            )  # 使用范围的最小值和最大值
            print(self._manual_control[0])
            if not np.isinf(self._manual_control[0]):
                msg.position[0] = self._manual_control[0]
            if not np.isinf(self._manual_control[1]):
                msg.position[1] = self._manual_control[1]
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
            logger.info(
                f"Received manual control signal - "
                + f"Set head pose: Yaw={yaw}, Pitch={pitch}"
            )
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
