#!/usr/bin/env python3

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from try_import import *
from collections import deque

class HeadControl(Node):
    def __init__(self):
        super().__init__("head_control", namespace="THMOS")
        self.logger = self.get_logger()
        self.logger.info("Head control node initialized")

        # 频率计算相关变量
        self._control_timestamps = deque(maxlen=100)  # 存储最近100次控制时间戳
        self._detection_timestamps = deque(maxlen=100)  # 存储最近100次检测时间戳
        
        # Initialize parameters
        self.declare_parameter("image_width", 672.0)
        self.declare_parameter("image_height", 376.0)
        self.declare_parameter("converge_to_ball_P", [0.4, 0.4])
        self.declare_parameter("converge_to_ball_D", [0.0, 0.0])
        self.declare_parameter("max_yaw", 1.5)
        self.declare_parameter("max_pitch", 1.0)
        self.declare_parameter("auto_mode_flag", float('inf'))
        
        # Get parameters
        self.image_width = self.get_parameter("image_width").value
        self.image_height = self.get_parameter("image_height").value
        self.max_yaw = self.get_parameter("max_yaw").value
        self.max_pitch = self.get_parameter("max_pitch").value
        self.auto_mode_flag = self.get_parameter("auto_mode_flag").value
        
        # Initialize data storage
        self._head_pose_db = []
        self._last_coord = None
        self._last_stamp = None
        self._last_error = None
        self._manual_target = np.array([0.0, 0.5])  # Default manual target
        self._control_mode = "auto"  # Default to auto mode
        
        # Create subscribers and publishers
        self._vision_sub = self.create_subscription(
            VisionDetections,
            "vision/detections",
            self._vision_cb,
            1)
        
        self._head_pose_pub = self.create_publisher(
            HeadPose,
            "hardware/set_head_pose",
            1)
            
        self._head_pose_sub = self.create_subscription(
            HeadPose,
            "hardware/get_head_pose",
            self._head_pose_cb,
            1)
            
        self._manual_control_sub = self.create_subscription(
            HeadPose,
            "head_control/manual_command",
            self._manual_control_cb,
            1)

        # 创建定时器用于输出频率信息
        self._freq_timer = self.create_timer(5.0, self._print_frequencies)
        
        self.logger.info("Subscribers and publishers initialized")

        # Set initial head pose
        self._set_head_pose(self._manual_target)

    def _vision_cb(self, msg: VisionDetections):
        current_time = self.get_clock().now()
        self._detection_timestamps.append(current_time)
        
        # 输出调试信息
        self.logger.debug(f"Vision detection received at: {current_time}")
        self.logger.debug(f"Detection message: xmin={msg.xmin}, ymin={msg.ymin}, xmax={msg.xmax}, ymax={msg.ymax}")
        self.logger.debug(f"Number of detected objects: {len(msg.detected_objects)}")
        
        # Process vision data only in auto mode
        if self._control_mode != "auto":
            self.logger.debug("Skipping vision processing in non-auto mode")
            return
            
        # 筛选label为Ball且置信度最高的目标
        ball_objects = [obj for obj in msg.detected_objects if obj.label == "Ball"]
        if not ball_objects:
            self.logger.info("No ball objects detected")
            return
            
        # 找到置信度最高的球
        best_ball = max(ball_objects, key=lambda obj: obj.confidence)
        self.logger.info(f"Best ball detected - Confidence: {best_ball.confidence}")
        
        # Validate message
        if best_ball.xmin >= best_ball.xmax or best_ball.ymin >= best_ball.ymax:
            self.logger.warning("Invalid bounding box received for best ball")
            return
            
        # Calculate target center coordinates
        curr_coord = (np.array([best_ball.xmin, best_ball.ymin]) + 
                      np.array([best_ball.xmax, best_ball.ymax])) * 0.5
        
        # Calculate normalized error
        center_x = self.image_width / 2
        center_y = self.image_height / 2 + 100
        error = np.array([
            (curr_coord[0] - center_x) / self.image_width,
            (curr_coord[1] - center_y) / self.image_height
        ])
                
        # Get current head pose
        head_pose = self._get_head_pose(msg.header.stamp)
        
        self.logger.debug(f"Auto mode - Current target: {curr_coord}, Error: {error}, Head pose: {head_pose}")
        self.logger.debug(f"Head pose database size: {len(self._head_pose_db)}")

        # Calculate PID control signal
        if self._last_error is not None and self._last_stamp is not None:
            time_diff = self._get_time_diff(self._last_stamp, msg.header.stamp)
            
            if time_diff > 0:
                pid_P = np.array(self.get_parameter("converge_to_ball_P").value)
                pid_D = np.array(self.get_parameter("converge_to_ball_D").value)
                
                error_derivative = (error - self._last_error) / time_diff
                
                control_signal = head_pose - (error * pid_P + error_derivative * pid_D)
                
                control_signal[0] = np.clip(control_signal[0], -self.max_yaw, self.max_yaw)
                control_signal[1] = np.clip(control_signal[1], -self.max_pitch, self.max_pitch)
                
                self._set_head_pose(control_signal)
                self._control_timestamps.append(current_time)  # 记录控制时间戳
                self.logger.debug(f"Auto control signal: {control_signal}")
                self.logger.debug(f"PID parameters: P={pid_P}, D={pid_D}")
                self.logger.debug(f"Error derivative: {error_derivative}")
            else:
                self.logger.warning("Zero time difference between messages")
        else:
            self.logger.info("First detection received, initializing tracking")

        self._last_error = error
        self._last_stamp = msg.header.stamp

    def _head_pose_cb(self, msg: HeadPose):
        # 输出调试信息
        self.logger.debug(f"Head pose received: yaw={msg.yaw}, pitch={msg.pitch}")
        
        # Store head pose and timestamp
        self._head_pose_db.append([msg.header.stamp, np.array([msg.yaw, msg.pitch])])
        
        # Limit history length
        max_history_length = 100
        if len(self._head_pose_db) > max_history_length:
            self._head_pose_db.pop(0)
            
        # 输出调试信息
        self.logger.debug(f"Head pose database updated, size: {len(self._head_pose_db)}")

    def _get_head_pose(self, target_stamp):
        # 输出调试信息
        self.logger.debug(f"Getting head pose for timestamp: {target_stamp}")
        self.logger.debug(f"Head pose database size: {len(self._head_pose_db)}")
        
        # Return default pose if no history available
        if not self._head_pose_db:
            self.logger.warning("No head pose data available, returning default")
            return np.array([0.0, 0.5])
            
        # 将ROS时间消息转换为Time对象
        target_time = Time.from_msg(target_stamp)
        
        # Remove outdated data points
        while len(self._head_pose_db) > 1 and Time.from_msg(self._head_pose_db[1][0]) <= target_time:
            self.logger.debug(f"Removing outdated head pose data: {self._head_pose_db[0][0]}")
            self._head_pose_db.pop(0)
            
        # Linear interpolation to get pose at target time
        if len(self._head_pose_db) > 1:
            t1, pose1 = self._head_pose_db[0]
            t2, pose2 = self._head_pose_db[1]
            
            # 将时间戳转换为Time对象
            t1_time = Time.from_msg(t1)
            t2_time = Time.from_msg(t2)
            
            self.logger.debug(f"Interpolating between: t1={t1}, pose1={pose1}, t2={t2}, pose2={pose2}")
            
            time_diff_total = self._get_time_diff(t1, t2)
            if time_diff_total > 0:
                time_diff = self._get_time_diff(t1, target_stamp)
                ratio = min(max(time_diff / time_diff_total, 0.0), 1.0)
                
                self.logger.debug(f"Interpolation ratio: {ratio}")
                
                return pose1 + (pose2 - pose1) * ratio
                
        # Return latest pose if interpolation not possible
        self.logger.debug(f"Using latest head pose: {self._head_pose_db[0][1]}")
        return self._head_pose_db[0][1]

    def _get_time_diff(self, stamp1, stamp2):
        """Calculate time difference between two timestamps (seconds)"""
        time1 = Time.from_msg(stamp1)
        time2 = Time.from_msg(stamp2)
        diff = (time2 - time1).nanoseconds * 1e-9
        self.logger.debug(f"Time difference: {diff} seconds")
        return diff

    def _set_head_pose(self, target):
        """Publish head pose control command"""
        msg = HeadPose()
        msg.header.frame_id = "head_control"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.yaw = float(target[0])
        msg.pitch = float(target[1])
        self._head_pose_pub.publish(msg)
        
        # 输出调试信息
        self.logger.debug(f"Publishing head pose: yaw={msg.yaw}, pitch={msg.pitch}")

    def _manual_control_cb(self, msg: HeadPose):
        """Handle manual control commands using HeadPose messages"""
        # 输出调试信息
        self.logger.debug(f"Manual control command received: yaw={msg.yaw}, pitch={msg.pitch}")
        
        # Check for auto mode command: yaw=+inf, pitch=any value
        if msg.yaw == self.auto_mode_flag:
            self._control_mode = "auto"
            self.logger.info("Switched to AUTO control mode")
            return
            
        # Any valid yaw and pitch values (excluding auto mode flag) switch to manual mode
        yaw = np.clip(msg.yaw, -self.max_yaw, self.max_yaw)
        pitch = np.clip(msg.pitch, -self.max_pitch, self.max_pitch)
        
        self._manual_target = np.array([yaw, pitch])
        self._control_mode = "manual"
        self._set_head_pose(self._manual_target)
        self.logger.info(f"Switched to MANUAL control mode - Set head pose: Yaw={yaw}, Pitch={pitch}")

    def _print_frequencies(self):
        """定期输出控制频率和检测频率"""
        current_time = self.get_clock().now()
        
        # 计算控制频率
        if len(self._control_timestamps) >= 2:
            time_span = (current_time - self._control_timestamps[0]).nanoseconds * 1e-9
            control_freq = (len(self._control_timestamps) - 1) / max(time_span, 0.001)
            self.logger.info(f"Control frequency: {control_freq:.2f} Hz")
        else:
            self.logger.info("Not enough control data to calculate frequency")
            
        # 计算检测频率
        if len(self._detection_timestamps) >= 2:
            time_span = (current_time - self._detection_timestamps[0]).nanoseconds * 1e-9
            detection_freq = (len(self._detection_timestamps) - 1) / max(time_span, 0.001)
            self.logger.info(f"Detection frequency: {detection_freq:.2f} Hz")
        else:
            self.logger.info("Not enough detection data to calculate frequency")

if __name__ == '__main__':
    try:
        rclpy.init(args=sys.argv)
        node = HeadControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node:
            node.destroy_node()
        rclpy.shutdown()