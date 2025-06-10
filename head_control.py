#!/usr/bin/env python3

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from try_import import *

class HeadControl(Node):
    def __init__(self):
        super().__init__("head_control", namespace="THMOS")
        self.logger = self.get_logger()
        self.logger.info("Head control node initialized")

        # Initialize parameters
        self.declare_parameter("image_width", 1280.0)
        self.declare_parameter("image_height", 720.0)
        self.declare_parameter("converge_to_ball_P", [0.4, 0.4])
        self.declare_parameter("converge_to_ball_D", [0.2, 0.2])
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

        self.logger.info("Subscribers and publishers initialized")

        # Set initial head pose
        self._set_head_pose(self._manual_target)

    def _vision_cb(self, msg: VisionDetections):
        # Process vision data only in auto mode
        if self._control_mode != "auto":
            return
            
        # Validate message
        if msg.xmin >= msg.xmax or msg.ymin >= msg.ymax:
            self.logger.warning("Invalid bounding box received")
            return
            
        # Calculate target center coordinates
        curr_coord = (np.array([msg.xmin, msg.ymin]) + 
                      np.array([msg.xmax, msg.ymax])) * 0.5
        
        # Calculate normalized error
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        error = np.array([
            (curr_coord[0] - center_x) / self.image_width,
            (curr_coord[1] - center_y) / self.image_height
        ])
                
        # Get current head pose
        head_pose = self._get_head_pose(msg.header.stamp)
        
        self.logger.debug(f"Auto mode - Current target: {curr_coord}, Error: {error}, Head pose: {head_pose}")

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
                self.logger.debug(f"Auto control signal: {control_signal}")
            else:
                self.logger.warning("Zero time difference between messages")
        else:
            self.logger.info("First detection received, initializing tracking")

        self._last_error = error
        self._last_stamp = msg.header.stamp

    def _head_pose_cb(self, msg: HeadPose):
        # Store head pose and timestamp
        self._head_pose_db.append([msg.header.stamp, np.array([msg.yaw, msg.pitch])])
        
        # Limit history length
        max_history_length = 100
        if len(self._head_pose_db) > max_history_length:
            self._head_pose_db.pop(0)

    def _get_head_pose(self, target_stamp):
        # Return default pose if no history available
        if not self._head_pose_db:
            self.logger.warning("No head pose data available, returning default")
            return np.array([0.0, 0.5])
            
        # Remove outdated data points
        while len(self._head_pose_db) > 1 and self._head_pose_db[1][0] <= target_stamp:
            self._head_pose_db.pop(0)
            
        # Linear interpolation to get pose at target time
        if len(self._head_pose_db) > 1:
            t1, pose1 = self._head_pose_db[0]
            t2, pose2 = self._head_pose_db[1]
            
            time_diff_total = self._get_time_diff(t1, t2)
            if time_diff_total > 0:
                time_diff = self._get_time_diff(t1, target_stamp)
                ratio = min(max(time_diff / time_diff_total, 0.0), 1.0)
                
                return pose1 + (pose2 - pose1) * ratio
                
        # Return latest pose if interpolation not possible
        return self._head_pose_db[0][1]

    def _get_time_diff(self, stamp1, stamp2):
        """Calculate time difference between two timestamps (seconds)"""
        time1 = Time.from_msg(stamp1)
        time2 = Time.from_msg(stamp2)
        return (time2 - time1).nanoseconds * 1e-9

    def _set_head_pose(self, target):
        """Publish head pose control command"""
        msg = HeadPose()
        msg.header.frame_id = "head_control"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.yaw = float(target[0])
        msg.pitch = float(target[1])
        self._head_pose_pub.publish(msg)

    def _manual_control_cb(self, msg: HeadPose):
        """Handle manual control commands using HeadPose messages"""
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
