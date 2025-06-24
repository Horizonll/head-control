#!/usr/bin/env python3

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from try_import import *
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
from datetime import datetime

class HeadControl(Node):
    def __init__(self):
        super().__init__("head_control", namespace="THMOS")
        self.logger = self.get_logger()
        self.logger.info("Head control node initialized")
 
        # TODO: upgrade to overridable yaml config 
        self.config = {"image_size": [1280, 736], \
                        "ball_target_position": [0.5, 0.5], \
                        "frequence": 1, \
                        "max_yaw": [-1, 1],  # clamp to [min, max]
                        "max_pitch": [-0.785, 0.785],  # clamp [min, max]
                        "converge_to_ball_P": [0.6, 0.6], \
                        "converge_to_ball_D": [0.05, 0.05],
                        "input_filter_alpha": 0.5}  # exp-filter args, higher is smoother
        self.ball_target_coord = np.array(self.config.get("image_size")) \
                            * np.array(self.config.get("ball_target_position"))
         
        # Initialize data storage with deque for efficient operations
        self._head_pose_db = deque(maxlen=100)  # Use deque instead of list
        self._last_stamp = self.get_clock().now().to_msg()
        self._prev_stamp = self.get_clock().now().to_msg()
        self._last_error = np.array([0, 0])
        self._prev_error = np.array([0, 0])  # Store previous error
        self._manual_control = np.array([np.nan, np.nan])   
        self._filtered_ball_coord = None
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers and publishers
        self._vision_sub = self.create_subscription(
            VisionDetections,
            "/THMOS/vision/obj_pos",
            self._vision_cb,
            qos)
        
        self._head_pose_pub = self.create_publisher(
            JointState,
            "/THMOS/hardware/set_head_pose",
            1)
            
        self._head_pose_sub = self.create_subscription(
            JointState,
            "/THMOS/hardware/get_head_pose",
            self._head_pose_cb,
            qos)
            
        self._manual_control_sub = self.create_subscription(
            JointState,
            "/THMOS/head_control/manual_command",
            self._manual_control_cb,
            1)
  
        self.logger.info("Subscribers and publishers initialized")
        self._set_head_pose([0, 0])
        

    def _vision_cb(self, msg: VisionDetections):
        current_time = self.get_clock().now()
        
        # Output debug information
        # self.logger.debug(f"Vision detection received at: {current_time}")
        # self.logger.debug(f"Number of detected objects: {len(msg.detected_objects)}")
        
        # Filter objects labeled "Ball" and find the one with highest confidence
        ball_objects = [obj for obj in msg.detected_objects if obj.label == "ball"]
        if not ball_objects:
            self.logger.info("No ball objects detected")
            return
        
        # Fetch head pose from database
        head_pose = self._get_head_pose(msg.header.stamp) 
        if head_pose is None:
            self.logger.warn("Outdated head pose!")
            return  # just do not things before receiving head pose
            
        # Find the best ball with highest confidence
        best_ball = max(ball_objects, key=lambda obj: obj.confidence)
        
        # Compute the coordination of the ball in vision
        ball_coord = 0.5 * (np.array([best_ball.xmin, best_ball.ymin]) \
                    + np.array([best_ball.xmax, best_ball.ymax]))
        
        # exp-filter
        alpha = self.config.get("input_filter_alpha")
        if self._filtered_ball_coord is None:
            self._filtered_ball_coord = ball_coord
        else:
            self._filtered_ball_coord = alpha * self._filtered_ball_coord + (1 - alpha) * ball_coord
            
        # self.logger.info(f"Raw ball coord: {ball_coord}, Filtered: {self._filtered_ball_coord}")
        
        # compute normalized error
        error = (self._filtered_ball_coord - self.ball_target_coord) / np.array(self.config.get("image_size"))
        error *= np.array([1, -1]); # HAVE TO alter the direction
        # Save current detection information
        self._prev_stamp, self._last_stamp = self._last_stamp, msg.header.stamp
        self._prev_error, self._last_error = self._last_error, error
        
        # Calculate the derivative of error
        time_diff = max(self._get_time_diff(self._prev_stamp, self._last_stamp), 1e-9)
        error_d = (self._last_error - self._prev_error) / time_diff; 
        
        # PID output as delta (control, not control change)
        pid_P = self.config.get("converge_to_ball_P")
        pid_D = self.config.get("converge_to_ball_D")
        control_delta = - (self._last_error * pid_P + error_d * pid_D)

        # Add up to the headpose 
        self._set_head_pose(head_pose + control_delta)
        
        self.logger.info(f"Error: {self._last_error}")
        self.logger.info(f"Error derivative: {error_d}, dt = {time_diff} last={self._last_error} prev={self._prev_error}")
        self.logger.info(f"Control delta: {control_delta}")


    def _head_pose_cb(self, msg: JointState):
        yaw = msg.position[0] if len(msg.position) > 0 else 0.0
        pitch = msg.position[1] if len(msg.position) > 1 else 0.0
        self._head_pose_db.append([msg.header.stamp, np.array([yaw, pitch])])


    def _get_head_pose(self, target_stamp):
        # DO NOT RETURN DEFAULT POSITION. WILL CAUSE INCORRECT CONTROL SIGNAL.
        # Just return None, we'll (and we have to) handle it
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
                self.logger.debug(f"Removing outdated head pose data: {self._head_pose_db[0][0]}")
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
            
            # self.logger.debug(f"Interpolating between: t1={t1}, pose1={pose1}, t2={t2}, pose2={pose2}")
            
            time_diff_total = self._get_time_diff(t1, t2)
            time_diff_total = max(time_diff_total, 1e-9)  # Avoid division by zero error
            
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
        diff = (time2 - time1).nanoseconds * 1e-9 # + (time2 - time1).seconds
        return diff


    def _set_head_pose(self, target):
        """Publish head pose control command"""
        try:
            msg = JointState()
            msg.header.frame_id = "head_control"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["head_yaw_joint", "head_pitch_joint"]
            msg.position = [0.0, 0.0]  # Initialize with zeros
            max_yaw = self.config.get("max_yaw")
            max_pitch = self.config.get("max_pitch")
            msg.position[0] = float(np.clip(target[0], max_yaw[0], max_yaw[1]))  # 使用范围的最小值和最大值
            msg.position[1] = float(np.clip(target[1], max_pitch[0], max_pitch[1]))  # 使用范围的最小值和最大值
            if not np.isnan(self._manual_control[0]):
                msg.position[0] = self._manual_control[0]
            if not np.isnan(self._manual_control[1]):
                msg.position[1] = self._manual_control[1]
            self._head_pose_pub.publish(msg)
            
            # Output debug information
            # self.logger.debug(f"Publishing head pose: yaw={msg.position[0]}, pitch={msg.position[1]}")
        except Exception as e:
            self.logger.error(f"Error publishing head pose command: {str(e)}")

    def _manual_control_cb(self, msg: JointState):
        """Handle manual control commands using HeadPose messages"""
        try:
            yaw = msg.position[0] if len(msg.position) > 0 else 0.0
            pitch = msg.position[1] if len(msg.position) > 1 else 0.0
            self._manual_control = np.array([yaw, pitch])
            self._set_head_pose(self._manual_control)
            self.logger.info(f"Received manual control signal - " +  \
                "Set head pose: Yaw={yaw}, Pitch={pitch}")
        except Exception as e:
            self.logger.error(f"Error handling manual control command: {str(e)}")

    def destroy_node(self):
        super().destroy_node()
        self.logger.info("Head control node destroyed")

if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    node = HeadControl()
    rclpy.spin(node)
    try:
        pass
    except KeyboardInterrupt:
        node.logger.info("Node interrupted by user")
    except Exception as e:
        if 'node' in locals() and node:
            node.logger.error(f"Error in node execution: {str(e)}")
    finally:
        if 'node' in locals() and node:
            node.destroy_node()
        rclpy.shutdown()
    
