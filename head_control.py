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

        # Frequency calculation related variables
        self._control_timestamps = deque(maxlen=100)  # Store last 100 control timestamps
        self._detection_timestamps = deque(maxlen=100)  # Store last 100 detection timestamps
        
        # Initialize parameters
        self.declare_parameter("image_width", 1280.0)
        self.declare_parameter("image_height", 736.0)
        self.declare_parameter("converge_to_ball_P", [1.0, 1.0])
        self.declare_parameter("converge_to_ball_D", [0.0, 0.0])
        self.declare_parameter("max_yaw", 1.5)
        self.declare_parameter("max_pitch", 1.0)
        self.declare_parameter("auto_mode_flag", float('inf'))
        self.declare_parameter("control_frequency", 20.0)  # Control frequency parameter
        self.declare_parameter("enable_plotting", False)
        
        # Get parameters and set as class attributes for quick access
        self.image_width = self.get_parameter("image_width").value
        self.image_height = self.get_parameter("image_height").value
        self.max_yaw = self.get_parameter("max_yaw").value
        self.max_pitch = self.get_parameter("max_pitch").value
        self.auto_mode_flag = self.get_parameter("auto_mode_flag").value
        self.control_frequency = self.get_parameter("control_frequency").value
        self.enable_plotting = self.get_parameter("enable_plotting").value
        
        # Data recording related
        self._data_records = []  # Store control data
        self._recording_enabled = True  # Whether recording is enabled
        self._record_file_path = None  # Record file path
        self._plot_file_path = None  # Plot file path
        
        # Initialize data storage with deque for efficient operations
        self._head_pose_db = deque(maxlen=100)  # Use deque instead of list
        self._last_coord = None
        self._last_stamp = None
        self._last_error = None
        self._prev_error = None  # Store previous error
        self._manual_target = np.array([0.0, 0.5])  # Default manual target
        self._control_mode = "auto"  # Default to auto mode
        self._last_control_signal = np.array([0.0, 0.0])  # Last control signal
        
        # Create QoS profile
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers and publishers
        self._vision_sub = self.create_subscription(
            VisionDetections,
            "vision/detections",
            self._vision_cb,
            qos)
        
        self._head_pose_pub = self.create_publisher(
            JointState,
            "hardware/set_head_pose",
            1)
            
        self._head_pose_sub = self.create_subscription(
            JointState,
            "hardware/get_head_pose",
            self._head_pose_cb,
            qos)
            
        self._manual_control_sub = self.create_subscription(
            JointState,
            "head_control/manual_command",
            self._manual_control_cb,
            1)

        # Create timer for fixed-frequency control
        self._control_timer = self.create_timer(1.0/self.control_frequency, self._control_loop)
        
        # Create timer for outputting frequency information
        self._freq_timer = self.create_timer(5.0, self._print_frequencies)
        
        # Create timer for saving data and plotting (根据参数决定是否创建)
        if self.enable_plotting:
            self._save_timer = self.create_timer(60.0, self._save_data_and_plot)  # Save once per minute
        else:
            self._save_timer = None
            self.logger.info("Plotting function is disabled by default")
        
        self.logger.info("Subscribers and publishers initialized")
        
        # Initialize data recording file
        self._init_data_recording()

        # Set initial head pose
        self._set_head_pose(self._manual_target)

    def _init_data_recording(self):
        """Initialize data recording file and path"""
        # Create save directory
        save_dir = os.path.join(os.getcwd(), "head_control_data")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Generate timestamped file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._record_file_path = os.path.join(save_dir, f"head_control_data_{timestamp}.csv")
        self._plot_file_path = os.path.join(save_dir, f"head_control_plot_{timestamp}.png")
        
        # Write CSV file header (including control delta)
        try:
            with open(self._record_file_path, 'w') as f:
                f.write("timestamp,error_x,error_y,control_yaw,control_pitch,current_yaw,current_pitch,control_delta_x,control_delta_y\n")
        except Exception as e:
            self.logger.error(f"Failed to initialize data recording file: {str(e)}")
            self._recording_enabled = False
            
        self.logger.info(f"Data recording initialized - File: {self._record_file_path}")

    def _vision_cb(self, msg: VisionDetections):
        current_time = self.get_clock().now()
        self._detection_timestamps.append(current_time)
        
        # Output debug information
        self.logger.debug(f"Vision detection received at: {current_time}")
        self.logger.debug(f"Number of detected objects: {len(msg.detected_objects)}")
        
        # Filter objects labeled "Ball" and find the one with highest confidence
        ball_objects = [obj for obj in msg.detected_objects if obj.label == "Ball"]
        if not ball_objects:
            self.logger.info("No ball objects detected")
            self._last_coord = None  # No ball detected
            return
            
        # Find the best ball with highest confidence
        best_ball = max(ball_objects, key=lambda obj: obj.confidence)
        self.logger.info(f"Best ball detected - Confidence: {best_ball.confidence}")
        
        # Validate message
        if best_ball.xmin >= best_ball.xmax or best_ball.ymin >= best_ball.ymax:
            self.logger.warning("Invalid bounding box received for best ball")
            self._last_coord = None  # Invalid bounding box
            return
            
        # Calculate target center coordinates
        curr_coord = (np.array([best_ball.xmin, best_ball.ymin]) + 
                      np.array([best_ball.xmax, best_ball.ymax])) * 0.5
        
        # Calculate normalized error
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        error = np.array([
            (curr_coord[0] - center_x) / self.image_width,
            - (curr_coord[1] - center_y) / self.image_height
        ])
        
        # Save current detection information
        self._last_coord = curr_coord
        self._last_error = error
        self._last_stamp = msg.header.stamp

    def _head_pose_cb(self, msg: JointState):
        yaw = msg.position[0] if len(msg.position) > 0 else 0.0
        pitch = msg.position[1] if len(msg.position) > 1 else 0.0
        
        # Output debug information
        self.logger.debug(f"Head pose received: yaw={yaw}, pitch={pitch}")
        
        # Store head pose and timestamp (use deque to manage length automatically)
        self._head_pose_db.append([msg.header.stamp, np.array([yaw, pitch])])
        
        # Output debug information
        self.logger.debug(f"Head pose database updated, size: {len(self._head_pose_db)}")

    def _get_head_pose(self, target_stamp):
        # Output debug information
        self.logger.debug(f"Getting head pose for timestamp: {target_stamp}")
        self.logger.debug(f"Head pose database size: {len(self._head_pose_db)}")
        
        # Return default pose if no history available
        if not self._head_pose_db:
            self.logger.warning("No head pose data available, returning default")
            return np.array([0.0, 0.5])
            
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
            
            self.logger.debug(f"Interpolating between: t1={t1}, pose1={pose1}, t2={t2}, pose2={pose2}")
            
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
        try:
            time1 = Time.from_msg(stamp1)
            time2 = Time.from_msg(stamp2)
            diff = (time2 - time1).nanoseconds * 1e-9
            self.logger.debug(f"Time difference: {diff} seconds")
            return diff
        except Exception as e:
            self.logger.error(f"Error calculating time difference: {str(e)}")
            return 1e-9  # Return a tiny value to avoid division by zero

    def _control_loop(self):
        """Fixed-frequency control loop"""
        current_time = self.get_clock().now()
        
        # Only control in auto mode
        if self._control_mode != "auto" or self._last_error is None or self._last_stamp is None:
            return
            
        # Get current head pose
        head_pose = self._get_head_pose(self._last_stamp)
        
        # Calculate PID control signal
        time_diff = self._get_time_diff(self._last_stamp, self.get_clock().now().to_msg())
        time_diff = max(time_diff, 1e-9)  # Avoid division by zero error
        
        try:
            pid_P = np.array(self.get_parameter("converge_to_ball_P").value)
            pid_D = np.array(self.get_parameter("converge_to_ball_D").value)
            
            if len(self._control_timestamps) < 2:
                error_derivative = np.array([0.0, 0.0])
                self._prev_error = self._last_error.copy()  # Save previous error
            else:
                if self._prev_error is not None:
                    error_derivative = (self._last_error - self._prev_error) / time_diff
                else:
                    error_derivative = np.array([0.0, 0.0])
                self._prev_error = self._last_error.copy()  # Update previous error
                
            # PID output as delta (control change)
            control_delta = - (self._last_error * pid_P + error_derivative * pid_D)
            
            # Final command = current pose + PID delta
            control_signal = head_pose + control_delta
            
            # Limit output range
            control_signal[0] = np.clip(control_signal[0], -self.max_yaw, self.max_yaw)
            control_signal[1] = np.clip(control_signal[1], -self.max_pitch, self.max_pitch)
            
            self._set_head_pose(control_signal)
            self._control_timestamps.append(current_time)  # Record control timestamp
            
            self.logger.debug(f"Auto control signal: {control_signal}")
            self.logger.debug(f"PID parameters: P={pid_P}, D={pid_D}")
            self.logger.debug(f"Error derivative: {error_derivative}")
            self.logger.debug(f"Control delta: {control_delta}")
            
            # Record data (including control delta)
            self._record_data(current_time, self._last_error, control_signal, head_pose, control_delta)
            
            # Save last control signal
            self._last_control_signal = control_signal
        except Exception as e:
            self.logger.error(f"Error in control loop: {str(e)}")

    def _set_head_pose(self, target):
        """Publish head pose control command"""
        try:
            msg = JointState()
            msg.header.frame_id = "head_control"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["head_yaw_joint", "head_pitch_joint"]
            msg.position = [0.0, 0.0]  # Initialize with zeros
            msg.position[0] = float(target[0])  # Set yaw
            msg.position[1] = float(target[1])
            self._head_pose_pub.publish(msg)
            
            # Output debug information
            self.logger.debug(f"Publishing head pose: yaw={msg.position[0]}, pitch={msg.position[1]}")
        except Exception as e:
            self.logger.error(f"Error publishing head pose command: {str(e)}")

    def _manual_control_cb(self, msg: JointState):
        """Handle manual control commands using HeadPose messages"""
        try:
            yaw = msg.position[0] if len(msg.position) > 0 else 0.0
            pitch = msg.position[1] if len(msg.position) > 1 else 0.0

            # Output debug information
            self.logger.debug(f"Manual control command received: yaw={yaw}, pitch={pitch}")
            
            # Check for auto mode command: yaw=+inf, pitch=any value
            if yaw == self.auto_mode_flag:
                self._control_mode = "auto"
                self.logger.info("Switched to AUTO control mode")
                return
                
            # Any valid yaw and pitch values (excluding auto mode flag) switch to manual mode
            yaw = np.clip(yaw, -self.max_yaw, self.max_yaw)
            pitch = np.clip(pitch, -self.max_pitch, self.max_pitch)
            
            self._manual_target = np.array([yaw, pitch])
            self._control_mode = "manual"
            self._set_head_pose(self._manual_target)
            self.logger.info(f"Switched to MANUAL control mode - Set head pose: Yaw={yaw}, Pitch={pitch}")
        except Exception as e:
            self.logger.error(f"Error handling manual control command: {str(e)}")

    def _print_frequencies(self):
        """Periodically output control frequency and detection frequency"""
        current_time = self.get_clock().now()
        
        try:
            # Calculate control frequency
            if len(self._control_timestamps) >= 2:
                time_span = (current_time - self._control_timestamps[0]).nanoseconds * 1e-9
                control_freq = (len(self._control_timestamps) - 1) / max(time_span, 0.001)
                self.logger.info(f"Control frequency: {control_freq:.2f} Hz (Target: {self.control_frequency} Hz)")
            else:
                self.logger.info("Not enough control data to calculate frequency")
                
            # Calculate detection frequency
            if len(self._detection_timestamps) >= 2:
                time_span = (current_time - self._detection_timestamps[0]).nanoseconds * 1e-9
                detection_freq = (len(self._detection_timestamps) - 1) / max(time_span, 0.001)
                self.logger.info(f"Detection frequency: {detection_freq:.2f} Hz")
            else:
                self.logger.info("Not enough detection data to calculate frequency")
        except Exception as e:
            self.logger.error(f"Error calculating frequencies: {str(e)}")

    def _record_data(self, timestamp, error, control_signal, current_pose, control_delta):
        """Record control data including PID output (control delta)"""
        if not self._recording_enabled:
            return
            
        try:
            # Convert ROS time to seconds
            time_sec = timestamp.nanoseconds * 1e-9
            
            # Save data point
            data_point = {
                'timestamp': time_sec,
                'error': error,
                'control_signal': control_signal,
                'current_pose': current_pose,
                'control_delta': control_delta
            }
            
            self._data_records.append(data_point)
            
            # Write to CSV file including control delta
            with open(self._record_file_path, 'a') as f:
                f.write(f"{time_sec},{error[0]},{error[1]},{control_signal[0]},{control_signal[1]},{current_pose[0]},{current_pose[1]},{control_delta[0]},{control_delta[1]}\n")
        except Exception as e:
            self.logger.error(f"Error recording data: {str(e)}")

    def _save_data_and_plot(self):
        """Save data and generate plots including PID output (control delta)"""
        if not self._data_records or not self._recording_enabled:
            return
            
        try:
            # Extract data for plotting
            timestamps = [data['timestamp'] for data in self._data_records]
            error_x = [data['error'][0] for data in self._data_records]
            error_y = [data['error'][1] for data in self._data_records]
            control_yaw = [data['control_signal'][0] for data in self._data_records]
            control_pitch = [data['control_signal'][1] for data in self._data_records]
            current_yaw = [data['current_pose'][0] for data in self._data_records]
            current_pitch = [data['current_pose'][1] for data in self._data_records]
            control_delta_x = [data['control_delta'][0] for data in self._data_records]
            control_delta_y = [data['control_delta'][1] for data in self._data_records]
            
            # Create figure for yaw data (with control delta)
            fig_yaw, ax_yaw = plt.subplots(figsize=(12, 6))
            
            # Plot yaw error, control signal, current pose and control delta
            ax_yaw.plot(timestamps, error_x, 'r-', label='Error X (Yaw)')
            ax_yaw.plot(timestamps, control_yaw, 'g-', label='Control Yaw')
            ax_yaw.plot(timestamps, current_yaw, 'b-', label='Current Yaw')
            ax_yaw.plot(timestamps, control_delta_x, 'm-', label='Control Delta X (PID Output)')  # Add control delta
            
            ax_yaw.set_title('Yaw Control Data')
            ax_yaw.set_xlabel('Time (s)')
            ax_yaw.set_ylabel('Value')
            ax_yaw.legend()
            ax_yaw.grid(True)
            
            # Create figure for pitch data (with control delta)
            fig_pitch, ax_pitch = plt.subplots(figsize=(12, 6))
            
            # Plot pitch error, control signal, current pose and control delta
            ax_pitch.plot(timestamps, error_y, 'r-', label='Error Y (Pitch)')
            ax_pitch.plot(timestamps, control_pitch, 'g-', label='Control Pitch')
            ax_pitch.plot(timestamps, current_pitch, 'b-', label='Current Pitch')
            ax_pitch.plot(timestamps, control_delta_y, 'm-', label='Control Delta Y (PID Output)')  # Add control delta
            
            ax_pitch.set_title('Pitch Control Data')
            ax_pitch.set_xlabel('Time (s)')
            ax_pitch.set_ylabel('Value')
            ax_pitch.legend()
            ax_pitch.grid(True)
            
            # Automatically adjust layout
            plt.tight_layout()
            
            # Save figures
            plot_dir = os.path.dirname(self._plot_file_path)
            base_name = os.path.basename(self._plot_file_path)
            name_parts = os.path.splitext(base_name)
            
            yaw_plot_path = os.path.join(plot_dir, f"{name_parts[0]}_yaw{name_parts[1]}")
            pitch_plot_path = os.path.join(plot_dir, f"{name_parts[0]}_pitch{name_parts[1]}")
            
            fig_yaw.savefig(yaw_plot_path)
            fig_pitch.savefig(pitch_plot_path)
            
            plt.close(fig_yaw)
            plt.close(fig_pitch)
            
            self.logger.info(f"Yaw data plot saved to: {yaw_plot_path}")
            self.logger.info(f"Pitch data plot saved to: {pitch_plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving plot: {str(e)}")

    def destroy_node(self):
        # Save final data and plots
        self._save_data_and_plot()
        
        super().destroy_node()
        self.logger.info("Head control node destroyed")

if __name__ == '__main__':
    try:
        rclpy.init(args=sys.argv)
        node = HeadControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.logger.info("Node interrupted by user")
    except Exception as e:
        if 'node' in locals() and node:
            node.logger.error(f"Error in node execution: {str(e)}")
    finally:
        if 'node' in locals() and node:
            node.destroy_node()
        rclpy.shutdown()