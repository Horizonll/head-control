# try to import packages


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from std_msgs.msg import Header, Float32
    from geometry_msgs.msg import Pose, Vector3
    from sensor_msgs.msg import CameraInfo, JointState
    from rclpy.executors import SingleThreadedExecutor
except Exception as e:
    print("Fatal: Can not import ros2 standard library: " )
    print(e)
    print("Have you source the ros2 setup.bash ?")
    exit(-1)


try:
    from thmos_msgs.msg import Velocity
    from thmos_msgs.msg import VisionObj, VisionDetections
except Exception as e:
    print("Fatal: Can not import thmos messages: " )
    print(e)
    print("Have you source the thmos_msgs packages ?")
    exit(-1)

