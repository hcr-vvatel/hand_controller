import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image

from hand_tracking.landmark_detector import LiveLandmarkDetector

import numpy as np
import cv2
from cv_bridge import CvBridge

class HandController(Node):
    def __init__(self):
        super().__init__("hand_controller")

        ### PARAMETERS START ###

        ### Topic Parameters
        self.declare_parameter("drive_pub_topic", "/drive")
        self.declare_parameter("webcam_sub_topic", "/image_raw")
        self.declare_parameter("image_pub_topic", "/image_drawn")

        self.DRIVE_TOPIC = self.get_parameter("drive_pub_topic").get_parameter_value().string_value
        self.WEBCAM_TOPIC = self.get_parameter("webcam_sub_topic").get_parameter_value().string_value
        self.IMAGE_TOPIC = self.get_parameter("image_pub_topic").get_parameter_value().string_value

        ### Constant Parameters
        self.declare_parameter("velocity_increments", 0.1) # controls how fast or slow the car changes speed
        self.declare_parameter("max_velocity", 3.0) # used to clamp the speed of the car
        self.declare_parameter("max_steering_angle", np.pi/4) # used to clamp the steering angle
        self.declare_parameter("timer_rate", 10) # rate (hz) at which drive commands are published

        self.VELOCITY_INCREMENTS = self.get_parameter("velocity_increments").get_parameter_value().double_value
        self.MAX_VELOCITY = self.get_parameter("max_velocity").get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter("max_steering_angle").get_parameter_value().double_value
        self.TIMER_RATE = self.get_parameter("timer_rate").get_parameter_value().integer_value

        ### PARAMETERS END ###

        ### Publishers and Subscribers 
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.image_publisher = self.create_publisher(Image, self.IMAGE_TOPIC, 10)

        self.webcam_subscriber = self.create_subscription(Image, self.WEBCAM_TOPIC, self.webcam_callback, 10)

        ### Instance Variables
        self.velocity = 1.0
        self.bridge = CvBridge()
        self.controller_timer = self.create_timer(1.0 / self.TIMER_RATE_HZ, self.timer_callback)
        self.hand_tracker = LiveLandmarkDetector(number_of_hands=1)

        self.get_logger().info("Hand Controller Ready...")
    
    def timer_callback(self):
        """
        Description: 
            Timer callback to process latest hands data into drive commands and then publish them
        """
        image, hands = self.hand_tracker.get_latest_data()
        if image is None: # if no data is availble, return
            return
        
        # Overlay hand landmarks on latest webcam frame
        drawn_image = self.hand_tracker.draw_landmarks_from_hands(image, hands)

        # Flip image horizontally so left and right match that of real world
        drawn_image = cv2.flip(drawn_image, 1)

        # Convert image from bgr to rgb
        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB) 

        # Publish image
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(drawn_image, encoding="rgb8"))

        if len(hands) == 0: # if no hands detected, return
            return
        
        # only read from one "control" hand, redundtant if hand_tracker is initialized with number_of_hands = 1
        control_hand = hands[0] 
        raised_fingers = control_hand.get_raised_fingers()
        num_fingers_raised = len(raised_fingers)

        drive_cmd = None

        if num_fingers_raised == 5:
            ### Stop
            self.get_logger().info("Command: Stop")
            drive_cmd = self.create_stop_cmd()
        elif num_fingers_raised == 2:
            ### Accelerate
            self.velocity += self.VELOCITY_INCREMENTS
            self.velocity = min(self.VELOCITY_MAX, self.velocity)
            self.get_logger().info(f"Command: Accelerate! Speed = {self.velocity}")
        elif num_fingers_raised == 3:
            ### Decelerate
            self.velocity -= self.VELOCITY_INCREMENTS
            self.velocity = max(0.0, self.velocity)
            self.get_logger().info(f"Command: Decelerate! Speed = {self.velocity}")
        elif num_fingers_raised == 1 and raised_fingers[0] == "index":
            ### Turn
            index_angle = control_hand.calculate_finger_angle("index")
            index_angle = min(max(index_angle, -self.MAX_ANGLE), self.MAX_ANGLE)  # Clamp steering angle
            self.get_logger().info(f"Command: Turn: {int(index_angle * (180/np.pi))} degrees")
            drive_cmd = self.create_angle_cmd(index_angle)
        elif num_fingers_raised == 0:
            ### Go Straight
            self.get_logger().info("Command: Go")
            drive_cmd = self.create_angle_cmd(0.0)
        else:
            ### Command Not Understood
            self.get_logger().info("Command: Unknown")
        
        if drive_cmd is None:
            return
        
        self.drive_publisher.publish(drive_cmd)
    
    def webcam_callback(self, webcam_msg):
        """
        Description:
            Callback that runs when an image is read from the webcam. Passes the image to the hand tracker
            which processes and saves its latest results

        Param:
            webcam_msg (sensor_msgs.msg.Image)
        """
        read_frame = self.bridge.imgmsg_to_cv2(webcam_msg, desired_encoding='rgb8')
        self.hand_tracker.process_frame(read_frame)
    
    def create_angle_cmd(self, steering_angle):
        """
        Description:
            Creates an AckerMannDrive message to change the steering angle of the racecar.

        Param:
            steering_angle (float) the steering angle for the drive message.
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = self.velocity

        return msg

    def create_stop_cmd(self):
        """
        Description:
            Creates an AckerMannDrive message to stop the racecar
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.drive.speed = 0.0

        return msg

def main():
    rclpy.init()
    hand_controller = HandController()
    rclpy.spin(hand_controller)
    hand_controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
