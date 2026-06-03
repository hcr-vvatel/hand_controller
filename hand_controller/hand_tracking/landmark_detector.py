import cv2
import time
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_controller.hand_tracking.hands import Hand

class LiveLandmarkDetector:
    def __init__(self, number_of_hands = 1):
        self.model_file_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

        self.options = vision.HandLandmarkerOptions(
            base_options = python.BaseOptions(self.model_file_path),
            running_mode = vision.RunningMode.LIVE_STREAM,
            num_hands = number_of_hands,
            result_callback = self._result_cb
        )

        self.landmarker = vision.HandLandmarker.create_from_options(self.options)
        self.latest_frame = None
        self.latest_hands = []
        self.latest_timestamp = 0


    @staticmethod
    def draw_landmarks_from_hands(drawn_image, hands) -> list:     
        """
        Description:
            Uses cv2 to draw circles where the landmarks exist on hands, and to write the angle of each finger
        
        Params:
            drawn_image (matlike): Image to draw on
            hands (list[Hand]): Hands data
        """
        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

        for hand in hands:
            raised_fingers = set(hand.get_raised_fingers())
            for name, points in hand.fingers.items():
                for pt in points:
                    x, y, _ = pt
                    u, v = int(x * drawn_image.shape[1]), int(y * drawn_image.shape[0])

                    cv2.circle(drawn_image, center=(u,v), radius = 5, color=(0,255,0))
                
                if name in raised_fingers:
                    finger_angle = hand.calculate_finger_angle(name)
                    cv2.putText(drawn_image, f"{finger_angle}", (u,v), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255,0,0), thickness=2)

        return drawn_image

    def _result_cb(self, result, output_img: mp.Image, timestamp_ms: int) -> None: 
        self.latest_hands = Hand.from_landmarker_result(result)

    def get_latest_data(self):
        return (self.latest_frame, self.latest_hands)
    
    def process_frame(self, frame):
        """
        Description:
            Processes the frame using landmark detection model
        
        Params:
            frame (matlike)
        """
        self.latest_frame = frame

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = frame,
        )

        self.latest_timestamp = max(int(time.time() * 1000), self.latest_timestamp)
        self.landmarker.detect_async(mp_image, self.latest_timestamp)