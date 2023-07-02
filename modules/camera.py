from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

class CameraManager:

    def __init__(
        self,
        camera_width: int = 640,
        camera_height: int = 480,
        draw_hands: bool = True,
        track_num_hands: int = 1,
        min_track_confidence: float = 0.8,
        min_detection_confidence: float = 0.5
    ) -> None:
        
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.draw_hands = draw_hands
        self.track_num_hands = track_num_hands
        self.min_track_confidence = min_track_confidence
        self.min_detection_confidence = min_detection_confidence

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=self.track_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_track_confidence
        )
    
    def process_hands(self) -> ProcessHandsOutput:
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(framergb)

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        
        if not result.multi_hand_landmarks:
            return ProcessHandsOutput(None, frame)

        hand_landmark_collection = []
        for hand_landmarks in result.multi_hand_landmarks:

            if self.draw_hands:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
            hand_landmark_collection.append(hand_landmarks.landmark)
            
        return ProcessHandsOutput(hand_landmark_collection, frame)
    
    def __del__(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

class ProcessHandsOutput:

    def __init__(
        self,
        hand_landmarks: list,
        frame
    ) -> None:
        self.hand_landmarks = hand_landmarks
        self.frame = frame
        