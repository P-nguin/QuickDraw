import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np 
import time

import Exercises
from GameStatesEnum import GameStates

# mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.60, min_tracking_confidence=0.80)
hands = mp_hands.Hands(min_detection_confidence=0.50, min_tracking_confidence=0.70)

# mediapipe model setup
gesture_path = "models/gesture_recognizer.task"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

hand_detections = [0, 0]
# Create a gesture recognizer instance with the live stream mode:
def get_result_left(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    detection_result = result.gestures
    if not len(detection_result) == 0:
        hand_detections[0] = detection_result[-1][-1].category_name

def get_result_right(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    detection_result = result.gestures
    if not len(detection_result) == 0:
        hand_detections[1] = detection_result[-1][-1].category_name

gesture_options_left = mp.tasks.vision.GestureRecognizerOptions(
    running_mode=VisionRunningMode.LIVE_STREAM,
    base_options=mp.tasks.BaseOptions(model_asset_path=gesture_path),
    result_callback=get_result_left)
recognizer_left = mp.tasks.vision.GestureRecognizer.create_from_options(gesture_options_left)

gesture_options_right = mp.tasks.vision.GestureRecognizerOptions(
    running_mode=VisionRunningMode.LIVE_STREAM,
    base_options=mp.tasks.BaseOptions(model_asset_path=gesture_path),
    result_callback=get_result_right)
recognizer_right = mp.tasks.vision.GestureRecognizer.create_from_options(gesture_options_right)

game_state = GameStates.MAIN_MENU
READY_TIME = 3
ready_cnt = 0

# setup cmaera
cam = cv2.VideoCapture(0)
start = time.time()
elapsed_time = 0
while cam.isOpened():
    ret, frame = cam.read()
    delta_time = time.time() - start
    elapsed_time += delta_time
    start = time.time()

    # get keyboard input
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # process image, convert from BRG to RGB and flip horizontally
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False

    if game_state == GameStates.MAIN_MENU:
        # proccess image through hand detection
        # blackout half of the image, this way the dectection positions are accurate
        # if both hands are giving thumbs up, then move to next screen.

        image_left = np.copy(image)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
        image_left[ :, image.shape[1]//2:, ] = [0, 0, 0]
        image_left.flags.writeable = False

        image_right = np.copy(image)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
        image_right[ :, :image.shape[1]//2, ] = [0, 0, 0]
        image_right.flags.writeable = False
        
        #hands_results_left = hands.process(image_left)
        #hands_results_right = hands.process(image_right)

        image.flags.writeable = True
        mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_left)
        mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_right)
        recognizer_left.recognize_async(mp_image_left, mp.Timestamp.from_seconds(time.time()).value)
        recognizer_right.recognize_async(mp_image_right, mp.Timestamp.from_seconds(time.time()).value)

    elif game_state == GameStates.PLAYING:
        pass

    # process image, convert from RGB to BRG
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    cv2.imshow('QuickDraw', image)

cam.release()
cv2.destroyAllWindows()  