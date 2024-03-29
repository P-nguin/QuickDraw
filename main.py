import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np 
import time

import Exercises
from GameStatesEnum import GameStates

#setup game states
exercise_manager = Exercises.ExerciseManager()
game_state = GameStates.MAIN_MENU
READY_TIME = 3.25
ready_cnt = 0

# mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.60, min_tracking_confidence=0.80)
hands = mp_hands.Hands(min_detection_confidence=0.50, min_tracking_confidence=0.70)

# mediapipe gesture model setup
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

gesture_options_right = mp.tasks.vision.GestureRecognizerOptions(
    running_mode=VisionRunningMode.LIVE_STREAM,
    base_options=mp.tasks.BaseOptions(model_asset_path=gesture_path),
    result_callback=get_result_right)

recognizer_left = mp.tasks.vision.GestureRecognizer.create_from_options(gesture_options_left)
recognizer_right = mp.tasks.vision.GestureRecognizer.create_from_options(gesture_options_right)

# mediapipe pose landmarker model setup
pose_path = "models/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def get_pose_result_left(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int, exercise_manager=exercise_manager):
    detection_result = result.gestures
    print(detection_result)


def get_pose_result_right(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int, exercise_manager=exercise_manager):
    detection_result = result.gestures
    print(detection_result)

pose_options_left = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_pose_result_left)

pose_options_right = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_pose_result_right)

landmarker_left = PoseLandmarker.create_from_options(pose_options_left)
landmarker_right = PoseLandmarker.create_from_options(pose_options_right)

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

        image.flags.writeable = True
        mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_left)
        mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_right)
        recognizer_left.recognize_async(mp_image_left, mp.Timestamp.from_seconds(time.time()).value)
        recognizer_right.recognize_async(mp_image_right, mp.Timestamp.from_seconds(time.time()).value)

        if hand_detections[0] == hand_detections[1] and hand_detections[0] == "Thumb_Up":
            game_state = GameStates.START_ROUND

    elif game_state == GameStates.START_ROUND:

        # update count down timer
        ready_cnt += delta_time

        
        if ready_cnt >= 3.5:
            game_state = GameStates.PLAYING

        image.flags.writeable = True

        # Display Exercise on screen
        exercise_text = exercise_manager.get_current_exercise()
        exercise_text_size = cv2.getTextSize(exercise_text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)
        exercise_text_x = (image.shape[1] - exercise_text_size[0][0]) // 2
        exercise_text_y = (image.shape[0] - exercise_text_size[0][1]) // 2
        box_margin = 15
        image[exercise_text_y - exercise_text_size[0][1] - box_margin : exercise_text_y + box_margin, 
              exercise_text_x - box_margin : exercise_text_x + exercise_text_size[0][0] + box_margin, ] = image[exercise_text_y - exercise_text_size[0][1] - box_margin : exercise_text_y + box_margin, 
                                                                                                                exercise_text_x - box_margin : exercise_text_x + exercise_text_size[0][0] + box_margin, ] * 0.5
        cv2.putText(image, exercise_text, (exercise_text_x, exercise_text_y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Display count down
        count_down_text = str(round(READY_TIME - ready_cnt))
        count_down_text_size = cv2.getTextSize(count_down_text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)
        count_down_text_x = (image.shape[1] - count_down_text_size[0][1]) // 2
        count_down_text_y = (image.shape[1] - count_down_text_size[0][0]) // 2 + exercise_text_size[0][1] + box_margin + 30
        cv2.putText(image, count_down_text, (count_down_text_x, count_down_text_y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
    elif game_state == GameStates.PLAYING:

        image_left = np.copy(image)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
        image_left[ :, image.shape[1]//2:, ] = [0, 0, 0]
        image_left.flags.writeable = False

        image_right = np.copy(image)
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
        image_right[ :, :image.shape[1]//2, ] = [0, 0, 0]
        image_right.flags.writeable = False

        image.flags.writeable = True
        mp_image_left = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_left)
        mp_image_right = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_right)
        landmarker_left.detect_async(mp_image_left, mp.Timestamp().from_seconds(time.time()).value)
        landmarker_right.detect_async(mp_image_right, mp.Timestamp().from_seconds(time.time()).value)


    # process image, convert from RGB to BRG
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    cv2.imshow('QuickDraw', image)

cam.release()
cv2.destroyAllWindows()  