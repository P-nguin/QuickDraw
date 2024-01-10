import cv2
import mediapipe as mp 
import numpy as np 
import time
import Exercises
from GameStates import GameStates

# mp models setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hand = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.80, min_tracking_confidence=0.80)
hands = mp_hand.Hands(min_detection_confidence=0.80, min_tracking_confidence=0.80)

game_state = GameStates.MAIN_MENU
READY_TIME = 3
ready_cnt = 0



# setup cmaera
cam = cv2.VideoCapture(0)
start = time.time()
while cam.isOpened():
    ret, frame = cam.read()
    delta_time = time.time() - start
    start = time.time()

    # get keyboard input
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    # process image, convert from BRG to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    if game_state == GameStates.MAIN_MENU:
        # proccess image through hand detection
        # crop photo in half horizontally
        # if both hands are giving thumbs up, then move to next screen.

        image_left = cv2.cvtColor(image[ :, :image.shape[1]//2], cv2.COLOR_BGR2RGB)
        image_left.flags.writeable = False

        image_right = cv2.cvtColor(image[ :, image.shape[1]//2:], cv2.COLOR_BGR2RGB)
        image_right.flags.writeable = False
        
        hands_results_left = hands.process(np.ascontiguousarray(image_left))
        hands_results_right = hands.process(np.ascontiguousarray(image_right))
        # https://www.youtube.com/watch?v=0W4nRBPu1hQ here for gesture dectection
    elif game_state == GameStates.PLAYING:
        pass

    # process image, convert from RGB to BRG
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    cv2.imshow('QuickDraw', image)

cam.release()
cv2.destroyAllWindows()  