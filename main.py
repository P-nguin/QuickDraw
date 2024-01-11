import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np 
import time

import Exercises
from GameStatesEnum import GameStates

# mp models setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.60, min_tracking_confidence=0.80)
hands = mp_hands.Hands(min_detection_confidence=0.50, min_tracking_confidence=0.70)

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

    # process image, convert from BRG to RGB and flip horizontally
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False

    if game_state == GameStates.MAIN_MENU:
        # proccess image through hand detection
        # crop photo in half horizontally
        # if both hands are giving thumbs up, then move to next screen.

        image_left = cv2.cvtColor(image[ :, :image.shape[1]//2], cv2.COLOR_BGR2RGB)
        image_left.flags.writeable = False

        image_right = cv2.cvtColor(image[ :, image.shape[1]//2:], cv2.COLOR_BGR2RGB)
        image_right.flags.writeable = False
        
        hands_results_left = hands.process(image_left)
        hands_results_right = hands.process(np.ascontiguousarray(image_right))

        if hands_results_left.multi_hand_landmarks:
            landmarks = []
            for handslms in hands_results_left.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * image.shape[1])
                    lmy = int(lm.y * image.shape[1])

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                #mp_drawing.draw_landmarks(image, handslms, mp_hands.HAND_CONNECTIONS)
                    
        image.flags.writeable = True
        if hands_results_left.multi_hand_landmarks:
            for num, hand in enumerate(hands_results_left.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )

    elif game_state == GameStates.PLAYING:
        pass

    # process image, convert from RGB to BRG
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    cv2.imshow('QuickDraw', image)

cam.release()
cv2.destroyAllWindows()  