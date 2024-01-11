import numpy as np
import mediapipe as mp
from enum import IntEnum

class Exercises(IntEnum):
    PUSHUPS = 0
    SQUAT = 1
    PLANK = 2
    PISTOL_SQUAT_RIGHT = 3
    PISTOL_SQUAT_LEFT = 4

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def check_exercise(action, landmark):
    if action == Exercises.PUSHUPS:
        
        pass # TODO: do it