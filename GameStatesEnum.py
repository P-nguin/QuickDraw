from enum import Enum

class GameStates(Enum):
    PAUSE = -1
    MAIN_MENU = 0
    START_ROUND = 1
    PLAYING = 2
    END_ROUND = 3
    GAME_OVER = 4