import cv2
import numpy as np


def prepare_instructions():
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    instructions = np.ones((800, 350, 3))
    color = (255, 0, 0)
    instructions = cv2.putText(
        instructions,
        "Q to save and quit",
        (50, 50),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    color = (0, 0, 0)
    instructions = cv2.putText(
        instructions,
        "Z to undo",
        (50, 100),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    color = (0, 0, 0)
    instructions = cv2.putText(
        instructions,
        "D to move forward",
        (50, 150),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    color = (0, 0, 0)
    instructions = cv2.putText(
        instructions,
        "A to move backward",
        (50, 200),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return instructions
