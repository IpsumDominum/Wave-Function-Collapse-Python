import cv2
import numpy as np
from sys import argv
import os
from algos.wfc_lib.wfc_encode import get_encoded_patterns
from algos.wfc_lib.wfc_utils import (
    hash_function,
    is_same_pattern,
    img_equal,
    cv_write,
    cv_img,
    cv_wait,
    get_hash_to_code,
)
from prep_utils import prepare_instructions

mouseX = 0
mouseY = 0


def mouse_callBack(event, x, y, flags, param):
    global mouseX, mouseY
    if event == 4:
        pass


image_dir = os.path.join("samples")
try:
    img_name = argv[1]
except IndexError:
    print("usage: python test.py --example_image.png")
    exit()

input_img = cv2.imread(os.path.join(image_dir, img_name)) / 255
N = 3
GROUND = False
# Put item in center,

# pattern_set,hash_frequency_dict,ground = get_encoded_patterns(input_img,N,GROUND=GROUND)
# pattern_code_set,hash_to_code_dict,avg_color_set,ground,code_frequencies = get_hash_to_code(pattern_set,ground,hash_frequency_dict,GROUND=GROUND)

cv2.namedWindow("main")
instructions = prepare_instructions()
cv2.setMouseCallback("main", mouse_callBack)
display_window = np.zeros((800, 1512, 3))
display_window[:800, :350] = instructions

# Draw some indicator blocks
for i in range(3):
    for j in range(3):
        display_window = cv2.rectangle(
            display_window,
            (412 + 50 + i * 200, 100 + j * 200),
            (412 + 50 + i * 200 + 200, 100 + j * 200 + 200),
            (255, 0, 0),
            5,
        )
        if i == 1 and j == 1:
            display_window[
                100 + j * 200 : 100 + j * 200 + 200,
                412 + 50 + i * 200 : 412 + 50 + i * 200 + 200,
            ] = cv2.resize(input_img, (200, 200))

cv2.imshow("main", display_window)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
# for pattern in pattern_code_set:
