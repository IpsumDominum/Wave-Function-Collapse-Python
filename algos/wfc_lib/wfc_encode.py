import numpy as np
import cv2
import random
from collections import defaultdict
import math
import os
from algos.wfc_lib.wfc_utils import *


def visualize_wfc_encode(input_padded, i, j, N):
    show = cv2.resize(input_padded.copy(), (512, 512), interpolation=3)
    scaleX = 512 // input_padded.shape[1]
    scaleY = 512 // input_padded.shape[0]
    cv2.rectangle(
        show,
        (j * scaleX, i * scaleY),
        ((j + N) * scaleX, (i + N) * scaleY),
        (0, 255, 0),
        10,
    )
    cv2.imshow("show", show)
    k = cv2.waitKey(0)
    if k == ord("q"):
        cv2.destroyAllWindows()
        exit()


def write_encoded_patterns(pattern_set):
    if os.path.isdir(os.path.join("vis", "encode")):
        for item in os.listdir(os.path.join("vis", "encode")):
            os.remove(os.path.join("vis", "encode", item))
    else:
        os.makedirs(os.path.join("vis", "encode"))
    for pattern in pattern_set:
        cv2.imwrite(
            os.path.join("vis", "encode", str(pattern) + ".png"),
            cv2.resize(pattern_set[pattern], (256, 256), interpolation=3),
        )

def get_encoded_patterns(
    input_img, N, VISUALIZE=False, GROUND={}, WRITE=False, SPECS={}
):
    pattern_set = {}
    hash_frequency_dict = defaultdict(lambda: 0)

    PAD = 0
    input_padded = np.pad(input_img, ((PAD, PAD), (PAD, PAD), (0, 0)), "wrap")
    # Ground is for things like Flowers.png
    ground = defaultdict(lambda:0)
    transforms = [np.fliplr, np.flipud,np.array]
    for i in range(input_padded.shape[0] - N+1):
        for j in range(input_padded.shape[1] - N+1):
            cropped_pattern = input_padded[i : i + N, j : j + N, :]

            hash_code = hash_function(cropped_pattern)
            pattern_set[hash_code] = cropped_pattern
            hash_frequency_dict[hash_code] += 1

            for transform in transforms:
                transformed = transform(cropped_pattern.copy())                
                for _ in range(3):
                    transformed = np.rot90(transformed)
                    hash_code = hash_function(transformed)
                    pattern_set[hash_code] = transformed
                    hash_frequency_dict[hash_code] += 1
            if GROUND :
                if(i==input_padded.shape[0]-N):
                    hash_code = hash_function(cropped_pattern)
                    ground[hash_code] +=1
                    hash_frequency_dict[hash_code] = 1
            if VISUALIZE :
                visualize_wfc_encode(input_padded, i, j, N)
    if WRITE:
        write_encoded_patterns(pattern_set)

    return pattern_set, hash_frequency_dict, ground
