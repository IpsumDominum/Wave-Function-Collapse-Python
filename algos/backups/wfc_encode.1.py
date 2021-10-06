import numpy as np
import cv2
import random
from collections import defaultdict
import math
import os
from algos.wfc_lib.wfc_utils import *


def get_encoded_patterns(input_img, N, VISUALIZE=False):
    pattern_set = {}
    hash_frequency_dict = defaultdict(lambda: 0)

    ENCODE_MODE = "overlap"
    GROUND = True

    if ENCODE_MODE == "overlap":
        PAD = 1
        input_padded = np.pad(input_img, ((PAD, PAD), (PAD, PAD), (0, 0)), "wrap")
        ground = np.zeros((input_img.shape[1]), dtype=np.int64)
        for i in range(input_padded.shape[0] - N + 1):
            for j in range(input_padded.shape[1] - N + 1):
                cropped_pattern = input_padded[i : i + N, j : j + N, :]
                for transform in [np.fliplr, np.flipud]:
                    transformed = transform(cropped_pattern.copy())
                    for _ in range(4):
                        transformed = np.rot90(transformed)
                        hash_code = hash_function(transformed)
                        pattern_set[hash_code] = transformed
                        hash_frequency_dict[hash_code] += 1
                hash_code = hash_function(cropped_pattern)
                pattern_set[hash_code] = cropped_pattern
                hash_frequency_dict[hash_code] += 1
                if i == input_padded.shape[0] - N and GROUND and j < input_img.shape[1]:
                    ground[j] = hash_code
                if VISUALIZE:
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
    else:
        PAD = 2 * N
        input_padded = np.pad(input_img, ((PAD, PAD), (PAD, PAD), (0, 0)), "wrap")
        ground = np.zeros(
            (math.ceil((input_img.shape[1] - N) // N) + 2), dtype=np.int64
        )
        for i in range(math.ceil((input_img.shape[0] - N) // N) + 2):
            for j in range(math.ceil((input_img.shape[1] - N) // N) + 2):
                cropped_pattern = input_padded[
                    PAD + i * N : PAD + (i + 1) * N, PAD + j * N : PAD + (j + 1) * N, :
                ]
                for transform in [np.fliplr, np.flipud]:
                    transformed = transform(cropped_pattern.copy())
                    for _ in range(4):
                        transformed = np.rot90(transformed)
                        hash_code = hash_function(transformed)
                        pattern_set[hash_code] = transformed
                        hash_frequency_dict[hash_code] += 1
                hash_code = hash_function(cropped_pattern)
                pattern_set[hash_code] = cropped_pattern
                hash_frequency_dict[hash_code] += 1
                if (
                    i == math.ceil((input_img.shape[0] - N) // N) + 1
                    and GROUND
                    and j < math.ceil((input_img.shape[1] - N) // N) + 2
                ):
                    ground[j] = hash_code
                if VISUALIZE:
                    show = cv2.resize(input_padded.copy(), (512, 512), interpolation=3)
                    scaleX = 512 // input_padded.shape[1]
                    scaleY = 512 // input_padded.shape[0]
                    cv2.rectangle(
                        show,
                        ((PAD + j * N) * scaleX, (PAD + i * N) * scaleY),
                        ((PAD + (j + 1) * N) * scaleX, (PAD + (i + 1) * N) * scaleY),
                        (0, 255, 0),
                        1,
                    )
                    cv2.imshow("show", show)
                    k = cv2.waitKey(0)
                    if k == ord("q"):
                        cv2.destroyAllWindows()
                        exit()
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
    return pattern_set, hash_frequency_dict, ground
