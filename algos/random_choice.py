import numpy as np
import cv2
import random

"""
Sample over NxN crops of Input image, to create selection array
for output, each grid selects out of random one of the selection array membe
"""


def random_choice_run(input_img, N=3, output_size=300):
    if N == 0:
        N = 1
    cropped_sets = []
    channels = input_img.shape[2]
    output_shape = np.zeros((output_size, output_size, channels))
    for i in range(input_img.shape[0] // N):
        for j in range(input_img.shape[1] // N):
            cropped_sets.append(input_img[i * N : (i + 1) * N, j * N : (j + 1) * N, :])

    for i in range(output_size // N):
        for j in range(output_size // N):
            sampled = random.sample(cropped_sets, k=1)[0]
            output_shape[(i * N) : (i + 1) * N, (j * N) : (j + 1) * N, :] = sampled
    cv2.imshow("output", cv2.resize(output_shape / 100, (512, 512)))
    cv2.imshow("input", cv2.resize(input_img, (128, 128)))
    k = cv2.waitKey(0)
    if k == ord("q"):
        exit()


def random_choice_run(input_img, N=3, output_size=300):
    if N == 0:
        N = 1
    cropped_sets = []
    channels = input_img.shape[2]
    output_shape = np.zeros((output_size, output_size, channels))
    for i in range(input_img.shape[0] // N):
        for j in range(input_img.shape[1] // N):
            cropped_sets.append(input_img[i * N : (i + 1) * N, j * N : (j + 1) * N, :])

    for i in range(output_size // N):
        for j in range(output_size // N):
            sampled = random.sample(cropped_sets, k=1)[0]
            output_shape[(i * N) : (i + 1) * N, (j * N) : (j + 1) * N, :] = sampled
    cv2.imshow("output", cv2.resize(output_shape / 100, (512, 512)))
    cv2.imshow("input", cv2.resize(input_img, (128, 128)))
    k = cv2.waitKey(0)
    if k == ord("q"):
        exit()
