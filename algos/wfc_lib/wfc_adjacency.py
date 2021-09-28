import numpy as np
import cv2
import random
import math
from collections import defaultdict
import os
from tqdm import tqdm
from algos.wfc_lib.wfc_utils import (
    hash_function,
    is_same_pattern,
    img_equal,
    cv_write,
    cv_img,
    cv_wait,
    get_hash_to_code,
)


def visualize_adjacency(image, image2, dy, dx, N, match, sliced2, sliced):
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=(30))
    padded2 = np.pad(image2, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=(30))
    show = cv2.resize(padded.copy(), (512, 512), interpolation=3) / 255
    cv2.rectangle(
        show,
        ((1 + dx) * 512 // (N + 2), (1 + dy) * 512 // (N + 2)),
        ((1 + N + dx) * 512 // (N + 2), (1 + N + dy) * 512 // (N + 2)),
        (0, 255, 0),
        10,
    )
    # Slice opposite side of image
    show2 = cv2.resize(padded2.copy(), (512, 512), interpolation=3) / 255
    overlay = padded.copy()
    if sliced2.shape == (3, 2, 3):
        overlay[1 + dy : 1 + dy + N, 1 + dx : 1 + dx + N - 1] = sliced2
    elif sliced2.shape == (2, 2, 3):
        overlay[1 + dy : 1 + dy + N - 1, 1 + dx : 1 + dx + N - 1] = sliced2
    elif sliced2.shape == (2, 3, 3):
        overlay[1 + dy : 1 + dy + N - 1, 1 + dx : 1 + dx + N] = sliced2
    overlay = cv2.resize(overlay, (512, 512), interpolation=3) / 255
    if match == True:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        show2,
        ((1 - dx) * 512 // (N + 2), (1 - dy) * 512 // (N + 2)),
        ((1 + N - dx) * 512 // (N + 2), (1 + N - dy) * 512 // (N + 2)),
        color,
        10,
    )
    cv2.rectangle(
        overlay,
        ((1) * 512 // (N + 2), (1) * 512 // (N + 2)),
        ((1 + N) * 512 // (N + 2), (1 + N) * 512 // (N + 2)),
        color,
        10,
    )
    cv2.imshow("overlay", overlay)
    cv2.imshow("show", show)
    cv2.imshow("show2", show2)
    cv_img(image, id="image")
    cv_img(image2, id="image2")
    cv_img(sliced, id="sliced")
    cv_img(sliced2, id="sliced2")
    k = cv2.waitKey(0)
    if k == ord("q"):
        cv2.destroyAllWindows()
        exit()


def extract_adjacency(hash_to_code_dict, pattern_set, N, directions_list,VISUALIZE=False):
    adjacency_list = defaultdict(lambda: [])
    # EXTRACT ADJACENCY
    # For all of the tiles, create a list of toleratable overlap adjacencies.
    for item in tqdm(pattern_set.keys()):
        image = pattern_set[item]        
        # For every other item, see if the overlap is OK
        for item2 in pattern_set.keys():
            image2 = pattern_set[item2]
            for direction in directions_list:
                # Use slicing to move pad up
                dy, dx = direction
                sliced = image[
                    max(dy, 0) : N + min(dy, 0) :, max(dx, 0) : N + min(dx, 0)
                ]
                sliced2 = image2[
                    -min(dy, 0) : N - max(dy, 0), -min(dx, 0) : N - max(dx, 0)
                ]
                match = False
                if img_equal(sliced, sliced2):
                    match = True
                    adjacency_list[direction].append(
                        (hash_to_code_dict[item], hash_to_code_dict[item2])
                    )
                # Slice the overlapping region of the two images to compare, and check if the overlap is the same.
                if VISUALIZE:
                    visualize_adjacency(
                    image, image2, dy, dx, N, match, sliced2, sliced
                    )
    return adjacency_list
def extract_adjacency_from_image(input_img,hash_to_code_dict, pattern_set, N, directions_list,VISUALIZE=False):
    # EXTRACT ADJACENCY FROM IMAGE
    adjacency_list = defaultdict(lambda: [])
    input_shape_i = input_img.shape[0]
    input_shape_j = input_img.shape[1]
    PAD = 0
    input_padded = np.pad(input_img, ((PAD, PAD), (PAD, PAD), (0, 0)), "wrap")
    for i in range(math.ceil((input_shape_i - N) // N) + 2):
        for j in range(math.ceil((input_shape_j - N) // N) + 2):
            crop_img = input_padded[
                (i) * N + PAD : (i + 1) * N + PAD, (j) * N + PAD : (j + 1) * N + PAD, :
            ]
            hash_code = hash_function(crop_img)
            for index, directions in enumerate(directions_list):
                idx = (i * N + PAD + (directions[0] - 1) * N) % input_padded.shape[0]
                jdx = (j * N + PAD + (directions[1] - 1) * N) % input_padded.shape[1]
                adjacent_img = input_padded[idx : idx + N, jdx : jdx + N, :]
                hash_code_adj = hash_function(adjacent_img)
                # For the direction in valid_neighbours,append the adjacent hashcode
                adjacency_list[directions].append(
                        (hash_to_code_dict[hash_code], hash_to_code_dict[hash_code_adj])
                )
    return adjacency_list

def write_adjacency_visualize(adjacency_list, pattern_code_set):
    if os.path.isdir(os.path.join("vis", "adjacency")):
        for item in os.listdir(os.path.join("vis", "adjacency")):
            os.remove(os.path.join("vis", "adjacency", item))
    else:
        os.makedirs(os.path.join("vis", "adjacency"))

    for d in adjacency_list:
        for i, adj in enumerate(adjacency_list[d]):
            dy, dx = d
            img1 = pattern_code_set[adjacency_list[d][i][0]]
            img2 = pattern_code_set[adjacency_list[d][i][1]]
            padded = np.pad(
                img1, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=50
            )
            N = img1.shape[0]
            padded[1 + dy : 1 + dy + N, 1 + dx : 1 + dx + N] = img2
            cv2.imwrite(
                os.path.join(
                    "vis",
                    "adjacency",
                    str(d[0])
                    + str(d[1])
                    + "_"
                    + str(adj[0])
                    + "_"
                    + str(adj[1])
                    + ".png",
                ),
                cv2.resize(padded, (512, 512), interpolation=3),
            )


def build_adjacency_matrix(adjacency_list, pattern_code_set, WRITE=False):
    adjacency_matrices = {}
    num_patterns = len(pattern_code_set.keys())
    for d in adjacency_list:
        m = np.zeros((num_patterns, num_patterns), dtype=bool)
        for i, adj in enumerate(adjacency_list[d]):
            m[adj[0], adj[1]] = True
        # Optional to use sparse matrix.
        # adjacency_matrices[d] = sparse.csr_matrix(m)
        adjacency_matrices[d] = m
    if WRITE:
        write_adjacency_visualize(adjacency_list, pattern_code_set)
    return adjacency_matrices
