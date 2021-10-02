import numpy as np
import cv2
import random
from collections import defaultdict
import math
import os
from algos.wfc_lib.wfc_utils import *



def visualize_wfc_encode(input_padded, i, j, N,ENCODE):
    show = cv2.resize(input_padded.copy(), (512, 512), interpolation=3)
    scaleX = 512 // input_padded.shape[1]
    scaleY = 512 // input_padded.shape[0]
    if(ENCODE=="overlap"):
        cv2.rectangle(
            show,
            (j * scaleX, i * scaleY),
            ((j + N) * scaleX, (i + N) * scaleY),
            (0, 255, 0),
            10,
        )
    elif(ENCODE=="tiled"):
        cv2.rectangle(
            show,
            (j*N * scaleX, i*N * scaleY),
            (((j+1)*N) * scaleX, ((i+1)*N) * scaleY),
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

def identity_grid(grid):
    return np.array(grid)
def reflect_grid(grid):
    return np.fliplr(grid)
def rotate_grid(grid):
    return np.rot90(grid,axes=(1, 0))
def get_encoded_patterns(
    input_img, N, VISUALIZE=False, GROUND={}, WRITE=False, SPECS={}
):
    pattern_set = {}
    hash_frequency_dict = defaultdict(lambda: 0)

    PAD = 0
    input_padded = np.pad(input_img, ((PAD, PAD), (PAD, PAD), (0, 0)), "wrap")
    # Ground is for things like Flowers.png
    ground = defaultdict(lambda:0)
    transforms = [np.array]    
    transforms = [np.fliplr, np.flipud,np.array]    
    transforms = [np.fliplr,np.array]    
    grid_ops = [
        identity_grid,
        reflect_grid,
        #rotate_grid,
        #reflect_grid,
        #rotate_grid,
        #reflect_grid,
        #rotate_grid,
        #reflect_grid,
    ]
    ENCODE = "overlap"    

    for op_idx,op in enumerate(grid_ops):
        input_padded = op(input_padded.copy())
        i_range = input_padded.shape[0] - N+1 if ENCODE=="overlap" else input_padded.shape[0]//N + 1
        j_range = input_padded.shape[1] - N+1 if ENCODE=="overlap" else input_padded.shape[1]//N
        for i in range(i_range):
            for j in range(j_range):
                cropped_pattern = input_padded[i : i + N, j : j + N, :] if ENCODE=="overlap" else input_padded[i*N : (i+1)*N, j*N : (j+1)*N, :]
                hash_code = hash_function(cropped_pattern)
                pattern_set[hash_code] = cropped_pattern
                hash_frequency_dict[hash_code] += 1
                """
                for transform in transforms:
                    transformed = transform(cropped_pattern.copy())                
                    for _ in range(2):
                        transformed = np.rot90(transformed)
                        hash_code = hash_function(transformed)
                        pattern_set[hash_code] = transformed
                        hash_frequency_dict[hash_code] += 1
                """
                if GROUND and op_idx==0:
                    if( (i==SPECS["GROUND_LEVEL"] and ENCODE=="overlap")
                        or 
                        (i==SPECS["GROUND_LEVEL"] and ENCODE!="overlap")
                    ):
                        
                        hash_code = hash_function(cropped_pattern)                    
                        ground[hash_code] +=1
                if VISUALIZE :
                    print(i)
                    visualize_wfc_encode(input_padded, i, j, N,ENCODE)
    if WRITE:
        write_encoded_patterns(pattern_set)
    
    return pattern_set, hash_frequency_dict, ground
