import numpy as np
import numpy.ma as ma
from algos.wfc_lib.wfc_utils import (cv_img,cv_wait,cv_write)
from algos.wfc_lib.wfc_backtrack import backtrack_memory, update_queue
from algos.wfc_lib.wfc_global_constraints import matrix_global_constr, count_pixel, delete_moon_tiles
import global_
import random
import imageio

LARGE_NUMBER = 131072


def pad_ground(output_matrix, ground,pattern_code_set,code_frequencies,avg_color_set,adjacency_matrices):
    output_matrix["valid_states"][:, -1, :] = False
    for item in ground:
        output_matrix["valid_states"][item, -1, :] = True    
    return output_matrix

def build_output_matrix(code_frequencies, output_w,output_h):
    output_matrix_valid_states = np.ones(
        (len(code_frequencies), output_w, output_h), dtype=bool
    )
    output_matrix_colors = np.zeros((output_w, output_h, 3))
    output_matrix_entropy = np.random.rand(output_w, output_h, 1) + np.sum(code_frequencies)
    output_matrix_chosen_states = (
        np.ones((output_w, output_h, 1), dtype=np.int64) * -1
    )
    output_matrix_timestep = np.ones((output_w, output_h, 1), dtype=np.int64) * 0
    output_matrix = {
        "valid_states": output_matrix_valid_states,
        "colors": output_matrix_colors,
        "entropy": output_matrix_entropy,
        "chosen_states": output_matrix_chosen_states,
        "timestep": output_matrix_timestep,
    }
    return output_matrix


def get_probs(frequency_list):
    return frequency_list / np.sum(frequency_list)

def choose_pattern(output_matrix,array_index,pattern_code_set,code_frequencies):
    # Collapse the Wave Function for this block here.
    # Choose from all the possible states based on adjacency constraint
    valid_states_list = np.array(list(pattern_code_set.keys()))[
        output_matrix["valid_states"][:, array_index[0], array_index[1]] > 0
    ]
    #==================
    # CONTRADICTION
    #==================
    if(len(valid_states_list)==0):
        contradiction = True    
    else:        
        #-------------------
        # Otherwise continue
        #-------------------
        #Get probabilities
        p = get_probs(
            code_frequencies[
                output_matrix["valid_states"][:, array_index[0], array_index[1]] > 0
            ]
        )
        #Set chosen pattern
        chosen_idx = (
            np.random.choice(valid_states_list, p=p)
        )
        #Update valid states
        output_matrix["chosen_states"][array_index[0], array_index[1]] = chosen_idx
        output_matrix["valid_states"][:, array_index[0], array_index[1]] = False
        output_matrix["valid_states"][chosen_idx, array_index[0], array_index[1]] = True
        contradiction = False

        output_matrix, pattern_code_set = matrix_global_constr(output_matrix, pattern_code_set, chosen_idx, array_index)
    return output_matrix, contradiction, pattern_code_set

def observe(output_matrix, pattern_code_set, hash_frequency_dict, code_frequencies):
    least_entropy_flat_index = np.argmin(output_matrix["entropy"])
    array_index = np.unravel_index(
        least_entropy_flat_index, output_matrix["entropy"].shape[:2]
    )
    done = (
        True
        if (
            output_matrix["entropy"][array_index[0], array_index[1]] == LARGE_NUMBER
            or output_matrix["entropy"][array_index[0], array_index[1]] < 1
        )
        else False
    )
    output_matrix,contradiction, pattern_code_set = choose_pattern(output_matrix,array_index,pattern_code_set,code_frequencies)
    return (done, contradiction,output_matrix, pattern_code_set)

def get_padded(output_matrix,PADMODE="FULLPERIODIC"):
    if(PADMODE=="FULLPERIODIC"):
        padded = np.pad(
        output_matrix["valid_states"],
        ((0, 0), (1, 1), (1, 1)),
        "wrap",
        )
    elif(PADMODE=="LRPERIODIC"):   
        padded = np.pad(
        output_matrix["valid_states"],
        ((0, 0), (0, 0), (1, 1)),
        "wrap",
        )
        padded = np.pad(
            padded,
            ((0, 0), (1, 1), (0, 0)),
            "constant",
            constant_values=True,
        ) 
    elif(PADMODE=="UDPERIODIC"):    
        padded = np.pad(
        output_matrix["valid_states"],
        ((0, 0), (1, 1), (0, 0)),
        "wrap",
        )
        padded = np.pad(
            padded,
            ((0, 0), (0, 0), (1, 1)),
            "constant",
            constant_values=True,
        ) 
    elif(PADMODE=="NOPERIODIC"):
        padded = np.pad(
            output_matrix["valid_states"],
            ((0, 0), (1, 1), (1, 1)),
            "constant",
            constant_values=True,
        )
    return padded

# around every yellow pixel, draw a 2N box around it, in this box if there are unchosen states, set entropy to 0
def prioritise_unique_obj(output_matrix, pattern_code_set, unique_obj_color, N):
    # matrix = [[-1, -1, -1, ...],
    #           [-1, 64, 43, ...]]
    chosen_states = output_matrix["chosen_states"]
    row = chosen_states.shape[0]
    col = chosen_states.shape[1]
    pix_matrix = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
            code = chosen_states[i][j]
            if code != -1:
                pix = pattern_code_set[int(code)][0][0]
                if (pix == unique_obj_color).all():         # if pixel matches unique color
                        uli = max(i - N, 0)   # upper left corners pixel/tile of i and j
                        ulj = max(j - N, 0)
                        for i2 in range(uli, uli+2*N):
                            for j2 in range(ulj, ulj+2*N):
                                out_i = min(i2, row-1)
                                out_j = min(j2, col-1)
                                output_matrix["entropy"][out_i][out_j] = output_matrix["entropy"][out_i][out_j] - LARGE_NUMBER if chosen_states[out_i][out_j] == -1 else output_matrix["entropy"][out_i][out_j]
    return output_matrix


# prioritise top left of unique obj pixels (tile based)
def prioritise_unique_obj_top_left_tile(output_matrix, pattern_code_set, unique_obj_color, N):
    matrix = output_matrix["chosen_states"]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if count_pixel([matrix[i][j]], unique_obj_color) > 0:
                output_matrix["entropy"][i][j] = output_matrix["entropy"][i][j] - LARGE_NUMBER
    return output_matrix


# prioritise top left of unique obj pixels, also prioritises +1 downwards and rightwards until there are no yellow pixels
def prioritise_unique_obj_top_left(output_matrix, pattern_code_set, unique_obj_color, N):
    chosen_states = output_matrix["chosen_states"]
    row = chosen_states.shape[0]
    col = chosen_states.shape[1]
    pix_matrix = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
            code = chosen_states[i][j]
            if code != -1:      # if state is chosen
                pix = pattern_code_set[int(code)][0][0]
                if (pix == unique_obj_color).all():         # if pixel matches unique color
                        uli = max(i - (N-1), 0)   # upper left corners pixel/tile of i and j
                        ulj = max(j - (N-1), 0)
                        for i2 in range(uli, uli+N):
                            for j2 in range(ulj, ulj+N):
                                out_i = min(i2, row-1)
                                out_j = min(j2, col-1)
                                output_matrix["entropy"][out_i][out_j] = output_matrix["entropy"][out_i][out_j] - LARGE_NUMBER if chosen_states[out_i][out_j] == -1 else output_matrix["entropy"][out_i][out_j]
                        inc_i = min(i+1, row-1)
                        inc_j = min(j+1, col-1)
                        output_matrix["entropy"][inc_i][j] = output_matrix["entropy"][inc_i][j] - LARGE_NUMBER if chosen_states[inc_i][j] == -1 else output_matrix["entropy"][inc_i][j]
                        output_matrix["entropy"][i][inc_j] = output_matrix["entropy"][i][inc_j] - LARGE_NUMBER if chosen_states[i][inc_j] == -1 else output_matrix["entropy"][i][inc_j]
                        output_matrix["entropy"][inc_i][inc_j] = output_matrix["entropy"][inc_i][inc_j] - LARGE_NUMBER if chosen_states[inc_i][inc_j] == -1 else output_matrix["entropy"][inc_i][inc_j]
    return output_matrix


def propagate(output_matrix, avg_color_set, adjacency_matrices, code_frequencies,directions_list, N, SPECS, pattern_code_set):
    #import time

    #start = time.time()
    # Elegant global propagation, reference to Issac Karth Implementation
    # For each direction, get the supports as matrix multiplication of the
    # Shifted array and the valid adjacency
    support = {}
    try:
        padded = get_padded(output_matrix,PADMODE=SPECS["PADMODE"])
    except KeyError:
        padded = get_padded(output_matrix,PADMODE="FULLPERIODIC")
    
    # Compute based on the current valid states, the neighbour's valid states.
    # Hence propagate the constraint wave.
    for direction in directions_list:
        dy, dx = direction
        shifted = padded[
            :,
            1 + dy : 1 + dy + output_matrix["valid_states"].shape[1],
            1 + dx : 1 + dx + output_matrix["valid_states"].shape[2],
        ]
        support[direction] = (
            adjacency_matrices[direction] @ shifted.reshape(shifted.shape[0], -1)
        ).reshape(shifted.shape) > 0
        # Update output_matrix valid states
    for direction in directions_list:
        output_matrix["valid_states"] *= support[direction]     
        
    # if global_.unique_tiles_deleted == True:
    #     output_matrix = delete_moon_tiles(output_matrix, pattern_code_set, global_.unique_threshold, global_.unique_pix)

    #print("PROPAGATION TOOK : ",time.time()-start)
    #start = time.time()
    # Update new entropy for the matrix based on current available states
    for i in range(output_matrix["entropy"].shape[0]):
        for j in range(output_matrix["entropy"].shape[1]):
            """
            valid_patterns = np.arange(output_matrix["valid_states"].shape[0])[
                output_matrix["valid_states"][:, i, j] > 0
            ]
            if(len(valid_patterns)==1):
                output_matrix["chosen_states"][i][j] = valid_patterns[0]
                output_matrix["entropy"][i][j] = LARGE_NUMBER
            """
            output_matrix["entropy"][i][j] = np.sum(code_frequencies[output_matrix["valid_states"][:, i, j] > 0])
            output_matrix["entropy"][i][j] = (
                LARGE_NUMBER
                if (output_matrix["chosen_states"][i][j] != -1)
                else output_matrix["entropy"][i][j] + random.random()
            )
            # output_matrix["colors"][i][j] = np.average(
            #     avg_color_set[output_matrix["valid_states"][:, i, j] > 0], axis=0
            # )
            output_matrix["colors"][i][j] = [0,90,9]        # TEMP
    output_matrix = prioritise_unique_obj_top_left(output_matrix, pattern_code_set, global_.unique_pix, N)
    #print("ENTROPY TOOK : ",time.time()-start)
    return output_matrix


def render(
    output_matrix, output_w,output_h, N, pattern_code_set, VISUALIZE=True, WRITE_VIDEO=False
):
    output_image = np.zeros((output_w + N - 1, output_h + N - 1, 3))
    for i in range(output_matrix["chosen_states"].shape[0]):
        for j in range(output_matrix["chosen_states"].shape[1]):
            pattern_code = output_matrix["chosen_states"][i, j, 0]
            avg_color = output_matrix["colors"][i, j, :]
            output_image[i : i + N, j : j + N] = (
                pattern_code_set[pattern_code] if pattern_code != -1 else avg_color
            )

    if VISUALIZE:
        cv_img(output_image)
    return output_image
    
