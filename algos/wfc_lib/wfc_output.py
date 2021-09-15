import numpy as np
import numpy.ma as ma
from algos.wfc_lib.wfc_utils import cv_img, cv_wait, cv_write
import random
import imageio


LARGE_NUMBER = 1000000

def pad_ground(output_matrix, ground,pattern_code_set,code_frequencies):
    output_matrix["valid_states"][:, -1, :] = False
    for item in ground:
        output_matrix["valid_states"][item, -1, :] = True
    return output_matrix

def build_output_matrix(code_frequencies, output_size):
    output_matrix_valid_states = np.ones(
        (len(code_frequencies), output_size, output_size), dtype=bool
    )
    output_matrix_colors = np.zeros((output_size, output_size, 3))
    output_matrix_entropy = np.random.rand(output_size, output_size, 1) + np.sum(code_frequencies)
    output_matrix_chosen_states = (
        np.ones((output_size, output_size, 1), dtype=np.int64) * -1
    )
    output_matrix = {
        "valid_states": output_matrix_valid_states,
        "colors": output_matrix_colors,
        "entropy": output_matrix_entropy,
        "chosen_states": output_matrix_chosen_states,
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
    return output_matrix,contradiction

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
    output_matrix,contradiction = choose_pattern(output_matrix,array_index,pattern_code_set,code_frequencies)
    #Return output matrix
    return (done, contradiction,output_matrix)


def propagate(output_matrix, avg_color_set, adjacency_matrices, code_frequencies):
    # Elegant global propagation, reference to Issac Karth Implementation
    # For each direction, get the supports as matrix multiplication of the
    # Shifted array and the valid adjacency
    directions_list = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]
    directions_list = [(0,-1),(0,1),(-1,0),(1,0)]
    support = {}
    padded = np.pad(
        output_matrix["valid_states"],
        ((0, 0), (1, 1), (1, 1)),
        "constant",
        constant_values=True,
    )
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

    # Update new entropy for the matrix based on current available states
    for i in range(output_matrix["entropy"].shape[0]):
        for j in range(output_matrix["entropy"].shape[1]):
            valid_patterns = output_matrix["valid_states"][:, i, j][
                output_matrix["valid_states"][:, i, j] > 0
            ]
            output_matrix["entropy"][i][j] = np.sum(code_frequencies[output_matrix["valid_states"][:, i, j] > 0])
            output_matrix["entropy"][i][j] = (
                LARGE_NUMBER
                if (output_matrix["chosen_states"][i][j] != -1)
                else output_matrix["entropy"][i][j] + random.random() * 0.5
            )
            output_matrix["colors"][i][j] = np.average(
                avg_color_set[output_matrix["valid_states"][:, i, j] > 0], axis=0
            )

    return output_matrix


def render(
    output_matrix, output_size, N, pattern_code_set, VISUALIZE=True, WRITE_VIDEO=False
):
    output_image = np.zeros((output_size + N - 1, output_size + N - 1, 3))
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
