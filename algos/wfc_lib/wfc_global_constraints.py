import numpy as np
import global_

# def initialise_global_constraints():
#     global_.unique_constraint = False
#     global_.unique_tiles_deleted = False
#     global_.max_non_unique = 2                # maximum number of unique pix a non unique-obj tile can hold
#     global_.unique_threshold = 0
#     global_.unique_pix = [59, 235, 255]    


# takes in a tile, and a RGB list representing a pixel to
# returns how many times the pixel appeared in the tile
def count_pixel(tile, pixel):
    count = 0
    for row in tile:
        for pix in row:
            if (pix == pixel).all(): count += 1
    return count


# takes in set of tiles, number of threshold to consider moon tile, thresholding pixel (eg. yellow for moon)
# returns pattern_code_set with deleted moon pixels
def delete_moon_tiles(output_matrix, pattern_code_set, thresh_count, thresh_pix):
    # # delete from output_matrix["valid_states"]
    # for i in range(len(output_matrix["valid_states"])):
    #     if count_pixel(pattern_code_set[i], thresh_pix) > thresh_count:
    #         # delete tiles
    #         valid_states = output_matrix["valid_states"][i]
    #         for row in range(valid_states.shape[0]):
    #             for col in range(valid_states.shape[1]):
    #                 output_matrix["valid_states"][i][row][col] = False
    #         # valid_states[:][:] = False

    for i in range(len(output_matrix["valid_states"])):
        if count_pixel(pattern_code_set[i], thresh_pix) > thresh_count:
            # delete tiles
            global_.deleted_tiles.append(i)
    return output_matrix 


# Takes in two tiles (patterns), sliced and sliced2
# Returns true if these two tiles satisfy global constraints if combined
# adjacency constraints eg. "beside each fully white tile, you cannot have blue pixels"
def adj_global_constr(sliced, sliced2):
    # add adjacency level constraints here

    # 26/09
    # sliced:  [[[r, g, b], [r, g, b]], 
    #           [[r, g, b], [r, g, b]],
    #           [[r, g, b], [r, g, b]]]  shape = (3, 2, 3) = (#rows, #col, #rgb)
    # sometimes does (2, 2, 3)

    # assuming (3, 3, 3)
    return True



# Changes the adjancencies of the output matrix to satisfy global constraints
# Returns the output matrix with modified adjacency matrixs
# matrix  constraints eg. "There can only be one moon, so once moon generated, cancel all moon tile adjacencies"
def matrix_global_constr(output_matrix, pattern_code_set, chosen_idx, array_index):
    # from global_ import unique_constraint, unique_tiles_deleted, max_non_unique, unique_threshold, unique_pix
    tile = pattern_code_set[chosen_idx]
    unique_count = count_pixel(tile, global_.unique_pix)    # count unique_pix
    
    if global_.unique_constraint == True and unique_count < 1 and global_.unique_tiles_deleted == False:  # if unique obj has been found and chosen tile has no unique obj pixels
        row_size = output_matrix["chosen_states"].shape[0]
        col_size = output_matrix["chosen_states"].shape[1]
        row = array_index[0]
        col = array_index[1]

        # check if unique pix at its top, left, or topleft by 1 index
        unassigned = [-1, -1, -1]
        left_code = int(output_matrix["chosen_states"][row][col-1]) 
        top_code = int(output_matrix["chosen_states"][row-1][col]) 
        topleft_code = int(output_matrix["chosen_states"][row-1][col-1]) 
        left_match = (pattern_code_set[left_code][0][0] == global_.unique_pix).all() if left_code != -1 else False
        top_match = (pattern_code_set[top_code][0][0] == global_.unique_pix).all() if top_code != -1 else False
        topleft_match = (pattern_code_set[topleft_code][0][0] == global_.unique_pix).all() if topleft_code != -1 else False

        if left_match or top_match or topleft_match:
            return output_matrix, pattern_code_set
        else:
            output_matrix = delete_moon_tiles(output_matrix, pattern_code_set, global_.unique_threshold, global_.unique_pix)
            global_.unique_tiles_deleted = True
            a=1

    elif unique_count > global_.max_non_unique:                     # if is unique obj (found)
        global_.unique_constraint = True
    
    
    # finish generating yellow tiles first, then delete
    # if moon_constraint == True:
    #     if yellow_count <= 0:
    #         # delete tiles
    # elif yellow_count > 2:
    #     moon_constraint = True
    return output_matrix, pattern_code_set


# Takes in the finalised output matrix, checks to see if it satisfies finalised-level global constraints
# Returns true or false depending on if it satisfy the constraints
# final constraints eg. "there must be more white pixels than blue in final picture"
def final_global_constr(output_matrix):
    return True