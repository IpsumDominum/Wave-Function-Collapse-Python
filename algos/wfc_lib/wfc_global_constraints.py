import numpy as np


moon_constraint = False

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

    # delete from pattern_code_set
    # for i in range(len(pattern_code_set)-1, -1, -1):
    #     # tile = pattern_code_set[i]
    #     if count_pixel(pattern_code_set[i], thresh_pix) > thresh_count:
    #         pattern_code_set.pop(i)
    
    # delete from output_matrix["valid_states"]
    for i in range(len(output_matrix["valid_states"])):
        tile_availability = output_matrix["valid_states"][i]
        tile_availability[:][:] = False
    return pattern_code_set


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


# Takes in the current output matrix, adjacency list, pattern to code set, assigned tile index
# Changes the adjancencies of the output matrix to satisfy global constraints
# Returns the output matrix with modified adjacency matrixs
# matrix  constraints eg. "There can only be one moon, so once moon generated, cancel all moon tile adjacencies"
def matrix_global_constr(output_matrix, pattern_code_set, chosen_idx):
    max_non_moon = 2
    moon_threshold = 3

    tile = pattern_code_set[chosen_idx]
    moon_yellow = [59, 235, 255]                        # yellow = 59 235 255
    yellow_count = count_pixel(tile, [59, 235, 255])    # count yellow color (for moon / window)
    if yellow_count > max_non_moon:
        pattern_code_set = delete_moon_tiles(output_matrix, pattern_code_set, moon_threshold, moon_yellow)

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



