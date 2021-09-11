# Takes in two tiles (patterns), sliced and sliced2
# Returns true if these two tiles satisfy global constraints if combined
# adjacency constraints eg. "beside each fully white tile, you cannot have blue pixels"
def adj_global_constr(sliced, sliced2):
    # add adjacency level constraints here
    return True

# Takes in the current output matrix, checks to see if the adjacency needs any changes
# Changes the adjancencies of the output matrix to satisfy global constraints
# Returns the output matrix with modified adjacency matrixs
# matrix  constraints eg. "There can only be one moon, so once moon generated, cancel all moon tile adjacencies"
def matrix_global_constr(output_matrix):
    # add output_matrix level constraints here
    return None


# Takes in the finalised output matrix, checks to see if it satisfies finalised-level global constraints
# Returns true or false depending on if it satisfy the constraints
# final constraints eg. "there must be more white pixels than blue in final picture"
def final_global_constr(output_matrix):
    return True
