import numpy as np

# def backtrack(output_matrix):
#     most_recent_index = np.argmin(output_matrix["timestep"])
#     array_index = np.unravel_index(most_recent_index, output_matrix["entropy"].shape[:2])
#     output_matrix["valid_states"][:, array_index[0], array_index[1]] = np.ones((pattern_num,output_size,output_size),dtype=bool)
#     output_matrix["chosen_states"][array_index[0],array_index[1]] = -1


# Takes in an output matrix, previous output matrix, and number of backtrack desired
# Outputs the output matrix of the desired previous state
# Backtrack number = 1 means backtrack only once (only to previous state)
def backtrack_memory(output_matrix, backtrack_queue, backtrack_no=1):
    backtrack = backtrack_no - 1
    if backtrack_queue[backtrack] == -1:         # if no previous state of the output matrix
        print("no previous state to backtrack on")
        raise Exception
    return backtrack_queue[backtrack]

# Updates backtrack queue with the new matrix, returns backtrack_queue with new matrix in it
def update_queue(backtrack_queue, new_matrix):
    backtrack_queue = backtrack_queue[:-1]
    backtrack_queue.insert(0, new_matrix)
    return backtrack_queue
