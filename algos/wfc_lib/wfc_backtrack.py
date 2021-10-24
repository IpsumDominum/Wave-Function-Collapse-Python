import numpy as np
"""
Author: Chong, Chen
"""

# Initialise backtrack
def prepare_backtrack(output_matrix,max_backtrack=5):
    backtrack_no = 1
    backtrack_queue = []
    backtrack_queue = update_queue(backtrack_queue, output_matrix, max_backtrack)
    return backtrack_queue, output_matrix, backtrack_no


# Takes in an output matrix, previous output matrix, and number of backtrack desired
# Outputs the output matrix of the desired previous state
# Backtrack number = 1 means backtrack only once (only to previous state)
def backtrack_memory(backtrack_queue, max_backtrack, backtrack_no=1):
    backtrack = backtrack_no 
    try: 
        assert backtrack_queue[-1 * backtrack] != -1 # if no more previous state of the output matrix  
    except:
        print("no previous states to backtrack to")
    backtrack_state = backtrack_queue[-1 * backtrack]
    if backtrack > 1:
        backtrack_queue[-1 * (backtrack-1)] = -1       # set prev state as empty state
    return backtrack_state

# Updates backtrack queue with the new matrix, returns backtrack_queue with new matrix in it
def update_queue(backtrack_queue, output_matrix, max_backtrack):
    backtrack_queue.append(output_matrix)
    if len(backtrack_queue) > max_backtrack:
        backtrack_queue.pop(0)
    for i in range(len(backtrack_queue)-1, -1, -1):
        if backtrack_queue[i] == -1:
            backtrack_queue.pop(i)
    return backtrack_queue