import numpy as np
import cv2
import random
import math
import copy
from collections import defaultdict
import os
import imageio
from algos.wfc_lib.wfc_global_constraints import final_global_constr
from algos.wfc_lib.wfc_global_constraints import matrix_global_constr

from algos.wfc_lib.wfc_utils import (
    hash_function,
    is_same_pattern,
    img_equal,
    cv_write,
    cv_img,
    cv_wait,
    get_hash_to_code,
    prepare_write_outputs,
)
from algos.wfc_lib.wfc_encode import get_encoded_patterns
from algos.wfc_lib.wfc_adjacency import extract_adjacency, build_adjacency_matrix,extract_adjacency_from_image
from algos.wfc_lib.wfc_output import (
    build_output_matrix,
    observe,
    propagate,
    render,
    pad_ground,
)
from algos.wfc_lib.wfc_backtrack import (
    backtrack_memory,
    update_queue,
    prepare_backtrack
)

"""
Sample over NxN crops of Input image, to create selection array
for output, each grid selects out of random one of the selection array membe
"""


def wfc_overlap_run(
    input_img,
    N=3,
    output_w=32,
    output_h=32,
    output_name="out_video.avi",
    GROUND=False,
    WRITE=False,
    VISUALIZE_ENCODE=False,
    VISUALIZE_ADJACENCY=False,
    MAX_BACKTRACK = 5,
    WRITE_VIDEO=False,
    SPECS={},
):  
    ###################################################
    print("RUNNING")
    video_out = []
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
    directions_list = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ###################################################
    print("ENCODING...")
    pattern_set, hash_frequency_dict, ground = get_encoded_patterns(
        input_img,
        N,
        VISUALIZE=VISUALIZE_ENCODE,
        GROUND=GROUND,
        WRITE=WRITE,
        SPECS=SPECS,
    )
    (
        pattern_code_set,
        hash_to_code_dict,
        avg_color_set,
        ground,
        code_frequencies,
    ) = get_hash_to_code(pattern_set, ground, hash_frequency_dict, GROUND=GROUND)
    ###################################################
    print("EXTRACTING ADJACENCY...")
    adjacency_list = extract_adjacency(
        hash_to_code_dict, pattern_set, N, directions_list,VISUALIZE=VISUALIZE_ADJACENCY
    )
    #adjacency_list = extract_adjacency_from_image(
    #    input_img,hash_to_code_dict, pattern_set, N, VISUALIZE=VISUALIZE_ADJACENCY
    #)
    adjacency_matrices = build_adjacency_matrix(
        adjacency_list, pattern_code_set, WRITE=WRITE
    )
    ###################################################
    print("BUILDING OUTPUT MATRIX...")
    output_matrix = build_output_matrix(code_frequencies, output_w,output_h)
    output_matrix = (
        pad_ground(output_matrix, ground,pattern_code_set,code_frequencies,avg_color_set,adjacency_matrices) if GROUND else output_matrix
    )    
    ###################################################
    print("PROPAGATING...")
    output_matrix = propagate(
        output_matrix, avg_color_set, adjacency_matrices, code_frequencies,directions_list,SPECS = SPECS
    )        
    backtrack_queue,output_matrix,backtrack_no = prepare_backtrack(copy.deepcopy(output_matrix),MAX_BACKTRACK)
    while True:
        #===========================
        # OBSERVE
        #===========================
        done,contradiction, output_matrix = observe(
            output_matrix, pattern_code_set, hash_frequency_dict, code_frequencies
        )
        #===========================
        # BACKTRACK IF CONTRADICTION
        #===========================
        if(contradiction):
            print("Contradiction! Backtracking...step {}".format(backtrack_no))
            try:
                output_matrix = copy.deepcopy(backtrack_memory(backtrack_queue,backtrack_no))                            
                backtrack_no = min(backtrack_no+2,MAX_BACKTRACK)
            except AssertionError:
                output_matrix = backtrack_memory(backtrack_queue,len(backtrack_queue))
                print("no previous state to backtrack on")                
                backtrack_no = 1
        else:            
            backtrack_queue = update_queue(backtrack_queue, copy.deepcopy(output_matrix))
            backtrack_no = max(1,backtrack_no-1)
        #===========================
        # PROPAGATE
        #===========================
        output_matrix = propagate(
            output_matrix, avg_color_set, adjacency_matrices, code_frequencies,directions_list,SPECS=SPECS
        )
        #===========================
        # RENDER
        #===========================
        rendered = render(
            output_matrix, output_w,output_h, N, pattern_code_set, WRITE_VIDEO=WRITE_VIDEO
        )
        #===========================
        # DISPLAY AND WRITE(OPTIONAL)
        #===========================
        if WRITE_VIDEO:
            im_rgb = cv2.cvtColor(rendered.astype(np.uint8), cv2.COLOR_BGR2RGB)
            im_rgb = cv2.resize(im_rgb,(512,512))
            video_out.append(im_rgb)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        #if done: exit()
    final_constraints_satisfied = final_global_constr(output_matrix)
    final_constraints_satisfied = matrix_global_constr(output_matrix)
    print("Constraints satisfied:", final_constraints_satisfied)  #If false, we can observe why and re-run mannually
    cv2.destroyAllWindows()
    if(WRITE_VIDEO):imageio.mimsave(os.path.join("wfc_outputs",output_name+".gif"),video_out)



    

