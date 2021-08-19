import numpy as np
import cv2
import random
import math
from collections import defaultdict
import os
import imageio
from algos.wfc_lib.wfc_utils import (hash_function,
                is_same_pattern,
                img_equal,
                cv_write,
                cv_img,cv_wait,
                get_hash_to_code,
                prepare_write_outputs)
from algos.wfc_lib.wfc_encode import (get_encoded_patterns)
from algos.wfc_lib.wfc_adjacency import (extract_adjacency,build_adjacency_matrix)
from algos.wfc_lib.wfc_output import (build_output_matrix,observe,propagate,render,pad_ground)
"""
Sample over NxN crops of Input image, to create selection array
for output, each grid selects out of random one of the selection array membe
"""
def wfc_overlap_run(input_img,N=3,output_size=32,output_name="out_video.avi",GROUND=False,WRITE=False,VISUALIZE_ENCODE=False,VISUALIZE_ADJACENCY=False,WRITE_VIDEO=False):
    ###################################################
    print('RUNNING')
    video_out = []
    ###################################################
    print("ENCODING...")
    pattern_set,hash_frequency_dict,ground = get_encoded_patterns(input_img,N,VISUALIZE=VISUALIZE_ENCODE,GROUND=GROUND,WRITE=WRITE)
    pattern_code_set,hash_to_code_dict,avg_color_set,ground,code_frequencies = get_hash_to_code(pattern_set,ground,hash_frequency_dict,GROUND=GROUND)
    ###################################################
    print("EXTRACTING ADJACENCY...")
    adjacency_list = extract_adjacency(hash_to_code_dict,pattern_set,N,VISUALIZE=VISUALIZE_ADJACENCY)
    adjacency_matrices = build_adjacency_matrix(adjacency_list,pattern_code_set,WRITE=WRITE)    
    ###################################################
    print("BUILDING OUTPUT MATRIX...")
    output_matrix = build_output_matrix(len(pattern_set.keys()),output_size)
    output_matrix = pad_ground(output_matrix,ground=ground) if GROUND else output_matrix
    ###################################################
    print("PROPAGATING...")
    
    output_matrix = propagate(output_matrix,avg_color_set,adjacency_matrices,code_frequencies)
    while True:
        done,output_matrix = observe(output_matrix,pattern_code_set,hash_frequency_dict,code_frequencies)
        output_matrix = propagate(output_matrix,avg_color_set,adjacency_matrices,code_frequencies)
        rendered = render(output_matrix,output_size,N,pattern_code_set,WRITE_VIDEO=WRITE_VIDEO)
        if(WRITE_VIDEO):video_out.append(rendered.astype(np.uint8))
        k = cv2.waitKey(1)
        if(k==ord('q')):
            break
        #if done: exit()
    cv2.destroyAllWindows()
    if(WRITE_VIDEO):imageio.mimsave(os.path.join("wfc_outputs",output_name+".gif"),video_out)
    
        
    

