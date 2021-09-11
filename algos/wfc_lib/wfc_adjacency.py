import numpy as np
import cv2
import random
import math
from collections import defaultdict
from algos.wfc_lib.wfc_global_constraints import adj_global_constr
import os
from tqdm import tqdm
from algos.wfc_lib.wfc_utils import (hash_function,
                is_same_pattern,
                img_equal,
                cv_write,
                cv_img,cv_wait,
                get_hash_to_code)
def visualize_adjacency(image,image2,dy,dx,N,match,sliced2,sliced):
    padded = np.pad(image,((1,1),(1,1),(0,0)),"constant",constant_values=(30))
    padded2 = np.pad(image2,((1,1),(1,1),(0,0)),"constant",constant_values=(30))
    show = cv2.resize(padded.copy(),(512,512),interpolation=3)/255
    cv2.rectangle(show,((1+dx)*512//(N+2),(1+dy)*512//(N+2)),( (1+N+dx)*512//(N+2),(1+N+dy)*512//(N+2)),(0,255,0),10)
    #Slice opposite side of image
    show2 = cv2.resize(padded2.copy(),(512,512),interpolation=3)/255
    cv2.rectangle(show2,((1-dx)*512//(N+2),(1-dy)*512//(N+2)),( (1+N-dx)*512//(N+2),(1+N-dy)*512//(N+2)),(0,255,0),10)
    overlay = padded.copy()
    if(sliced2.shape==(3,2,3)):
        overlay[1+dy:1+dy+N,1+dx:1+dx+N-1] = sliced2
    elif(sliced2.shape==(2,2,3)):
        overlay[1+dy:1+dy+N-1,1+dx:1+dx+N-1] = sliced2
    elif(sliced2.shape==(2,3,3)):
        overlay[1+dy:1+dy+N-1,1+dx:1+dx+N-1] = sliced2
    overlay = cv2.resize(overlay,(512,512),interpolation=3)/255
    if(match==True):
        color = (255,0,0)
    else:
        color = (0,0,255)
    cv2.rectangle(overlay,((1)*512//(N+2),(1)*512//(N+2)),( (1+N)*512//(N+2),(1+N)*512//(N+2)),color,10)                    
    cv2.imshow("overlay",overlay)
    cv2.imshow('show',show)
    cv2.imshow('show2',show2)
    cv_img(image,id="image")
    cv_img(image2,id="image2")
    cv_img(sliced,id="sliced")
    cv_img(sliced2,id="sliced2")
    k = cv2.waitKey(0)
    if(k==ord('q')):
        cv2.destroyAllWindows()
        exit()
def extract_adjacency(hash_to_code_dict,pattern_set,N,VISUALIZE=False):
    adjacency_list = defaultdict(lambda:[])
    directions_list = [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1),(-1,1),(1,1)]
    #directions_list = [(0,-1),(0,1),(-1,0),(1,0)]
    #EXTRACT ADJACENCY
    #For all of the tiles, create a list of toleratable overlap adjacencies.
    for item in tqdm(pattern_set.keys()):
        image = pattern_set[item]
        #For every other item, see if the overlap is OK
        for item2 in pattern_set.keys():
            image2 = pattern_set[item2]
            for direction in directions_list:
                #Use slicing to move pad up
                dy,dx = direction
                sliced = image[max(dy,0):N+min(dy,0):,max(dx,0):N+min(dx,0)]
                sliced2 = image2[-min(dy,0):N-max(dy,0),-min(dx,0):N-max(dx,0)]
                match = False
                global_constr_match = adj_global_constr(sliced, sliced2)
                if(img_equal(sliced,sliced2) and global_constr_match):
                    match = True
                    adjacency_list[direction].append((hash_to_code_dict[item],hash_to_code_dict[item2]))
                #Slice the overlapping region of the two images to compare, and check if the overlap is the same.
                if(VISUALIZE):
                    visualize_adjacency(image,image2,dy,dx,N,match,sliced2,sliced)
    return adjacency_list
def write_adjacency_visualize(adjacency_list,pattern_code_set):
    if(os.path.isdir(os.path.join("vis","adjacency"))):
        for item in os.listdir(os.path.join("vis","adjacency")):
            os.remove(os.path.join("vis","adjacency",item))
    else:
        os.makedirs(os.path.join("vis","adjacency"))

    for d in adjacency_list:
        for i, adj in enumerate(adjacency_list[d]):
            dy,dx = d
            img1 = pattern_code_set[adjacency_list[d][i][0]]
            img2 = pattern_code_set[adjacency_list[d][i][1]]
            padded = np.pad(img1,((1,1),(1,1),(0,0)),mode="constant",constant_values=50)
            N = img1.shape[0]
            padded[1+dy:1+dy+N,1+dx:1+dx+N] = img2
            cv2.imwrite(os.path.join("vis","adjacency",str(d[0])+str(d[1])+"_"+str(adj[0])+"_"+str(adj[1])+".png"),cv2.resize(padded,(512,512),interpolation=3))
def build_adjacency_matrix(adjacency_list,pattern_code_set,WRITE=False):
    adjacency_matrices = {}
    num_patterns = len(pattern_code_set.keys())
    for d in adjacency_list:
        m = np.zeros((num_patterns, num_patterns), dtype=bool)
        for i, adj in enumerate(adjacency_list[d]):
            m[adj[0], adj[1]] = True
        #Optional to use sparse matrix.
        #adjacency_matrices[d] = sparse.csr_matrix(m)
        adjacency_matrices[d] = m    
    if(WRITE):
        write_adjacency_visualize(adjacency_list,pattern_code_set)     
    return adjacency_matrices
