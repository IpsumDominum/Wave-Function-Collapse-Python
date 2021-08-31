import numpy as np
import cv2
import random
from collections import defaultdict
import os
output_dir= "outputs"
if(os.path.isdir(output_dir)):
    pass
else:
    os.makedir(output_dir)
"""
Sample over NxN crops of Input image, to create selection array
for output, each grid selects out of random one of the selection array membe
"""

def cv_img(img,resize=True,wait=True,id=str(random.random())):
    if(resize):
        img = cv2.resize(img,(512,512),interpolation=3)/255
    cv2.imshow(id,img)        
    #if(wait):
def write_img(img,out,resize=True,wait=True,id=str(random.random())):
    if(resize):
        img = cv2.resize(img,(512,512))
    out.write(img.astype(np.uint8))

def img_equal(img1,img2):
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            for k in range(img2.shape[2]):
                if(img1[i][j][k]!=img2[i][j][k]):
                    return False
    return True
def is_same_pattern(pattern1,pattern2):
    #rotate and compare
    if(img_equal(pattern1,pattern2)):
        return True
    if(img_equal(pattern1,np.fliplr(pattern2))):
        return True
    if(img_equal(pattern1,np.flipud(pattern2))):
        return True
    for i in range(4):
        img = np.rot90(pattern2)
        if(img_equal(pattern1,img)):
            return True
        if(img_equal(pattern1,np.fliplr(img))):
            return True
        if(img_equal(pattern1,np.flipud(img))):
            return True
    return False        
    
def hash_function(img,random_state=0):
    #HASH FUNCTION FROM Isaac Karth implementation (Modified)
    state = np.random.RandomState(random_state)
    u = img.reshape(
        np.array(np.prod(img.shape), dtype=np.int64)
    )
    v = state.randint(1 - (1 << 63), 1 << 63, np.prod(img.shape), dtype="int64")
    return np.inner(u, v).astype("int64")


def wfc_run(input_img,N=3,output_size=32,write_output=False,output_name="out_video.avi"):
    ROW = 0
    COL = 1

    if(write_output):
        frame_width = 512
        frame_height = 512
        out = cv2.VideoWriter(os.path.join(output_dir,output_name+".avi"),cv2.VideoWriter_fourcc('M','J','P','G'), 80, (frame_width,frame_height))
    else:
        out = None

    coe_matrix = np.ones((output_size,output_size))
    if(len(input_img.shape)>2):
        channels = input_img.shape[2]
    else:
        channels = 1
    input_shape_i = input_img.shape[0]
    input_shape_j = input_img.shape[1]

    

    cropped_list = []
    hash_to_idx_dict = {}
    hash_frequency_dict = defaultdict(lambda:0)
    PAD = 2*N                                               #TODO: pad could be causing problems
    input_padded = np.pad(
        input_img,
        ((PAD,PAD),(PAD,PAD),(0,0)),
        "wrap"
    )
    pad_row = input_padded.shape[0]
    pad_col = input_padded.shape[1]

    cropped_sets = np.zeros((*input_padded.shape[:2],N,N,channels))
    
    unique_patterns = []
    valid_neighbours = defaultdict(lambda:[[],[],[],[],[],[],[],[]])
    seen_patterns = []
    debug_output = np.zeros((N*3,N*3,3))
    directions_list = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]

    import math
    VISUALIZE = False
    print("ENCODING...")
    #ENCODE EVERYTHING
    for i in range(math.ceil((pad_row-N))):
        for j in range(math.ceil((pad_col-N))):
            crop_img = input_padded[i:i+N, j:j+N, :]
            cropped_sets[i][j][:] = crop_img
            hash_code = hash_function(crop_img)
            cropped_list.append(hash_code)
            hash_to_idx_dict[hash_code] = (i,j)
            hash_frequency_dict[hash_code] += 1

            if(VISUALIZE):
                show = input_padded.copy()
                cv2.rectangle(show,(j,i),((j+N),(i+N)),(0,255,0),1)
                cv_img(show)
                k = cv2.waitKey(0)
                if(k==ord('q')):
                    cv2.destroyAllWindows()
                    exit()

    # # non overlapping crop img sets
    # for i in range(math.ceil((input_col-N)//N)+1):
    #     for j in range(math.ceil((input_row-N)//N)+1):
    #         crop_img = input_padded[i*N:(i+1)*N,j*N:(j+1)*N,:]
    #         cropped_sets[i][j][:] = crop_img
    #         hash_code = hash_function(crop_img)
    #         cropped_list.append(hash_code)
    #         hash_to_idx_dict[hash_code] = (i,j)
    #         hash_frequency_dict[hash_code] += 1
    #         if(VISUALIZE):
    #             show = input_padded.copy()
    #             cv2.rectangle(show,(j*N,i*N),((j+1)*N,(i+1)*N),(0,255,0),1)
    #             cv_img(show)
    #             k = cv2.waitKey(0)
    #             if(k==ord('q')):
    #                 cv2.destroyAllWindows()
    #                 exit()



    print("EXTRACTING ADJACENCY...")
    VISUALIZE = False
    #EXTRACT ADJACENCY
    for i in range(math.ceil((input_shape_i-N)//N)+2):
        for j in range(math.ceil((input_shape_j-N)//N)+2):
            # crop_img = input_padded[(i)*N+PAD:(i+1)*N+PAD,(j)*N+PAD:(j+1)*N+PAD,:]
            crop_img = input_padded[(i)+PAD:(i+N)+PAD, (j)+PAD:(j+N)+PAD, :]
            hash_code = hash_function(crop_img)
            if(VISUALIZE):
                debug_output[N:N+N,N:N+N] = crop_img
                cv_img(crop_img,id="input_padded")
                print("====")
                print(i,j)
                print(i,i+N,j,j+N)
                print("====")
                show = input_padded.copy()
                cv2.rectangle(show,(j*N+PAD,i*N+PAD),((j+1)*N+PAD,(i+1)*N+PAD),(0,255,0),1)
                #cv2.rectangle(debug_output,(N,N),(N+N,N+N),(0,255,0),1)

            for index,directions in enumerate(directions_list):
                idx = (i+PAD+(directions[ROW])*N)%pad_row
                jdx = (j+PAD+(directions[COL])*N)%pad_col
                # idx = (i + PAD + (directions[ROW]) * N) % pad_row
                # jdx = (j + PAD + (directions[COL]) * N) % pad_col
                adjacent_img = input_padded[idx:idx+N,jdx:jdx+N,:]
                hash_code_adj = hash_function(adjacent_img)

                #For the direction in valid_neighbours,append the adjacent hashcode
                valid_neighbours[hash_code][index].append(hash_code_adj)
                if(VISUALIZE):
                    #print(idx,jdx)
                    #print(input_padded.shape)
                    #print(idx,idx+N,jdx,jdx+N)
                    debug_output[directions[0]*N:directions[0]*N+N,directions[1]*N:directions[1]*N+N] = adjacent_img
            if(VISUALIZE):
                cv_img(debug_output,id="debug")
                cv_img(show,id="show")
                cv_img(input_img,id="original")
                k = cv2.waitKey(0)
                if(k==ord('q')):
                    cv2.destroyAllWindows()
                    exit()

    # # code for non overlapping extraction
    # for i in range(math.ceil((input_shape_i-N)//N)+2):
    #     for j in range(math.ceil((input_shape_j-N)//N)+2):
    #         # crop_img = input_padded[(i)*N+PAD:(i+1)*N+PAD,(j)*N+PAD:(j+1)*N+PAD,:]
    #         crop_img = input_padded[(i)+PAD:(i+N)+PAD, (j)+PAD:(j+N)+PAD, :]
    #         hash_code = hash_function(crop_img)
    #         if(VISUALIZE):
    #             debug_output[N:N+N,N:N+N] = crop_img
    #             cv_img(crop_img,id="input_padded")
    #             print("====  ")
    #             print(i,j)
    #             print(i,i+N,j,j+N)
    #             print("====")
    #             show = input_padded.copy()
    #             cv2.rectangle(show,(j*N+PAD,i*N+PAD),((j+1)*N+PAD,(i+1)*N+PAD),(0,255,0),1)
    #             #cv2.rectangle(debug_output,(N,N),(N+N,N+N),(0,255,0),1)
    #
    #         for index,directions in enumerate(directions_list):
    #             # idx = (i*N+PAD+(directions[0]-1)*N)%pad_row
    #             # jdx = (j*N+PAD+(directions[1]-1)*N)%pad_col
    #             idx = (i + PAD + (directions[ROW] - 1) * N) % pad_row
    #             jdx = (j + PAD + (directions[COL] - 1) * N) % pad_col
    #             adjacent_img = input_padded[idx:idx+N,jdx:jdx+N,:]
    #             hash_code_adj = hash_function(adjacent_img)
    #
    #             #For the direction in valid_neighbours,append the adjacent hashcode
    #             valid_neighbours[hash_code][index].append(hash_code_adj)
    #             if(VISUALIZE):
    #                 #print(idx,jdx)
    #                 #print(input_padded.shape)
    #                 #print(idx,idx+N,jdx,jdx+N)
    #                 debug_output[directions[0]*N:directions[0]*N+N,directions[1]*N:directions[1]*N+N] = adjacent_img


    #Do some testing
    #One : Ensure that all patterns are unique
    TEST = False
    if(TEST):
        print("TESTING ALL PATTERNS ARE UNIQUE")
        for i in range(len(cropped_list)):
            for j in range(len(cropped_list)):
                if(cropped_list[i]==cropped_list[j]):
                    idx = hash_to_idx_dict[cropped_list[i]]
                    jdx = hash_to_idx_dict[cropped_list[j]]
                    assert(img_equal(cropped_sets[idx[0]][idx[1]],cropped_sets[jdx[0]][jdx[1]]))                
                else:
                    try:
                        assert(not img_equal(cropped_sets[idx[0]][idx[1]],cropped_sets[jdx[0]][jdx[1]]))
                    except AssertionError:
                        img_i = cropped_sets[idx]
                        img_j = cropped_sets[jdx]

    #Find all possible patterns
    #Build a table of patterns and their valid neighbours

    #cropped_list = cropped_sets.reshape((input_img.shape[0]*input_img.shape[1],*cropped_sets.shape[2:]))

    #Brute force check all seen patterns
    def get_pattern(pattern_code):
        idx = hash_to_idx_dict[pattern_code]
        return cropped_sets[idx[0]][idx[1]]

    VISUALIZE = False
    print("MAKING ADJACENCIES_UNIQUE")
    adjacency_visualization_grid = np.zeros((N*9,N*9,3))    
    for nei in valid_neighbours:
        #get some unique neighbours
        if(VISUALIZE==True):
            adjacency_vis_list = []
            adjacency_visualization_grid[N*3:N*6,N*3:N*6,:] = cv2.resize(get_pattern(nei),(N*3,N*3),interpolation=3)            
        for i,direction in enumerate(directions_list):
            valid_neighbours[nei][i] = list(np.unique(valid_neighbours[nei][i]))
            if(VISUALIZE==True):
                #For all the neighbours, put them in adjacency visualization grid,
                adjacency_vis_list.append([])
                for j in valid_neighbours[nei][i]:
                    adjacency_vis_list[i].append(j)
        if(VISUALIZE):
            while(True):            
                for i,direction in enumerate(directions_list):
                    j = np.random.choice(adjacency_vis_list[i])
                    adjacency_visualization_grid[direction[0]*N*3:direction[0]*N*3+N*3,direction[1]*N*3:direction[1]*N*3+N*3,:] = cv2.resize(get_pattern(j),(N*3,N*3),interpolation=3)
                cv2.rectangle(adjacency_visualization_grid,(N*3,N*3),(N*6-1,N*6-1),(0,255,0),1)
                cv_img(adjacency_visualization_grid,id="main")
                k = cv2.waitKey(1)
                if(k==ord('n')):
                    break
                elif(k==ord('q')):
                    cv2.destroyAllWindows()
                    exit()
        cv2.destroyAllWindows()
    

    def get_avg_color(hash_list):
        avg_color = np.zeros((3))
        for key in hash_list:
            idx = hash_to_idx_dict[key]
            avg_color = (avg_color + np.mean(cropped_sets[idx[0]][idx[1]].reshape(-1,3),axis=0))/2
        return avg_color    

    def render_img(output,output_img,out):
        for i in range(output_size):
            for j in range(output_size):
                if(output[i*output_size+j][1]==-1):
                    output_img[i*N:(i+1)*N,j*N:(j+1)*N,:] = output[i*output_size+j][3]
                else:
                    pattern_code = output[i*output_size+j][1]
                    idx = hash_to_idx_dict[pattern_code]
                    #cv_img(cropped_sets[idx[0]][idx[1]])
                    output_img[i*N:(i+1)*N,j*N:(j+1)*N,:] = cropped_sets[idx[0]][idx[1]]
        if(write_output):
            write_img(output_img,out)
        cv_img(output_img)

    #For each block we have entropy + pattern
    avg_color = get_avg_color(valid_neighbours.keys())
    output = [[
        len(valid_neighbours)+random.random(),
        -1,
        list(valid_neighbours.keys()),
        avg_color] for _ in range(output_size*output_size)]
    output_img = np.ones((output_size*N,output_size*N,3))

    #Begin by setting every entropy as all possible states
    #output is = [entropy,chosen_img_code,valid_patterns]
    LARGE_NUM =  10000000000
    def observe(output):
        #find lowest entropy
        if(output[np.argmin(np.array(output)[:,0])][0]==LARGE_NUM):
            return -1
        else:
            return np.argmin(np.array(output)[:,0])
    def get_probs(code_list):
        prob_list = np.zeros((len(code_list)))
        total_freq = 0
        for idx,item in enumerate(code_list):
            freq = hash_frequency_dict[item]
            prob_list[idx] = 1/freq
            total_freq += 1/freq
        return prob_list / total_freq

    
    def collapse_wave_function(output,index):
        #Collpase the current value of the next observed obj
        if(len(output[index][2])==0):
            output[index][0] = LARGE_NUM
            output[index][3] = np.array([255,0,0])
            return output
        output[index][1] = np.random.choice(output[index][2],p=get_probs(output[index][2]))
        output[index][0] = LARGE_NUM
        #propagate the constraint wave
        #For all near states, propagate.
        for i,direction in enumerate(directions_list):
            near_idx = (index+(direction[0]-1)*output_size+(direction[1]-1))%len(output)
            if(output[near_idx][0]!=LARGE_NUM):
                #Limit the amount of possible states to be chosen based on output selected
                new_valid = []
                for item in valid_neighbours[output[index][1]][i]:
                    if(item in output[near_idx][2]):
                        new_valid.append(item)
                #If there is only one choice, collapse immediately.
                output[near_idx][2] = new_valid
                #Set entropy to length of current possible states
                output[near_idx][0] = len(new_valid)+random.random()
                #Set average color
                output[near_idx][3] = get_avg_color(output[near_idx][2])
        return output
    #from queue import Queue
    #queue = Queue()
    #queue.put(observe(output))
    print("BEGINNING PROPAGATION")
    while True:
        index = observe(output)
        if(index==-1):
            done = True
        else:
            done = False
        output = collapse_wave_function(output,index)        
        render_img(output,output_img,out)
        
        k = cv2.waitKey(1)
        if(k==ord('q') or done==True):            
            break
    if(write_output):
        out.release()        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
        
                
            
            
