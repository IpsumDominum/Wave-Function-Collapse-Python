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
        img = cv2.resize(img,(512,512))/255
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
    #HASH FUNCTION FROM  Isaac Karth implementation (Modified)
    state = np.random.RandomState(random_state)
    u = img.reshape(
        np.array(np.prod(img.shape), dtype=np.int64)
    )
    v = state.randint(1 - (1 << 63), 1 << 63, np.prod(img.shape), dtype="int64")
    return np.inner(u, v).astype("int64")


def wfc_run(input_img,N=3,output_size=32,write_output=False,output_name="out_video.avi"):
    if(write_output):
        frame_width = 512
        frame_height = 512
        out = cv2.VideoWriter(os.path.join(output_dir,output_name+".avi"),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    else:
        out = None

    coe_matrix = np.ones((output_size,output_size))
    if(len(input_img.shape)>2):
        channels = input_img.shape[2]
    else:
        channels = 1
    cropped_sets = np.zeros((*input_img.shape[:2],N,N,channels))
    cropped_list = []
    hash_to_idx_dict = {}
    hash_frequency_dict = defaultdict(lambda:0)
    input_padded = np.pad(
        input_img,
        ((N,N),(N,N),(0,0)),
        "wrap"
    )

    #Get all subsets of these.
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            cropped_sets[i][j][:] = input_padded[i:i+N,j:j+N,:]
            hash_code = hash_function(input_padded[i:i+N,j:j+N])
            cropped_list.append(hash_code)
            hash_to_idx_dict[hash_code] = (i,j)
            hash_frequency_dict[hash_code] += 1

    #Do some testing
    #One : Ensure that all patterns are unique
    TEST = False
    if(TEST):
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
    
    unique_patterns = []
    valid_neighbours = defaultdict(lambda:[[],[],[],[],[],[],[],[]])
    seen_patterns = []

    for j in range(len(cropped_list)):
        #if(is_same_pattern(cropped_list[i],cropped_list[j])):
        #Add j's neighbours to valid i neighbours
        row_dx = j//cropped_sets.shape[1]
        column_dx = j%cropped_sets.shape[1]
        j_neighbours = {}
        
        #up-0 down-1 left-2 right-3
        #up
        up_idx = ((row_dx-N)%cropped_sets.shape[0])*cropped_sets.shape[1] +column_dx
        j_neighbours[0] = cropped_list[up_idx]

        #down
        down_idx = ((row_dx+N)%cropped_sets.shape[0])*cropped_sets.shape[1] +column_dx
        j_neighbours[1] = cropped_list[down_idx]

        #left
        left_idx = row_dx*cropped_sets.shape[1] + ((column_dx-N)%cropped_sets.shape[1])
        j_neighbours[2] = cropped_list[left_idx]

        #right
        right_idx = row_dx*cropped_sets.shape[1] +((column_dx+N)%cropped_sets.shape[1])
        j_neighbours[3] = cropped_list[right_idx]

        #upleft
        upleft_idx = ((row_dx-N)%cropped_sets.shape[0])*cropped_sets.shape[1] +((column_dx-N)%cropped_sets.shape[1])
        j_neighbours[4] = cropped_list[upleft_idx]

        #upright
        upright_idx = ((row_dx-N)%cropped_sets.shape[0])*cropped_sets.shape[1] +((column_dx+N)%cropped_sets.shape[1])
        j_neighbours[5] = cropped_list[upright_idx]

        #downleft
        downleft_idx = ((row_dx+N)%cropped_sets.shape[0])*cropped_sets.shape[1] +((column_dx-N)%cropped_sets.shape[1])
        j_neighbours[6] = cropped_list[downleft_idx]

        #downright
        downright_idx = ((row_dx+N)%cropped_sets.shape[0])*cropped_sets.shape[1] +((column_dx+N)%cropped_sets.shape[1])
        j_neighbours[7] = cropped_list[downright_idx]

        idx = hash_to_idx_dict[cropped_list[j]]

        #cv_img(cropped_sets[idx[0],idx[1],:],id="main")
        for i in range(8):
            #idx = hash_to_idx_dict[j_neighbours[i]]
            #cv_img(cropped_sets[idx[0],idx[1],:],id=str(i))
            valid_neighbours[cropped_list[j]][i].append(j_neighbours[i])
        """
        k = cv2.waitKey(0)
        if(k==ord('q')):
            cv2.destroyAllWindows()
            exit()
        """
    """
    cv2.destroyAllWindows()
    """

    """
    for nei in valid_neighbours:
        #get some unique neighbours
        for i in range(4):
            valid_neighbours[nei] = np.unique(valid_neighbours[nei])
        #for i in valid_neighbours[nei]:
    """
    print("pattern extraction complete")

    def get_pattern(pattern_code):
        idx = hash_to_idx_dict[pattern_code]
        return cropped_sets[idx[0]][idx[1]]

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
    # Initialising output
    avg_color = get_avg_color(hash_to_idx_dict.keys())
    output = [[len(hash_to_idx_dict.keys())+random.random()/100,-1,list(hash_to_idx_dict.keys()),avg_color] for _ in range(output_size*output_size)]
    output_img = np.ones((output_size*N,output_size*N,3))

    #Begin by setting every entropy as all possible states
    #output is = [entropy,chosen_img_code,valid_patterns]
    def observe(output):
        #find lowest entropy
        if(output[np.argmin(np.array(output)[:,0])][0]==10000000):
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

    def collapse_wave_function(output,index,queue):
        #Go to the index, choose a valid pattern
        #Collapse the possible states of neighbours
        #Choose a pattern based on frequency
        #If there is no valid pattern to choose from, break and say oh no
        #print(output[index][2])
        if(len(output[index][2])==0):
            print("no more available output for at index", index)
            output[index][2] = list(hash_to_idx_dict.keys())
        probability_distribution = get_probs(output[index][2])
        output[index][1] = np.random.choice(output[index][2],p=probability_distribution)
        #output[index][1] = np.random.choice(output[index][2])

        #Set entropy to 0
        output[index][0] = 10000000
        
        #Collapse the subsequent wave functions
        #up down left right
        #up

        up = (index-output_size)%(output_size*output_size)
        row_dx = up // output_size
        column_dx = up % output_size
        up = row_dx*output_size + column_dx
        #down
        down = (index+output_size)%(output_size*output_size)
        row_dx = down // output_size
        column_dx = down % output_size
        down = row_dx*output_size + column_dx
        #left
        left = (index-1)%(output_size*output_size)
        row_dx = left // output_size
        column_dx = left % output_size
        left = row_dx*output_size + column_dx
        #right
        right = (index+1)%(output_size*output_size)
        row_dx = right // output_size
        column_dx = right % output_size
        right = row_dx*output_size + column_dx
        #upleft
        upleft = (index+output_size-1)%(output_size*output_size)
        row_dx = upleft // output_size
        column_dx = upleft % output_size
        upleft = row_dx*output_size + column_dx
        #upright
        upright = (index+output_size+1)%(output_size*output_size)
        row_dx = upright // output_size
        column_dx = upright % output_size
        upright = row_dx*output_size + column_dx
        #downleft
        downleft = (index-output_size-1)%(output_size*output_size)
        row_dx = downleft // output_size
        column_dx = downleft % output_size
        downleft = row_dx*output_size + column_dx
        #downright
        downright = (index-output_size+1)%(output_size*output_size)
        row_dx = downright // output_size
        column_dx = right % output_size
        downright = row_dx*output_size + column_dx
        
        #For each, update their valid patterns to only include the ones
        #available in valid neighbours for the chosen index
        for i,direction in enumerate([up,down,left,right,upleft,upright,downleft,downright]):
            #if the direction is already set, we don't have to do anything
            if(output[direction][0]==10000000):
                continue
            #output[index][1] is chosen pattern
            #i is the direction 
            #remember up-0 down-1 left-2 right-3 upleft-4 upright-5 downleft-6 downright-7
            new_valid = []
            check = valid_neighbours[output[index][1]][i]
            for item in list(hash_to_idx_dict.keys()):
                if(item in check and item in output[index][2]):
                    new_valid.append(item)
            """
            for iaa,valid in enumerate(new_valid):
                idx = hash_to_idx_dict[valid]
                cv_img(cropped_sets[idx],id=str(iaa))
            cv_img(cropped_sets[hash_to_idx_dict[output[index][1]]],id="hi")
            print(i)
            cv2.waitKey(0)
            """
            output[direction][2] = new_valid
            output[direction][0] = len(new_valid) - random.random()/100
            #Update avg_color
            output[direction][3] = get_avg_color(output[direction][2])
            #queue.put(direction)
        return output

    from queue import Queue
    queue = Queue()
    #queue.put(observe(output))
    while True:
        index = observe(output)
        if(index==-1):
            done = True
        else:
            done = False
        output = collapse_wave_function(output,index,queue)        
        render_img(output,output_img,out)
        k = cv2.waitKey(1)
        if(k==ord('q') or done==True):
            break
    if(write_output):
        out.release()        
    cv2.destroyAllWindows()
        
        
                
            
            
