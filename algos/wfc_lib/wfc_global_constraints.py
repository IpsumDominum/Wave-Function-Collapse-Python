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

# enables global constraint where an object can only generate within a certain height
def apply_absolute_global_constraints(output_matrix,hash_to_code_dict):
    leaves = [-1594224019795209414,121603043545895334,4018928990512404903,3795647373440428352,-5560531983198038104,4430478191549367938,1477480221035906530,-2005773220832172449,4936574010010790946,2419565912035275032,3404961282477216824,-3559628511854731877,-3148079310817768842,-4595340896671238436,2686730047883978862,-5509867884550305410,-868028714481886793,-9214970800262945380,-1504893975048942610,454161419975752569,1544423957898382469,8264384078907809415,8141510735523976336,7809437305270912798,-3148079310817768842,-4595340896671238436,-2005773220832172449,-7936826850135914219,-6168570062971257755,2419565912035275032,7161813355791837121,-7936826850135914219]
    output_matrix["valid_states"][hash_to_code_dict[4011487003521511759],:,:] = False
    output_matrix["valid_states"][hash_to_code_dict[4011487003521511759],10:11,:] = True

    for l in leaves:
        output_matrix["valid_states"][hash_to_code_dict[l],:,:] = False
        output_matrix["valid_states"][hash_to_code_dict[l],8:-1,:] = True
    return output_matrix

def apply_relative_distance_constraints(output_matrix,hash_to_code_dict,array_index,chosen_idx):
    relative_distance_constraints = [
            {
                "patternSource": [-7372931810443495452],
                "patternDestination": [4011487003521511759],
                "distanceX": 0,
                "distanceY": -20,
            },
             {
                "patternSource": [4011487003521511759],
                "patternDestination": [-7372931810443495452],
                "distanceX": 0,
                "distanceY": -1,
            },
        ]
    red_pixels = [6792183754579830353,4316627595727781747,3744677450247838733,3775301369191246084]
    white_black = []
    for hash in hash_to_code_dict.keys():
        if(hash not in red_pixels):
            white_black.append(hash)
    #for red in red_pixels:
    #        if(hash_to_code_dict[red]==chosen_idx):
                
    relative_distance_constraints = [
        {
            "patternSource": red_pixels,
            "patternDestination":white_black,
            "distanceX" :2,
            "distanceY":0,
        },
        {
            "patternSource": red_pixels,
            "patternDestination":white_black,
            "distanceX" :-2,
            "distanceY":0,
        },
        {
            "patternSource": red_pixels,
            "patternDestination":white_black,
            "distanceX" :0,
            "distanceY":-2,
        },
        {
            "patternSource": red_pixels,
            "patternDestination":white_black,
            "distanceX" :0,
            "distanceY":2,
        }
    ]
    """
        {
            "patternSource":[-1594224019795209414,121603043545895334,4018928990512404903,3795647373440428352,-5560531983198038104,4430478191549367938,1477480221035906530,-2005773220832172449,4936574010010790946,2419565912035275032,3404961282477216824,-3559628511854731877,-3148079310817768842,-4595340896671238436,2686730047883978862,-5509867884550305410,-868028714481886793,-9214970800262945380,-1504893975048942610,454161419975752569,1544423957898382469,8264384078907809415,8141510735523976336,7809437305270912798,-3148079310817768842,-4595340896671238436,-2005773220832172449,-7936826850135914219,-6168570062971257755,2419565912035275032,7161813355791837121,-7936826850135914219],
            "patternDestination":[-1594224019795209414,121603043545895334,4018928990512404903,3795647373440428352,-5560531983198038104,4430478191549367938,1477480221035906530,-2005773220832172449,4936574010010790946,2419565912035275032,3404961282477216824,-3559628511854731877,-3148079310817768842,-4595340896671238436,2686730047883978862,-5509867884550305410,-868028714481886793,-9214970800262945380,-1504893975048942610,454161419975752569,1544423957898382469,8264384078907809415,8141510735523976336,7809437305270912798,-3148079310817768842,-4595340896671238436,-2005773220832172449,-7936826850135914219,-6168570062971257755,2419565912035275032,7161813355791837121,-7936826850135914219],
            "distanceX":10,
            "distanceY":0,
        }
    """
    code_to_hash_dict = {}
    for key in hash_to_code_dict:
        code_to_hash_dict[hash_to_code_dict[key]] = key

    for constraint in relative_distance_constraints:
        if code_to_hash_dict[chosen_idx] in constraint["patternSource"]:
            if(constraint["distanceY"]==-1):
                added_Y = output_matrix["valid_states"].shape[1]-1
            else:
                added_Y = (array_index[0] + constraint["distanceY"]) % output_matrix["valid_states"].shape[1]
                
            if(constraint["distanceX"]==-1):
                added_X = output_matrix["valid_states"].shape[2]-1
            else:
                added_X = (array_index[1] + constraint["distanceX"]) % output_matrix["valid_states"].shape[2]
            output_matrix["valid_states"][
                :,
                added_Y,
                added_X,
            ] = False
            for i in range(len(constraint["patternDestination"])):
                output_matrix["valid_states"][
                    hash_to_code_dict[constraint["patternDestination"][i]],
                    added_Y,
                    added_X,
                ] = True
                #output_matrix["chosen_states"][added_Y,added_X] = hash_to_code_dict[constraint["patternDestination"][i]]

    """
        leaves = [-1594224019795209414,121603043545895334,4018928990512404903,3795647373440428352,-5560531983198038104,4430478191549367938,1477480221035906530,-2005773220832172449,4936574010010790946,2419565912035275032,3404961282477216824,-3559628511854731877,-3148079310817768842,-4595340896671238436,2686730047883978862,-5509867884550305410,-868028714481886793,-9214970800262945380,-1504893975048942610,454161419975752569,1544423957898382469,8264384078907809415,8141510735523976336,7809437305270912798,-3148079310817768842,-4595340896671238436,-2005773220832172449,-7936826850135914219,-6168570062971257755,2419565912035275032,7161813355791837121,-7936826850135914219]
        if(hash_to_code_dict[4011487003521511759] == chosen_idx):
            for l in leaves:
                output_matrix["valid_states"][hash_to_code_dict[l],:,:] = False
                output_matrix["valid_states"][hash_to_code_dict[l],array_index[0]:-1,:] = True
        """
        #others = [1419649332853391511,-6193635647972180021,-6943975255523720358,-576014106182276452,8178981485339150217,-2642761215386554919,6759332152485758706,0,7285085998938077148,7241020459408101240,2506275869316181273,7307995805344226847,5245357020372433277,6057150301901806375,8817563606098293930,-8073969474061300547,-1995663439035667963,0,-7625001372983846480,1419649332853391511,-6193635647972180021,-6943975255523720358,-576014106182276452,-2642761215386554919,8817563606098293930,-788883475123223399,5245357020372433277,-5104507772869668093,2506275869316181273,-3527964626179475403,-8073969474061300547,-3527964626179475403,-4537312772273672559,-7817034565590377692,-5163254815500733306,6195958123835565220,-8971064156557281263,-788883475123223399,-8971064156557281263,-3999842533663079169,5402250834209234456,-1357081318276524250,-8982082691260370730,62568014576867261,-4537312772273672559,62568014576867261,-8073969474061300547,-1995663439035667963,0,7610783642185849366,1576543146690192690,-6520336133777257556,764749865160819592,-6564401673307233464,-3955776994133103261,1576543146690192690,-6520336133777257556,764749865160819592,-6564401673307233464,-419120292345475273,4315624297746444694,-8096879280467450246,0,4074126940398221378,7397914273244902419,-1357081318276524250,-8982082691260370730,7687569387560713741,1328123893811013078,7397914273244902419,-1357081318276524250,-8982082691260370730,7687569387560713741,5402250834209234456,-5431208258674745628,-8754995591521426669,0,0,0,0,-3550874432585625102,-6078306035025632584,-1995663439035667963,7285085998938077148,7241020459408101240,2506275869316181273,7307995805344226847,-2039728978565643871,6101215841431782283,7285085998938077148,7241020459408101240,-5104507772869668093,-5104507772869668093,-5104507772869668093,2506275869316181273,22909806406149699,-1995663439035667963,-1567851071082040105,7520865174285173794,-6193635647972180021,-6943975255523720358,7048987266801570028,6759332152485758706,-7625001372983846480,1419649332853391511,8178981485339150217,8178981485339150217,8178981485339150217,-6193635647972180021,7966112116398203270,5245357020372433277,6195958123835565220,-8971064156557281263,-788883475123223399,5245357020372433277,2506275869316181273,-3527964626179475403,-4537312772273672559,-7817034565590377692,-8982082691260370730,62568014576867261,-4537312772273672559,-7817034565590377692,6195958123835565220,-8971064156557281263,-3999842533663079169,5402250834209234456,7610783642185849366,8861629145628269838,-6564401673307233464,-419120292345475273,8389751238144666072,-698965007222547827,-8982082691260370730,62568014576867261,-8073969474061300547,-1995663439035667963,4074126940398221378,-227087099738944061,7687569387560713741,5402250834209234456,-5431208258674745628,-8754995591521426669,7610783642185849366,1576543146690192690,-6520336133777257556,-6520336133777257556,764749865160819592,-6564401673307233464,-8029903934531324639,-8096879280467450246]
        """
        filter_block = []
        SIZE_LIMIT = 1
        for i in range(SIZE_LIMIT-1,SIZE_LIMIT-1):
            for j in range(SIZE_LIMIT-1,SIZE_LIMIT-1):
                filter_block.append((i,j))
        if(hash_to_code_dict[6821900167062625967] == chosen_idx):
            for i in range(-SIZE_LIMIT+1,SIZE_LIMIT):
                for j in range(-SIZE_LIMIT+1,SIZE_LIMIT):
                    idx = array_index[0] + i
                    idy = array_index[1] + j
                    if((i,j) not in filter_block):
                        output_matrix["valid_states"][chosen_idx,idx,idy] = False
    """
    return output_matrix      
