
def initialise():
    global unique_constraint, unique_tiles_deleted, max_non_unique, unique_threshold, unique_pix

    # hyperparameters
    unique_constraint = False         # found unique tile
    unique_tiles_deleted = False
    max_non_unique = 2                # maximum number of unique pix a non unique-obj tile can hold
    unique_threshold = 1              # any tiles with more than this unique pixel will be deleted
    unique_pix = [59, 235, 255] 


