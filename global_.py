
def initialise():
    global unique_constraint, unique_tiles_deleted, max_non_unique, unique_threshold, unique_pix, deleted_tiles

    # hyperparameters
    max_non_unique = 2                  # maximum number of unique pix a non unique-obj tile can hold (UNIQUE_ID_THRESHOLD)
    unique_threshold = 2                # any tiles with more than this unique pixel will be deleted (UNIQUE_DEL_THRESHOLD)
    unique_pix = [59, 235, 255]
    # red car: [28, 28, 183], moon yellow: [59, 235, 255]

    # parameters
    unique_constraint = False           # found unique tile
    unique_tiles_deleted = False
    deleted_tiles = []
