# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import convolve

# Read source and mask (if exists) for a given id
def Read(id, path = ""):
    source = plt.imread(path + "image_" + id + ".jpg") / 255
    maskPath = path + "mask_" + id + ".jpg"
    
    if os.path.isfile(maskPath):
        mask = plt.imread(maskPath)
        assert(mask.shape == source.shape), 'size of mask and image does not match'
        mask = (mask > 128)[:, :, 0].astype(int)
    else:
        mask = np.zeros_like(source)[:, :, 0].astype(int)

    return source, mask

def generate_energy_map(input):
    didx = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    didy = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    didx = np.stack([didx] * 3, axis=2)
    didy = np.stack([didy] * 3, axis=2)

    energy_map = np.absolute(convolve(input, didx)) + np.absolute(convolve(input, didy))

    # Add RGB channel
    return energy_map.sum(axis=2)

# Reduce width of the entire image through the iteration
def reduce_width(input, scale, mask):
    h, w = input.shape[:2]
    new_w = int(scale * w)
    for _ in range(w - new_w):
        input, boolmask = remove_seam(input, mask)
        if mask is not None:
            mask = remove_seam_mask(mask, boolmask)
    size = (new_w, h)
    return input, size

# Reduce height of the entire image through the iteration
def reduce_height(input, scale, mask):
    input = np.rot90(input, 1, (0, 1))
    if mask is not None:
        mask = np.rot90(mask, 1, (0, 1))
    input, size_90 = reduce_width(input, scale, mask)
    input = np.rot90(input, 3, (0, 1))
    size = (size_90[1], size_90[0])
    return input, size

def remove_seam_mask(input, boolmask):
    h, w = input.shape[:2]
    return input[boolmask].reshape((h, w - 1))

# Remove the seam with the least energy
def remove_seam(input, mask):
    h, w = input.shape[:2]

    M, backtrack = get_minimum_seam(input, mask)
    boolmask = np.ones((h, w), dtype=np.bool_)
    j = np.argmin(M[-1])
    for i in reversed(range(h)):
        boolmask[i, j] = False
        j = backtrack[i, j]
    boolmask3 = np.stack([boolmask] * 3, axis=2)
    return input[boolmask3].reshape((h, w - 1, 3)), boolmask

# Get minimum seam using the dynamic programming
def get_minimum_seam(input, mask):
    h, w = input.shape[:2]
    energy_map = generate_energy_map(input)

    M = energy_map
    if mask is not None:
        M[np.where(mask > 0)] = 100000
    backtrack = np.zeros_like(M, dtype=np.int32)

    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                min_index = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = min_index + j
                min_energy = M[i-1, min_index + j]
            else:
                min_index = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = min_index + j - 1
                min_energy = M[i - 1, min_index + j - 1]

            M[i, j] += min_energy

    boolmask = np.ones((h, w), dtype=np.bool_)
    j = np.argmin(M[-1])
    for i in reversed(range(h)):
        boolmask[i, j] = False
        j = backtrack[i, j]

    boolmask = np.stack([boolmask] * 3, axis=2)
    input = input[boolmask].reshape((h, w - 1, 3))

    return M, backtrack

# Main Seam carving algorithm
def SeamCarve(input, widthFac, heightFac, mask):
    
    # Main seam carving function. This is done in three main parts: 1)
    # computing the energy function, 2) finding optimal seam, and 3) removing
    # the seam. The three parts are repeated until the desired size is reached.

    assert (widthFac == 1 or heightFac == 1), 'Changing both width and height is not supported!'
    assert (widthFac <= 1 and heightFac <= 1), 'Increasing the size is not supported!'

    if widthFac < 1:
        output, size = reduce_width(input, widthFac, mask)
    elif heightFac < 1:
        output, size = reduce_height(input, heightFac, mask)
    return output, size


# Setting up the input output paths
inputDir = '../Images/'
outputDir = '../Results/'

widthFac = 1; # To reduce the width, set this parameter to a value less than 1
heightFac = 0.5;  # To reduce the height, set this parameter to a value less than 1
N = 5 # number of images

for index in range(1, N + 1):

    input, mask = Read(str(index).zfill(2), inputDir)
    # Performing seam carving. This is the part that you have to implement.
    output, size = SeamCarve(input, widthFac, heightFac, mask)    

    # Writing the result
    plt.imsave("{}/result_{}_{}x{}.jpg".format(outputDir, 
                                            str(index).zfill(2), 
                                            str(size[0]).zfill(2), 
                                            str(size[1]).zfill(2)), output)