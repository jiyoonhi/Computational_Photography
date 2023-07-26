# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2
from GetMask import GetMask as getmask

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = cv2.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) # normalize the image into range 0 and 1
    target = cv2.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) # normalize the image into range 0 and 1
    mask   = cv2.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / 255 # normalize the image into range 0 and 1

    return source, mask, target

def generate_pyramid(img, num_levels):
    # generate Gaussian pyramid for img
    G = img.copy()
    gp = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gp.append(G)

    # generate Laplacian pyramid for img
    laplacian_top = gp[-1]
    num_levels = len(gp) - 1
    
    lp = [laplacian_top]
    for i in range(num_levels,0,-1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gp[i], dstsize=size)
        laplacian = np.subtract(gp[i-1], gaussian_expanded)
        lp.append(laplacian)
    
    return gp, lp

# Pyramid Blend
def PyramidBlend(source, mask, target, levels=5):
    # Generate Gaussian pyramids for source, target, and mask
    gp1, lp1 = generate_pyramid(source, levels)
    gp2, lp2 = generate_pyramid(target, levels)
    m_gp, m_lp = generate_pyramid(mask, levels)
    m_gp.reverse()
    # Combine the Laplacian pyramids using the mask
    blended_pyramid = []
    for l1,l2,mask in zip(lp1, lp2, m_gp):
        ls = l2 * mask + l1 * (1.0 - mask)
        blended_pyramid.append(ls)

    # Reconstruct the final image from the blended Laplacian pyramid
    l_top = blended_pyramid[0]
    l_bot = [l_top]
    for i in range(levels):
        size = (blended_pyramid[i + 1].shape[1], blended_pyramid[i + 1].shape[0])
        l_ex = cv2.pyrUp(l_top, dstsize=size)
        l_top = cv2.add(blended_pyramid[i+1], l_ex)
        l_bot.append(l_top)
    return l_bot[levels]


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image

    index = 1

    # Read data and clean mask
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    # Create a custom mask 
    # maskOriginal = getmask(target)
    # cv2.imwrite("{}mask_eye_{}.jpg".format(outputDir, str(index).zfill(2)), maskOriginal)
    
    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    ### The main part of the code ###
    # Implement the PyramidBlend function (Task 2)
    level = 6
    pyramidOutput = PyramidBlend(source, mask, target, level)

    # make all values between 0 and 1. (This is not required if you use cv2.imwrite and cv2.imread)
    # pyramidOutput = (pyramidOutput - np.min(pyramidOutput)) / (np.max(pyramidOutput) - np.min(pyramidOutput))

    # Writing the result
    cv2.imwrite("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
    