# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from GetMask import GetMask as getmask

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    # Get a custom mask
    # mask = getmask(source)
    # cv2.imwrite("../Images/mask_{}.jpg".format(str(id).zfill(2)), mask)
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

# Adjust parameters, source and mask for negative offsets or out of bounds of offsets
def AlignImages(mask, source, target, offset):
    sourceHeight, sourceWidth, _ = source.shape
    targetHeight, targetWidth, _ = target.shape
    xOffset, yOffset = offset
    
    if (xOffset < 0):
        mask    = mask[abs(xOffset):, :]
        source  = source[abs(xOffset):, :]
        sourceHeight -= abs(xOffset)
        xOffset = 0
    if (yOffset < 0):
        mask    = mask[:, abs(yOffset):]
        source  = source[:, abs(yOffset):]
        sourceWidth -= abs(yOffset)
        yOffset = 0
    # Source image outside target image after applying offset
    if (targetHeight < (sourceHeight + xOffset)):
        sourceHeight = targetHeight - xOffset
        mask    = mask[:sourceHeight, :]
        source  = source[:sourceHeight, :]
    if (targetWidth < (sourceWidth + yOffset)):
        sourceWidth = targetWidth - yOffset
        mask    = mask[:, :sourceWidth]
        source  = source[:, :sourceWidth]
    
    maskLocal = np.zeros_like(target)
    maskLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = mask
    sourceLocal = np.zeros_like(target)
    sourceLocal[xOffset:xOffset + sourceHeight, yOffset:yOffset + sourceWidth] = source

    return sourceLocal, maskLocal

# Implement Poisson Blending
def PoissonBlend(source, mask, target, isMix): 
    # membrane - Blending region   
    mask_h, mask_w = mask.shape
    numPixel = mask_h * mask_w
    
    # Flatten source, target, mask image
    source = source.flatten(order='C')
    target = target.flatten(order='C')
    mask = mask.flatten(order='C')

    # guidance vector field and laplacian
    field = np.empty_like(mask)
    
    # initialize the sparse matrix
    lap = sparse.lil_matrix((numPixel, numPixel), dtype='float64')

    for i in range(numPixel):
        # construct the sparse laplacian block matrix
        # membrane interpolation
        if(mask[i] > 0.99):
            lap[i, i] = 4
            np_s = [0]*4
            np_t = [0]*4

            # construct laplacian, and compute source and target gradient in mask
            if(i - mask_w > 0):
                lap[i, i-mask_w] = -1
                np_s[0] = source[i] - source[i-mask_w]
                np_t[0] = target[i] - target[i-mask_w]
            else:
                np_s[0] = source[i]
                np_t[0] = target[i]
                
            if(i % mask_w != 0):
                lap[i, i-1] = -1
                np_s[1] = source[i] - source[i-1]
                np_t[1] = target[i] - target[i-1]
            else:
                np_s[1] = source[i]
                np_t[1] = target[i]
                
            if(i + mask_w < numPixel):
                lap[i, i+mask_w] = -1
                np_s[2] = source[i] - source[i+mask_w]
                np_t[2] = target[i] - target[i+mask_w]
            else:
                np_s[2] = source[i]
                np_t[2] = target[i]
                
            if(i % mask_w != mask_w - 1):
                lap[i, i+1] = -1
                np_s[3] = source[i] - source[i+1]
                np_t[3] = target[i] - target[i+1]
            else:
                np_s[3] = source[i]
                np_t[3] = target[i]
            
            # when it's not mixing gradients, choose better gradient
            if not isMix:
                np_t = [0]*4
                
            field[i] = sum(x if abs(x) > abs(y) else y for x, y in zip(np_s, np_t))

        else:
            # Boundary condition
            # if point is out of range of membrane, copy the target function
            lap[i, i] = 1
            field[i] = target[i]
    
    return [lap, field]
  
# Perform Poisson Blending for each channel image and reconstruct it to final image.
def ConstructImage(source, mask, target, isMix, isPoisson):
    if not isPoisson:
        final_image = source * mask + target * (1 - mask)
        
    elif isPoisson:
        poisson_eq = []
        
        # construct poisson equation 
        for ch in range(3):
            source_ch = source[:,:,ch]
            target_ch = target[:,:,ch]
            mask_ch = mask[:,:,ch]
            poisson_eq.append(PoissonBlend(source_ch, mask_ch, target_ch, isMix))

        # solve poisson equation and perform poisson cloning
        tmp_image = np.empty_like(source)
        for i in range(3):
            A = poisson_eq[i][0]
            b = poisson_eq[i][1]
            lls_solved = spsolve(A.tocsc(),b)
            tmp_image[:,:,i]  = np.reshape(lls_solved,(source.shape[0],source.shape[1]))

        final_image = np.clip(tmp_image,0.0,1.0)
    return final_image

if __name__ == '__main__':
    # Setting up the input output patmask_h
    inputDir = '../Images/'
    outputDir = '../Results/'
    
    # False for source gradient, true for mixing gradients
    isMix = False

    # (1) Naive blending or (2) Poisson blending
    isPoisson = True  # False for naive blending (copy and paste)

    # Source offsets in target
    offsets = [[210, 10], [10, 28], [140, 80], [-40, 90], [60, 100], [20, 20], [-28, 88], [20, 20], [50, 20]]

    # main area to specify files and display blended image
    for index in range(len(offsets)):

       # Read data and clean mask
        source, maskOriginal, target = Read(str(index+1).zfill(2), inputDir)

        # Cleaning up the mask
        mask = np.ones_like(maskOriginal)
        mask[maskOriginal < 0.5] = 0

        # Align the source and mask using the provided offest
        source, mask = AlignImages(mask, source, target, offsets[index])

        ### The main part of the code ###
        # Mixing Gradients Blending 
        if index == 5 or index == 6 or index == 8:
           isMix = True
        # Gradient Blending
        else:
           isMix = False

        # Implement the Poisson Blending 
        poissonOutput = ConstructImage(source, mask, target, isMix, isPoisson)

        # Writing the result
        if not isMix:
            plt.imsave("{}poisson_{}.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)
        else:
            plt.imsave("{}poisson_{}_Mixing.jpg".format(outputDir, str(index+1).zfill(2)), poissonOutput)