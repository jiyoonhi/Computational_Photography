import math
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from gsolve import gsolve

# Based on code by James Tompkin
#
# reads in a directory and parses out the exposure values
# files should be named like: "xxx_yyy.jpg" where
# xxx / yyy is the exposure in seconds. 
def ParseFiles(calibSetName, dir):
    imageNames = os.listdir(os.path.join(dir, calibSetName))
    
    filePaths = []
    exposures = []
    
    for imageName in imageNames:
        exposure = imageName.split('.')[0].split('_')
        exposures.append(int(exposure[0]) / int(exposure[1]))
        filePaths.append(os.path.join(dir, calibSetName, imageName))
    
    # sort by exposure
    sortedIndices = np.argsort(exposures)[::-1]
    filePaths = [filePaths[i] for i in sortedIndices]
    exposures = [exposures[i] for i in sortedIndices]
    
    return filePaths, exposures

# Setting up the input output paths and the parameters
inputDir = '../Images/'
outputDir = '../Results/'

_lambda = 50

dir = ['Chapel', 'Office'] 

for calibSetName in dir:

    # Parsing the input images to get the file names and corresponding exposure
    # values
    filePaths, exposures = ParseFiles(calibSetName, inputDir)


    """ Task 1 """
    P = len(exposures) # number of images
    images = [cv2.imread(f) for f in filePaths]
    row = len(images[0])
    col = len(images[0][0])

    # Sample the images
    N = math.ceil(5 * 256 / (P - 1)) # number of samples to perform the optimization.
    nPix = row * col
    step = int(np.floor(nPix/N))
    sampleIdx = list(range(0, nPix, step))[:-1]

    flatImage = np.zeros((P, 3, nPix), dtype=np.uint8)
    for i in range(P):
        for c in range(3):
            flatImage[i,c] = np.reshape(images[i][:,:,c], (nPix,))

    Z_R = np.zeros((N, P), dtype=np.uint8)
    Z_G = np.zeros((N, P), dtype=np.uint8)
    Z_B = np.zeros((N, P), dtype=np.uint8)

    for i in range(P):            
        Z_B[:,i] = flatImage[i,0][sampleIdx]
        Z_G[:,i] = flatImage[i,1][sampleIdx]
        Z_R[:,i] = flatImage[i,2][sampleIdx]

    r, c = Z_G.shape[:2]
    B = np.log(exposures)

    # Create the triangle function
    def triangleFunction():
        Zmin, Zmax = 0, 255
        Zmiddle = (Zmin + Zmax) // 2
        w = np.zeros((Zmax - Zmin + 1))

        for z in range(Zmin, Zmax + 1):
            if z > Zmiddle:
                w[z] = Zmax - z + 1
            else:
                w[z] =  z - Zmin + 1
        return w

    w = triangleFunction()

    # Recover the camera response function (CRF) using Debevec's optimization code (gsolve.m)
    # g(z) is the log exposure corresponding to pixel value z
    # lE(i) is the log film irradiance at pixel location i

    g_R, lE_R = gsolve(Z_R, B, _lambda, w)
    g_G, lE_G = gsolve(Z_G, B, _lambda, w)
    g_B, lE_B = gsolve(Z_B, B, _lambda, w)

    # Plot CRF 
    px = list(range(0,256))
    fig = plt.figure(constrained_layout=False,figsize=(10,8))
    plt.title("Response curves for RGB", fontsize=20)
    plt.plot(g_R, px, 'r')
    plt.plot(g_G, px, 'g')
    plt.plot(g_B, px, 'b')
    plt.xlabel("log Exposure X", fontsize=13)
    plt.ylabel("pixel value Z", fontsize=13)
    # plt.xlim(-20, 5)
    # plt.ylim(0, 300)
    fig.savefig("{}/CRF_plot_{}.jpg".format(outputDir, calibSetName))


    """ Task 2 """
    # Reconstruct the radiance using the calculated CRF
    E_i = np.zeros(flatImage.shape[1:]) # ln(E_i) = sum_P(w(Z)(g(Z)-ln(delta(t))) / sum_P(w(Z)) --> E_i = exp(ln(E_i))
    w_sum = np.zeros(flatImage.shape[1:]) # sum(w(Z))
    lnDeltat = np.log(exposures) # ln(delta(t))


    for i in range(P):
        # w(Z)
        w_sum[0, :] += w[flatImage[i, 0]] # w_B
        w_sum[1, :] += w[flatImage[i, 1]] # w_G 
        w_sum[2, :] += w[flatImage[i, 2]] # w_R

        # g(Z)-ln(delta(t))
        g_lnDeltat_B = np.subtract(g_B[flatImage[i,0]], B[i])
        g_lnDeltat_G = np.subtract(g_G[flatImage[i,1]], B[i])
        g_lnDeltat_R = np.subtract(g_R[flatImage[i,2]], B[i])

        # sum_P(w(Z)*(g(Z)-ln(delta(t)))
        E_i[0] += np.multiply(g_lnDeltat_B, w[flatImage[i, 0]])
        E_i[1] += np.multiply(g_lnDeltat_G, w[flatImage[i, 1]])
        E_i[2] += np.multiply(g_lnDeltat_R, w[flatImage[i, 2]])

    # ln(E_i) = sum_P(w(Z)(g(Z)-ln(delta(t))) / sum_P(w(Z)) 
    E_i = np.divide(E_i, w_sum)
    # E_i = exp(ln(E_i))
    E_i = np.exp(E_i)
    E_i = np.reshape(np.transpose(E_i), (row, col, 3))


    """ Task 3 """

    # Perform both local and global tone-mapping

    # Global tone-mapping
    gamma = 0.1
    global_out = np.power(E_i / np.amax(E_i), gamma).astype(np.float32)
    global_output = cv2.cvtColor(global_out,cv2.COLOR_BGR2RGB)
    plt.imsave("{}/global_{}.jpg".format(outputDir, calibSetName), global_output)

    # Local tone-mapping
    if calibSetName == 'Office':
        gamma_local = 0.3
    else:
        gamma_local = 0.5
    dR = 5
    std = 2

    # Local tone-mapping
    # Compute the log-average luminance of the HDR image
    intensity = E_i.astype(np.float32)
    log_luminance = np.log2(np.mean(intensity))

    # Filter that with a Gaussian filter: B = filter(L)
    kernel_size = (5, 5)
    gaussian_B = cv2.GaussianBlur(np.array([log_luminance]), kernel_size, std)
    # log_luminance = log_luminance - gaussian_B
    D = log_luminance - gaussian_B
    o = np.max(gaussian_B)
    # scale:  s = dR / (max(B) - min(B)).
    scale = dR / (np.max(gaussian_B) - np.min(gaussian_B) + 1e-6)
    B_offset = np.multiply((np.subtract(gaussian_B, o)), scale)
    # Reconstruct the log intensity O = 2^(B'+D)
    reO = np.power(2, B_offset + D)
    # R',G',B' = O * (R/I, G/I, B/I)
    if calibSetName == 'Office':
        output = np.multiply(reO, np.divide(intensity,log_luminance))
        ranMax = 6
    else:
        output = np.multiply(reO, np.divide(intensity,1+intensity))
        ranMax = 1

    # Perform local tonemapping
    tone_mapped_image = cv2.normalize(output, None, 0, ranMax, cv2.NORM_MINMAX)
    # Apply gamma compression
    tone_mapped_image = np.power(tone_mapped_image, gamma_local)

    # Apply the tone mapping function
    Ld_gamma_rgb = cv2.cvtColor(tone_mapped_image, cv2.COLOR_BGR2RGB)
    Ld_gamma_rgb = np.clip(Ld_gamma_rgb, 0, 1)
    plt.imsave("{}/local_{}.jpg".format(outputDir, calibSetName), Ld_gamma_rgb)