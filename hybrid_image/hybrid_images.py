"""
Credit: Alyosha Efros
""" 
# Task 1

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage.transform as sktr
isGray = True # False for colored hybrid image

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=2)
    else:
        im2 = sktr.rescale(im2, 1./dscale, channel_axis=2)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2

def gaussian_filter(kernel_width, std, image):
    kernel = np.zeros((kernel_width, kernel_width))
    center = kernel_width // 2
    for i in range(kernel_width):
        for j in range(kernel_width):
            kernel[i, j] = np.exp(-((i-center)*(i-center) + (j-center)*(j-center))/(2*std*std))
            
    # normalize kernel 
    kernel = kernel / np.sum(kernel)

    # Apply low-pass filter to the first image
    image_low = scipy.signal.convolve2d(image, kernel, mode='same')
    return image_low


if __name__ == "__main__":

    imageDir = '../Images/'
    outDir = '../Results/'

    # im1_name = 'Monroe.jpg'
    # im2_name = 'Einstein.jpg'

    im1_name = 'Hermione.jpg'
    im2_name = 'Malfoy.jpg'

    # im1_name = 'dog.jpg'
    # im2_name = 'Elon.jpg'
    

    # 1. load the images
	
	# Low frequency image
    im1 = plt.imread(imageDir + im1_name) # read the input image
    info = np.iinfo(im1.dtype) # get information about the image type (min max values)
    im1 = im1.astype(np.float32) / info.max # normalize the image into range 0 and 1
    
	# High frequency image
    im2 = plt.imread(imageDir + im2_name) # read the input image
    info = np.iinfo(im2.dtype) # get information about the image type (min max values)
    im2 = im2.astype(np.float32) / info.max # normalize the image into range 0 and 1
    


    # 2. align the two images by calling align_images
    im1_aligned, im2_aligned = align_images(im1, im2)

    if isGray:
        im1_aligned = np.mean(im1_aligned, axis=2)
        im2_aligned = np.mean(im2_aligned, axis=2)
    # else: 
        # grayscale
        # im1_aligned = np.mean(im1_aligned, axis=2)
       

    # Now you are ready to write your own code for creating hybrid images!
    std = 2  # standard deviation
    kernel_width = std * 3 * 2 # Rule of thumb for Gaussian: set filter half-width to about 3*std (3sigma)
    
    if isGray:
        # Apply low-pass filter (gaussian filter) to the first image 
        im_low = gaussian_filter(kernel_width, std, im1_aligned)
        # Apply high-pass filter (impulse filter) to the second image
        im_high = im2_aligned - gaussian_filter(kernel_width, std, im2_aligned)
    else:
        # Extra credit - hybrid image in color
        # Split the image into R,G,B channel
        low_RGB = []
        high_RGB = []
        for ch in range(3):
            # Apply low-pass filter (gaussian filter) to the first image 
            im_high_ch = im2_aligned[:,:,ch] - gaussian_filter(kernel_width, std, im2_aligned[:,:,ch])
            high_RGB.append(im_high_ch)
            im_low_ch = gaussian_filter(kernel_width, std, im1_aligned[:,:,ch])
            low_RGB.append(im_low_ch)
        im_high = np.dstack((high_RGB[0], high_RGB[1], high_RGB[2]))
        im_low = np.dstack((low_RGB[0], low_RGB[1], low_RGB[2]))


    # Combine two images to produce hybrid images
    im = im_low + im_high

    im = im / im.max()
	
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    

    if isGray:
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_Hybrid.jpg', im, cmap='gray')
    else:
        # Extra credit - hybrid image in color
        plt.imsave(outDir + im1_name[:-4] + '_' + im2_name[:-4] + '_Hybrid.jpg', im)
    
    pass

    # Display hybrid images
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(im_low, cmap='gray')
    plt.title('Low-Pass Filtered Image')

    plt.subplot(1, 2, 2)
    plt.imshow(im_high, cmap='gray')
    plt.title('High-Pass Filtered Image')

    plt.figure(figsize=(10,5))
    plt.imshow(im, cmap='gray')
    plt.title('Hybrid Image')
    plt.show()
