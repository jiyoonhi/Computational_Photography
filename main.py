# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import math

# Function to retrieve r, g, b planes from Prokudin-Gorskii glass plate images
def read_strip(path):
    image = plt.imread(path) # read the input image
    info = np.iinfo(image.dtype) # get information about the image type (min max values)
    image = image.astype(np.float64) / info.max # normalize the image into range 0 and 1

    height = int(image.shape[0] / 3)

    # For images with different bit depth
    scalingFactor = 255 if (np.max(image) <= 255) else 65535
    
    # Separating the glass image into R, G, and B channels
    b = image[: height, :]
    g = image[height: 2 * height, :]
    r = image[2 * height: 3 * height, :]
    return r, g, b

# circshift implementation similar to matlab
def circ_shift(channel, shift):
    shifted = np.roll(channel, shift[0], axis = 0)
    shifted = np.roll(shifted, shift[1], axis = 1)
    return shifted

def resize(image, out_h, out_w):
    h, w = image.shape[:2]

    tmp_h = out_h / h
    tmp_w = out_w / w

    out = np.zeros([out_h, out_w], dtype=image.dtype)

    for i in range(out_h):
        for j in range(out_w):
            tmp_i = int(i / tmp_h)
            tmp_j = int(j / tmp_w)

            out[i, j] = image[tmp_i, tmp_j]
    return out

def find_shift(im1, im2, flag=0):
    # define the level depending on the size of the image.
    if flag == 1:
      level = 0
    else:
      level =round(math.log2(im1.shape[1]//100))

    save = {}
    save[level+1] = [0, 0]

    def pyramid(level, tmp=20):
        scale = 2**level
        min_score = float('inf')
        min_tx, min_ty = 0, 0

        # scaled image
        print("how small is the image at the current level of ",level, " : ", im1.shape[0]//scale)
        print("min [tx,ty] at the previous level? ", save[level+1][0],save[level+1][1])
        im1_resized = resize(im1, im1.shape[0]//scale, im1.shape[1]//scale)
        im2_resized = resize(im2, im2.shape[0]//scale, im2.shape[1]//scale)

        # Extra credits - Edge Detection (better features)
        if flag == 2:    
          im1_edges = np.abs(signal.convolve2d(im1_resized,  np.array([[-1, 0, 1]])))
          im2_edges = np.abs(signal.convolve2d(im2_resized,  np.array([[-1, 0, 1]])))
          im1_resized = im1_edges
          im2_resized = im2_edges
        
        if level < 3:
            size = 5
        else:
            size = 20
        if flag == 1:
            size = 20
        
            
        for tx in range(2*save[level+1][0] - size, 2*save[level+1][0] + size):
            for ty in range(2*save[level+1][1] - size, 2*save[level+1][1] + size):
                # Shift the G image
                shift = [tx, ty]

                if flag == 1:
                  im1_translated = circ_shift(im1, shift)
                else:
                  im1_translated = circ_shift(im1_resized, shift)
                  # Calculate the SSD between the B image and the shifted G image
                  m = im2_resized.shape[0]
                  n = im2_resized.shape[1]
                
                # RGB based SSD
                if flag == 1:
                  SSD = np.sum((im1_translated - im2)**2) 
                else:
                  SSD = np.sum((im1_translated[m//2 - tmp: m//2 + tmp, n//2 - tmp: n//2 + tmp] - 
                            im2_resized[m//2 - tmp: m//2 + tmp, n//2 - tmp: n//2 + tmp])**2)

                # Update the minimum SSD and the corresponding tx and ty
                if SSD < min_score:
                    min_score = SSD
                    min_tx, min_ty = tx, ty

        save[level] = [min_tx, min_ty]
        # original size of image (level 0)
        if level == 0: 
            print("Final min_tx and min_ty : ", min_tx, min_ty)
            return min_tx, min_ty

        return pyramid(level=level-1, tmp = tmp * 2)

    return pyramid(level, tmp=20)

# Automatic Contrasting
def contrast(img, alpha=1): # alpha = 1 means no scaling
    img_changed = img * alpha
    img_changed = np.clip(img_changed, 0, 255).astype(np.uint8)
    return img_changed

# Automatic white balance
def white_balance(img):
    # Convert the image to float32
    img = np.float32(img) / 255.0

    # Find the brightest color in the image
    brightest = np.max(img.reshape(-1, 3), axis=0)

    # Scale the image - the brightest color becomes white (255, 255, 255)
    scale = np.divide(255.0, brightest)
    img_scaled = np.multiply(img, scale)
    img_final = np.uint8(img_scaled)

    return img_final

if __name__ == '__main__':
    # Setting the input output file path
    imageDir = '../Images/'
    outDir = '../Results/'
    imageName = 'monastery.jpg'
    
    # Get r, g, b channels from image strip
    r, g, b = read_strip(imageDir + imageName)

    # Calculate shift without edge detection.
    if imageName[-3:] == "jpg":
        rShift = find_shift(r, b, 1)
        gShift = find_shift(g, b, 1)
    else:
        rShift = find_shift(r, b)
        gShift = find_shift(g, b)

    # # EXTRA - Calculate shift using edge detection.
    # print("Better features - Edge Detection")
    # rShift = find_shift(r, b, 2)
    # gShift = find_shift(g, b, 2)

    print("\n ** Final Shift - R and G compared to B. **\n", "R: ", rShift, "G: ", gShift)
    
    # Shifting the images using the obtained shift values
    finalB = b
    finalG = circ_shift(g, gShift)
    finalR = circ_shift(r, rShift)

    # Putting together the aligned channels to form the color image
    finalImage = np.stack((finalR, finalG, finalB), axis = 2)

    # Writing the image to the Results folder
    plt.imsave(outDir + imageName[:-4] + '_jiyoon.jpg', finalImage)
    print("Task 1 / 2 is done. Image is saved as a name of " + outDir + imageName[:-4] + '_jiyoon.jpg' + " at the end.")

    # EXTRA - Writing Automatic Contrasting
    tmp = plt.imread(outDir + imageName[:-4] + '_jiyoon.jpg')
    finalImage = contrast(tmp, alpha=0.5)
    plt.imsave(outDir + imageName[:-4] + '_contrast.jpg', finalImage)
    plt.imshow(finalImage)
    print("\nAutomatic Contrasting Done - " + outDir + imageName[:-4] + "_contrast.jpg")
    
    # EXTRA - Automatic white balance
    finalImage = white_balance(tmp)
    plt.imsave(outDir + imageName[:-4] + '_white_balance.jpg', finalImage)
    print("\nAutomatic White Balance Done - " + outDir + imageName[:-4] + "_white_balance.jpg")