## Instruction to align images

1. Task 1 - Image alignment of ".img" image formats.
Just run the code. It will save the aligned image in the ../Results/ folder with '_jiyoon.jpg' name tag.

2. Task 2 - Image alignment of ".tif" image formats using pyramid.
Just run the code. It will save the aligned image in the ../Results/ folder with '_jiyoon.jpg' name tag.

3. [Extra] Better features - Edge detection
Comment line 149 - 155 and uncomment line 157 - 160 to set input flag as 2.
```
[149] # Calculate shift without edge detection.
if imageName[-3:] == "jpg":
    rShift = find_shift(r, b, 1)
    gShift = find_shift(g, b, 1)
else:
    rShift = find_shift(r, b)
    gShift = find_shift(g, b)

[157] # # EXTRA - Calculate shift using edge detection.
# print("Better features - Edge Detection")
# rShift = find_shift(r, b, 2)
# gShift = find_shift(g, b, 2)
```

4. [Extra] Writing Automatic Contrasting
It is implemented between line 176 ~ 182. It is run by default and show you the image.

5. [Extra] Writing white balance
It is implemented between line 184 ~ 188. It is run by default and show you the image.

[Note] If you want to deactivate contrasting and white balancing functions, you can comment all the code lines.


## main function

There are 3 options (using flag) to align images. 
Note that you only need to change flag to 2 when you want to align images with edge detection.
Otherwise, you don't have to change flag.
```
rShift = find_shift(r, b, flag) 
gShift = find_shift(g, b, flag)
```

1. flag = 0 (default) - Find shift of ".tif" image files without Edge detection 
```
rShift = find_shift(r, b) 
gShift = find_shift(g, b)
```

2. flag = 1 - Find shift of ".jpg" image files without Edge detection 
```
rShift = find_shift(r, b, 1) 
gShift = find_shift(g, b, 1)
```

3. flag = 2 - Find shift of any image files with Edge detection 
```
rShift = find_shift(r, b, 2) 
gShift = find_shift(g, b, 2)
```

