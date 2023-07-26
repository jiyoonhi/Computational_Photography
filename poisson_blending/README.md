## Main Task 
Run the "main.py" with isPoisson flag set to True on the line number 165. 
```
isPoisson = True 
```

To get a naive blended image, change the flag to False
```
isPoisson = False
```
## Extra Credit - Mixing gradients
When you run the "main.py", it will automatically implement Mixing gradients blending for images with index 05 and 06 and 08 (my custom image) because of the if statement. (line 185~189)

```
# Mixing Gradients Blending 
if index == 5 or index == 6 or index == 8:
    isMix = True
# Gradient Blending
else:
    isMix = False
```

