```python
import numpy as np 
import imageio.v2 as imageio
import cv2
import scipy.ndimage
import argparse
def grayscale(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gaussian_blur(image, sigma=5):
    return scipy.ndimage.gaussian_filter(image, sigma=sigma)
def main(*args):
    # read the image
    img = imageio.imread(args[0])
    # convert to grayscale
    img_gray = grayscale(img)
    # invert the image
    img_invert = 255 - img_gray
    # blur the image by gaussian blur
    img_blur = gaussian_blur(img_invert, sigma=5)
    # create the pencil sketch image
    final_sketch = dodgeV2(img_gray, img_blur)
    # display the sketch image
    cv2.imwrite(args[1], final_sketch)
    
def dodgeV2(x, y):
    return cv2.divide(x, 255 - y, scale=256)    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help="Image Path")
    parser.add_argument('-s', '--save', required=True, help="Save Path")
    args = parser.parse_args()
    main(args.image, args.save)
```
