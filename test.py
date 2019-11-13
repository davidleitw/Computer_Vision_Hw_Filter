from cv2 import cv2
import numpy as np
from filter import Filters

if __name__ == '__main__':

    image = cv2.imread('RGBvalue_4_6-gray.bmp', 0)
    
    print(image.shape, image.ndim)
    F = Filters()
    #print(image)
    #image = F.padding(image, 2, True)
    #print(image, '\n')
    #image = F.signal_function(image, Image_show=False, Image_info_show=True)
    #print(image)
    
    img = np.ones((2, 4))
    idx = 1
    for i in range(2):
        for j in range(4):
            img[i, j] = idx
            idx = idx + 1
    print(img)
    img = F.padding(img, 2, True)
    