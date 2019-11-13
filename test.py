from cv2 import cv2
import numpy as np
from filter import Filters

if __name__ == '__main__':

    image = cv2.imread('RGBvalue_4_6-gray.bmp', 0)
    
    print(image.shape, image.ndim)
    F = Filters()
    #print(image)
    image = F.padding(image, 2, True)
    #print(image, '\n')
    image = F.signal_function(image, Image_show=False, Image_info_show=True)
    print(image)
    
    '''
    idx = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            image[x, y] = idx
            idx = idx + 1

    print(image)
    #  if x not in range(1, 3),  if y not in range(1, 5)
    a = [x for x in range(4)]
    row = image[a, 1]
    print(row)
    '''