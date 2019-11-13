from cv2 import cv2
import numpy as np
import time
from filter import Filters

if __name__ == '__main__':

    #image = cv2.imread('RGBvalue_4_6-gray.bmp', 0)
    #image = cv2.imread('Lena_color.bmp')
    image = cv2.imread('c.png', 0)
    F = Filters()
    '''
    idx = 1
    for i in range(4):
        for j in range(6):
            image[i, j] = idx
            idx = idx + 1
    print(image)
    '''
    cv2.imshow('src image', image)
    
    kernel_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_2 = np.array([[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]])
    image_1 = F.filter2D(image, kernel=kernel_1, unit_test=True)
    image_2 = F.filter2D(image, kernel=kernel_2, unit_test=True)
    #print(image_1)
    
    b = time.time()
    image_3 = cv2.filter2D(image, ddepth=-1, kernel=kernel_1)
    e = time.time()
    print(e - b)
    
    image_4 = cv2.filter2D(image, ddepth=-1, kernel=kernel_2)
    b = time.time()
    e = time.time()
    print(e - b)

    cv2.waitKey(0)
    cv2.destroyAllWindows()