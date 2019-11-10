from cv2 import cv2
import numpy as np
from filter import Filters

if __name__ == '__main__':
    '''
    image = cv2.imread('god.jpg')    
    image = cv2.resize(image, (600, 800))
    cv2.imshow('before Canny', image)
    image = cv2.Canny(image, 100, 200)
    
    print(f'image shape is {image.shape}')
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    
    image = cv2.imread('RGBvalue_4_6-gray.bmp', 0)
    print(image.shape, image.ndim)
    F = Filters()
    idx = 1
    for x in range(4):
        for y in range(6):
            image[x, y] = idx
            idx = idx + 1

    #print(f'image shape: {image.shape}\nimage = {image}') 
    image = np.pad(image, (1, 1), mode='symmetric')
    #print(f'image shape: {image.shape}\nimage = {image}') 
    
    pix = image[0:3, 0:3]
    print(f'pix = {pix}')
    print(f'type of pix is {type(pix)}')


    #print(f'image = {image}')
    #new = F.filter2D(image, gray=True)
    
    '''
    image = cv2.imread('Lena_color.bmp', cv2.IMREAD_GRAYSCALE)
    
    K = np.array([[ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1],
                 [ 1, 1, 1, 1, 1]])
    K = K/25.
    print(f'before image processing, image shape is {image.shape}')
    cv2.imshow('before processing', image)
    

    image = cv2.filter2D(image, ddepth=-1, kernel=K)
    print(f'image shape is {image.shape}')
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''