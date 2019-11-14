from cv2 import cv2
import numpy as np
import time
from filter import Filters

def Test(image:np.ndarray, kernel:np.ndarray, image_show=False):
    F = Filters()

    b1 = time.time()
    result_image1 = F.filter2D(image, kernel)
    e1 = time.time()
    result_time1 = (e1 - b1)

    b2 = time.time()
    result_image2 = cv2.filter2D(image, -1, kernel)
    e2 = time.time()
    result_time2 = (e2 - b2)
    
    if image_show:
        cv2.imshow('source image', image)
        cv2.imshow('F.filter2D', result_image1)
        cv2.imshow('cv2.filter2D', result_image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f'F.filter2D time: {result_time1}')
    print(f'cv2.filter2D time: {result_time2}')

kernel_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
kernel_2 = np.array([[-1, 0, -1], [-2, 0, 2], [-1, 0, 1]])
blur_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

if __name__ == '__main__':
    image = cv2.imread('Lena_color.bmp', 0)
    # image = cv2.imread('c.png', 0)
    Test(image, kernel=kernel_1, image_show=True)