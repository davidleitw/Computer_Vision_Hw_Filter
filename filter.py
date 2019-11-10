import numpy as np
#import cv2
from cv2 import cv2
from signal_to_image import Image_from_Signal
import time 

# def filter2D(src, kernel, padding=True)

class Filters():
    def __init__(self)->None:
        self.execution_time = 0

    def filter2D(self, src:np.ndarray, kernel:np.ndarray=np.ones((3, 3), dtype=np.uint8), padding:bool=True, unit_test:bool=False)->np.ndarray:
        '''
            Parameters
            ----------
                src: np.ndarray
                    Input your untreated image. 
                kernel: np.ndarray
                    Input filter kernel(convolution matrix) doing convolution between a kernel and an image.
                    kernel size must be n * n, and n must be odd.  
                padding: bool
                    Decide whether you want to padding for the image. 
                    If padding is True, the image shape will not change after filter2D.  
                unit_test: bool 
                    For demo on the class, let teacher see how it work.                  
            ----------

            Return
            ----------
                return the image after filter processing. 
            ----------
        '''
        if not isinstance(kernel, np.ndarray):
            raise Exception('Kernel must be a ndarray. Please check again. ')
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise Exception('The kernel size must be odd. it should be an n * n np.ndarray. ')
        
        self.input_image = src
        self.image_shape = src.shape
        self.filter_kernel = kernel

        gray = True if src.ndim == 2 else False
        length, width = self.image_shape[0], self.image_shape[1]
        kernel_size = kernel.shape[0]
        kernel_flatten = kernel.flatten()
        p_width = (kernel_size - 1) / 2

        image = self.padding(src, pad_width=(p_width, p_width), gray=gray)
        new_image = np.asarray(src)
        
        begin = time.time()
        if gray is True:
            for x in range(length):
                for y in range(width):
                    filter_area = image[x:x+kernel_size, y:y+kernel_size]
                    new_image[x, y] = filter_area.flatten() * kernel_flatten
                    
        else:
            for x in range(length):
                for y in range(width):
                    for channel in range(src.ndim):
                        filter_area = image[x:x+kernel_size, y:y+kernel_size, channel]
                        new_image[x, y, channel] = filter_area.flatten() * kernel_flatten()

        end = time.time()
        self.execution_time = (end - begin)
        
        new_image = Image_from_Signal(new_image, Image_show=False, Image_info_show=False)
        self.output_image = new_image

        if unit_test is True:
            self.filter2D_test()
            
    def padding(self, image:np.ndarray, pad_width:tuple=(0, 0), gray=False)->np.ndarray:
        '''
            return the image what after filter, the shape will not be changed. 
        '''

        length, width = image.shape[0], image.shape[1]

        new_length = length + pad_width[0] * 2
        new_width  = width + pad_width[1] * 2
        new_image  = np.zeros((new_length, new_width), dtype=np.float32) if gray is True else np.zeros((new_length, new_width, image.shape[2]), dtype=np.float32)
        
        if gray is True:   # gray
            pass
        elif gray is False: # normal
            pass
        
        
        return new_image

    def get_image_shape(self)->tuple:
        '''
            return the last image shape what your input.  
        '''
        return (self.image_shape)

    def particular_filter2D(self, src, mode='identity')->np.ndarray:
        pass
    
    def filter2D_test(self)->None:
        '''
            for class demo. 
            Show input, output image shape, show the image after filter. 
        '''
        cv2.imshow('Before filter', self.input_image)
        cv2.imshow('After filter', self.output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('*' * 10)
        print(f'Input image shape is {self.input_image.shape}.')
        print(f'Input kernel shape is {self.filter_kernel.shape}.')
        print(f'After filter, image shape is {self.output_image.shape}.')
        print(f'Total execution time: {self.execution_time} seconds. ')
        print('*' * 10)

    