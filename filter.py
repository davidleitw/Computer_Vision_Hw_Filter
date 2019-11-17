import time 
import numpy as np
from cv2 import cv2
from signal_to_image import Image_from_Signal
from config import filter_list

class Filters(object):
    def __init__(self):
        self.execution_time = 0
        self.signal_function = Image_from_Signal
        self.flist = filter_list()

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
        assert type(src) is np.ndarray, 'Input should be an ndarray. '
        assert type(kernel) is np.ndarray, 'Kernel must be a ndarray. Please check again. '

        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError('The kernel size must be odd. it should be an n * n np.ndarray. ')
        if src.ndim == 3:
            print(f'Your input has 3 channels {src.shape}.')
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            print(f'Converted to gray channels, shape is {src.shape}.')
            
        self.input_image = src
        self.input_shape = src.shape
        self.filter_kernel = kernel

        length, width = self.input_shape[0], self.input_shape[1]
        kernel_size = kernel.shape[0]
        p_width = kernel_size // 2
        
        image = self.padding(src, pad_width=p_width, mode='mirror')
        new_image = np.array(src)

        begin = time.time()
        for x in range(length):
            for y in range(width):
                filter_area = image[x:x+kernel_size, y:y+kernel_size]
                value = (filter_area * kernel).sum() 
                new_image[x, y] = value if value >= 0 else 0
                    
        end = time.time()
        self.execution_time = (end - begin)
        # Change singal to image. 
        new_image = self.signal_function(new_image, Image_show=False, Image_info_show=False)
        self.output_image = new_image
        self.output_shape = self.output_image.shape

        if unit_test is True:
            self.filter2D_test()
        
        return new_image
            
    def padding(self, image:np.ndarray, pad_width:int=0, mode:str='mirror')->np.ndarray:
        '''
            return the image what after filter, the shape will not be changed. 
        '''

        length, width = image.shape[0], image.shape[1]

        new_length = length + pad_width * 2
        new_width  = width  + pad_width * 2
        new_image  = np.zeros((new_length, new_width), dtype=np.float32)
        
        if mode is 'mirror':
            for x in range(length):
                for y in range(width):
                    new_image[x + pad_width, y + pad_width] = image[x, y]
                
            start = pad_width - 1
            w1_shift = width  + 1
            l1_shift = length + 1
                
            while start >= 0:
                new_image[start, start] = new_image[start + 1, start + 1]
                
                #new_image[start, start + 1: start + w1_shift] = new_image[start + 1, start + 1: start + w1_shift]
                for x in range(1, w1_shift):
                    new_image[start, start + x] = new_image[start + 1, start + x]
                for x in range(l1_shift):
                    new_image[start + x, start + w1_shift] = new_image[start + x, start + w1_shift - 1]
                for x in range(start + w1_shift, start, -1):
                    new_image[start + l1_shift, x] = new_image[start + l1_shift - 1, x]
                for x in range(start + l1_shift, start, -1):
                    new_image[x, start] = new_image[x, start + 1]
                
                #print(f'w1_shift = {w1_shift}, l1_shift = {l1_shift}')
                w1_shift = w1_shift + 2
                l1_shift = l1_shift + 2
                start = start - 1
                #print(new_image)
        elif mode is 'zero':
            new_image[pad_width: new_length - pad_width, pad_width: new_width - pad_width] = image  
        else:
            raise ValueError('mode error, please check your input string again. ')

        return new_image

    def get_image_shape(self)->tuple:
        '''
            return the last image shape what your input.  
        '''
        return (self.input_shape, self.output_shape)

    def get_execution_time(self)->float:
        return self.execution_time

    def particular_filter2D(self, src:np.ndarray, mode:str='identity')->np.ndarray:
        self.filter_kernel = self.flist.get_filter(mode=mode)
        return self.filter2D(src, self.filter_kernel)
    
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

