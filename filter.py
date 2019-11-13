import time 
import numpy as np
from cv2 import cv2
from signal_to_image import Image_from_Signal

# def filter2D(src, kernel, padding=True)

class Filters(object):
    def __init__(self):
        self.execution_time = 0
        self.signal_function = Image_from_Signal

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
        
        if not isinstance(kernel, np.ndarray):
            raise TypeError('Kernel must be a ndarray. Please check again. ')
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError('The kernel size must be odd. it should be an n * n np.ndarray. ')
        
        self.input_image = src
        self.image_shape = src.shape
        self.filter_kernel = kernel

        gray = True if src.ndim == 2 else False
        length, width = self.image_shape[0], self.image_shape[1]
        kernel_size = kernel.shape[0]
        kernel_flatten = kernel.flatten()
        p_width = kernel_size // 2
        begin = time.time()
        image = self.padding(src, pad_width=p_width, gray=gray)
        new_image = np.array(src)
        
        
        if gray is True:
            for x in range(length):
                for y in range(width):
                    filter_area = image[x:x+kernel_size, y:y+kernel_size]
                    new_image[x, y] = (filter_area.flatten() * kernel_flatten).sum()
                    
        else:
            for x in range(length):
                for y in range(width):
                    for channel in range(src.ndim):
                        filter_area = image[x:x+kernel_size, y:y+kernel_size, channel]
                        new_image[x, y, channel] = (filter_area.flatten() * kernel_flatten).sum()

        end = time.time()
        self.execution_time = (end - begin)
        # Change singal to image. 
        #new_image = Image_from_Signal(new_image, Image_show=False, Image_info_show=False)
        new_image = self.signal_function(new_image, Image_show=False, Image_info_show=False)
        self.output_image = new_image

        if unit_test is True:
            self.filter2D_test()
        
        return new_image
            
    def padding(self, image:np.ndarray, pad_width:int=0, gray:bool=False)->np.ndarray:
        '''
            return the image what after filter, the shape will not be changed. 
        '''

        length, width = image.shape[0], image.shape[1]

        new_length = length + pad_width * 2
        new_width  = width  + pad_width * 2
        new_image  = np.zeros((new_length, new_width), dtype=np.float32) if gray is True else np.zeros((new_length, new_width, image.shape[2]), dtype=np.float32)
        
        if gray is True:   # gray
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
            print(new_image)

        else:              # normal
            for x in range(length):
                for y in range(width):
                    new_image[x + pad_width, y + pad_width, :] = image[x, y, :]
            
            start = pad_width - 1
            w1_shift = width  + 1
            l1_shift = length + 1
            
            while start >= 0:
                new_image[start, start] = new_image[start + 1, start + 1]
                
                for x in range(1, w1_shift):
                    new_image[start, start + x, :] = new_image[start + 1, start + x, :]
                for x in range(l1_shift):
                    new_image[start + x, start + w1_shift, :] = new_image[start + x, start + w1_shift - 1, :]
                for x in range(start + w1_shift, start, -1):
                    new_image[start + l1_shift, x, :] = new_image[start + l1_shift - 1, x, :]
                for x in range(start + l1_shift, start, -1):
                    new_image[x, start, :] = new_image[x, start + 1, :]
            
                w1_shift = w1_shift + 2
                l1_shift = l1_shift + 2
                start = start - 1
            print(new_image)
        
        return new_image

    def get_image_shape(self)->tuple:
        '''
            return the last image shape what your input.  
        '''
        return (self.image_shape)

    def particular_filter2D(self, src:np.ndarray, mode:str='identity')->np.ndarray:
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

    