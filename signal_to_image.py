from cv2 import cv2
import numpy as np
import os 
from skimage import img_as_ubyte


def Image_from_Signal(Input=None, Image_show:bool=True, Image_info_show:bool=True)->np.ndarray:
    # check Image whether or not is an image. 
    global Image
    if isinstance(Input, np.ndarray):
        Image = Input
        #print('Input is an image. ')
    # if not an image, check Image whether or not is a path 
    elif isinstance(Input, str):
        if os.path.isfile(Input):
            #print('Input is an image path. ')
            Image = cv2.imread(Input)
            if isinstance(Image, type(None)):
                raise TypeError("Input path is a file, but not path to the image. ")
        else:
            raise TypeError("Input path is not a file, please check again(Input image or path). ")
    else: 
        raise TypeError(
                "Your input data type is {}"
                ", please check again(Input an image or a path). ".format(type(Input))
            )
    
    Image = Image.astype(np.float64)
    Image = Image / Image.max()
    Image = Image * 255              # [0 ~ 255]
    Image = Image.astype(np.uint8)
    '''
    Image = np.abs(Image)
    Image = (Image - np.min(Image)) / (np.max(Image) - np.min(Image)) * 255
    Image = np.round(Image)
    Image.astype(np.uint8)
    '''
    if Image_info_show is True:
        # if input is an image path, show that path. 
        print('*' * 10 + ' ' + 'Image_information' + ' ' + '*' * 10)
        
        if isinstance(Input, str) and os.path.isfile(Input):
            print('Image path: {}. '.format(Input))
        elif isinstance(Input, np.ndarray):
            print('Input is an image. ')

        print('The image shape is {}. '.format(Image.shape))
        print('Max value in image is {}, min value in image is {}. '.format(Image.max(), Image.min()))
        print('After the processing, type is {} now. '.format(Image.dtype))
        print('*' * 39 + '\n')

    if Image_show is True:
        cv2.imshow('Result image', Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return Image

if __name__ == '__main__':
    image = cv2.imread('Lena_color.bmp')
    
    gx=np.array([[-1,-2,-1],
                [ 0, 0, 0],
                [ 1, 2, 1]])

    gy=np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])

    himage = cv2.filter2D(image * 1., cv2.CV_64F, gx)
    vimage = cv2.filter2D(image * 1., cv2.CV_64F, gy)
    print(f'himage dtype = {himage.dtype}, vimage dtype = {vimage.dtype}')


    himage = Image_from_Signal(himage, Image_show=False, Image_info_show=True)
    vimage = Image_from_Signal(vimage, Image_show=False, Image_info_show=True)

    print(f'himage shape = {himage.shape}, vimage shape = {vimage.shape}')
    cv2.imshow('image', image)
    cv2.imshow('himage', himage)
    cv2.imshow('vimage', vimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    '''
    # two test input check our function
    Image_path = 'Lena_color.bmp'
    Image_1 = cv2.imread(Image_path)

    print(Image_1.max(), Image_1.min())
    Image_2 = Image_1.astype(np.float) * 1.89 # change to signal 
    print(Image_2.max(), Image_2.min())
    
    print(Image_2)
    result = Image_from_Signal(Image_2, Image_show=False) # turn signal to image
    #print(np.sum(Image_2 == result))
    #print('pixes num = {}'.format(Image_1.shape[0] * Image_1.shape[1] * Image_1.shape[2]))
    #print(np.sum(Image_1 == Image_2))
    #print(np.sum(Image_1 == result))
    #print(result.max(), result.min())
    cv2.imshow('result_1', Image_1)
    cv2.imshow('result_2', Image_2)
    cv2.imshow('after processing', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Image_path_result = Image_from_Signal(Image_path)

    #Image_1_result = Image_from_Signal(Image_1)
    '''
