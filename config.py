import numpy as np
import cv2

class filter_list():
    def __init__(self):
        self.flist = [
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64), 
            np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float64),
            np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64),
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64) / 9,
            np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16,

            np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                      [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=np.float64) / 256,

            np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6],
                      [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=np.float64) / -256,
        ]

    def get_filter(self, mode:str)->np.ndarray:
        if mode is 'identity':
            return self.flist[0]
        elif mode is 'ed1':
            return self.flist[1]
        elif mode is 'ed2':
            return self.flist[2]
        elif mode is 'ed3':
            return self.flist[3]
        elif mode is 'sharpen':
            return self.flist[4]
        elif mode is 'boxblur':
            return self.flist[5]
        elif mode is 'gaublur1':
            return self.flist[6]
        elif mode is 'gaublur2':
            return self.flist[7]
        elif mode is 'unshpaemark':
            return self.flist[8]
        else:
            raise TypeError("Your input mode is not correct, please check again. ")