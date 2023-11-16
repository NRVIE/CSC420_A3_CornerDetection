import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import scipy
from torchvision import transforms
from torchvision.io import read_image

def gradient(img):
    """Assume the imput image has only one channel"""
    sobel_x = np.outer([1, 2, 1], [-1, 0, 1])
    sobel_y = np.outer([-1, 0, 1], [1, 2, 1])
    gx = scipy.signal.convolve2d(img, sobel_x, mode='same')
    gy = scipy.signal.convolve2d(img, sobel_y, mode='same')
    return gx, gy

def create_grid_cells(img, length):
    """

    :param img: Assuming the img is gray scale
    :param length: size of a cell
    :return:
    """
    # Computing the size of grid cell
    grid_size = (img.shape[0]//length, img.shape[1]//length)

    # Computing 3D array for a grid cell
    grid_arr = np.zeros((grid_size[0], grid_size[1], 6))
    # Center-cropped the image to the size [cell_size[0]*length, cell_size[1]*length]
    transform = transforms.CenterCrop((grid_size[0] * length, grid_size[1] * length))
    crop_img = transform(img).numpy()

    # Calculating the angle of orientation of each pixel in img
    gx, gy = gradient(crop_img)
    angle_arr = np.degrees(np.arctan(gy / gx))
    mag_arr = gx ** 2 + gy ** 2
    for x in range(0, grid_size[0]):
        for y in range(0, grid_size[1]):
            for row in range(0 + (length * x), length * (x + 1)):
                for col in range(0 + (length * y), length * (y + 1)):
                    # Classify which bin does current index belong to
                    # and store the mag. to that correspond bin.
                    curr_angle = angle_arr[row][col]
                    if -15 <= curr_angle < 165:
                        if curr_angle > 0:
                            curr_angle = curr_angle - 180
                        else:
                            curr_angle = curr_angle + 180

                    if -15 <= curr_angle < 15:
                        grid_arr[x][y][0] += mag_arr[row][col]
                    elif 15 <= curr_angle < 45:
                        grid_arr[x][y][1] += mag_arr[row][col]
                    elif 45 <= curr_angle < 75:
                        grid_arr[x][y][2] += mag_arr[row][col]
                    elif 75 <= curr_angle < 105:
                        grid_arr[x][y][3] += mag_arr[row][col]
                    elif 105 <= curr_angle < 135:
                        grid_arr[x][y][4] += mag_arr[row][col]
                    elif 135 <= curr_angle < 165:
                        grid_arr[x][y][5] += mag_arr[row][col]
    return grid_arr

image = read_image('Q3/1.jpg')
