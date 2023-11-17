import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import scipy
import torch
from torchvision import transforms
from torchvision.io import read_image

def gradient(img):
    """Assume the imput image has only one channel"""
    sobel_x = np.outer([1, 2, 1], [-1, 0, 1])
    sobel_y = np.outer([-1, 0, 1], [1, 2, 1])
    gx = scipy.signal.convolve2d(img, sobel_x, mode='same')
    gy = scipy.signal.convolve2d(img, sobel_y, mode='same')
    return gx, gy

def create_grid_cells(img, length, visual: bool = True):
    """

    :param img: Assuming the img is gray scale in Numpy array
    :param length: size of a cell
    :return:
    """
    # Computing the size of grid cell
    grid_size = (img.shape[0]//length, img.shape[1]//length)

    # Computing 3D array for a grid cell
    grid_arr = np.zeros((grid_size[0], grid_size[1], 6))
    # Center-cropped the image to the size [cell_size[0]*length, cell_size[1]*length]
    transform = transforms.CenterCrop((grid_size[0] * length, grid_size[1] * length))
    crop_img = transform(torch.from_numpy(img)).numpy()

    # Calculating the angle of orientation of each pixel in img
    gx, gy = gradient(crop_img)
    # Find the indices of gx == 0, and set the value of gy to be 0 at the same indices
    idxs = np.where(gx == 0)
    for i in range(0, np.where(gx == 0)[0].shape[0]):
        gy[idxs[0][i]][idxs[1][i]] = 0
        gx[idxs[0][i]][idxs[1][i]] = 1

    angle_arr = np.degrees(np.arctan(gy / gx))
    mag_arr = np.sqrt(gx ** 2 + gy ** 2)
    for x in range(0, grid_size[0]):
        for y in range(0, grid_size[1]):
            for row in range(0 + (length * x), length * (x + 1)):
                for col in range(0 + (length * y), length * (y + 1)):
                    # Classify which bin does current index belong to
                    # and store the mag. to that correspond bin.
                    curr_angle = angle_arr[row][col]
                    orientation_bin = -1
                    if not(-15 <= curr_angle < 165):
                        if curr_angle > 0:
                            curr_angle = curr_angle - 180
                            neg = True
                        else:
                            curr_angle = curr_angle + 180
                            neg = True

                    if -15 <= curr_angle < 15:
                        orientation_bin = 0
                    elif 15 <= curr_angle < 45:
                        orientation_bin = 1
                    elif 45 <= curr_angle < 75:
                        orientation_bin = 2
                    elif 75 <= curr_angle < 105:
                        orientation_bin = 3
                    elif 105 <= curr_angle < 135:
                        orientation_bin = 4
                    elif 135 <= curr_angle < 165:
                        orientation_bin = 5

                    grid_arr[x][y][orientation_bin] += mag_arr[row][col]
    # Normalizing the grid_arr
    l2_norm = np.linalg.norm(grid_arr, 2, axis=2)
    l2_norm = l2_norm.reshape((grid_size[0], grid_size[1], 1)).repeat(6, axis=2)
    grid_arr = grid_arr / l2_norm  # divide l2_norm element-wise

    # Plot the HOG graph if visual is true
    if visual:
        graph_x = []
        graph_y = []
        graph_u = []
        graph_v = []
        for x in range(0, grid_size[0]):
            for y in range(0, grid_size[1]):
                graph_x += [y + 0.5] * 6
                graph_y += [grid_size[0] - 1.5 - x] * 6
                for i in range(0, 6):
                    if i == 0:
                        graph_u += [grid_arr[x][y][i] * 1]
                        graph_v += [grid_arr[x][y][i] * 0]
                    elif i == 1:
                        graph_u += [grid_arr[x][y][i] * math.sqrt(3) / 2]
                        graph_v += [grid_arr[x][y][i] * 1 / 2]
                    elif i == 2:
                        graph_u += [grid_arr[x][y][i] * 1 / 2]
                        graph_v += [grid_arr[x][y][i] * math.sqrt(3) / 2]
                    elif i == 3:
                        graph_u += [grid_arr[x][y][i] * 0]
                        graph_v += [grid_arr[x][y][i] * 1]
                    elif i == 4:
                        graph_u += [grid_arr[x][y][i] * -1 / 2]
                        graph_v += [grid_arr[x][y][i] * math.sqrt(3) / 2]
                    elif i == 5:
                        graph_u += [grid_arr[x][y][i] * -math.sqrt(3) / 2]
                        graph_v += [grid_arr[x][y][i] * 1 / 2]
        fig, ax = plt.subplots()
        ax.quiver(graph_x, graph_y,
                  graph_u,
                  graph_v,
                  angles='uv', scale_units='xy', scale=1., headaxislength=0)
        ax.set_xlim([0, grid_size[1]])
        ax.set_ylim([0, grid_size[0]])
        plt.show()
    return grid_arr



# image = read_image('Q3/1.jpg')
# arr0 = create_grid_cells(image, 8)
image = cv2.imread("Q3/3.jpg", cv2.IMREAD_GRAYSCALE)
arr0 = create_grid_cells(image, 3)
# transform = transforms.CenterCrop((330, 300))
# image2 = transform(torch.from_numpy(image)).numpy()
# cv2.imwrite("test_img.jpg", image2)
