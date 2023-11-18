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
    threshold = 80  # For approach one
    # threshold = 8  # For approach two

    # Computing 3D array for a grid cell
    grid_arr = np.zeros((grid_size[0], grid_size[1], 6))
    # Center-cropped the image to the size [cell_size[0]*length, cell_size[1]*length]
    transform = transforms.CenterCrop((grid_size[0] * length, grid_size[1] * length))
    crop_img = transform(torch.from_numpy(img)).numpy()

    # Calculating the angle of orientation of each pixel in img
    gy, gx = gradient(crop_img)
    mag_arr = np.sqrt(gx ** 2 + gy ** 2)
    mag_arr[mag_arr < threshold] = 0  # Threshold for approach one
    # Find the indices of gx == 0, and set the value of gy to be 0 at the same indices
    idxs = np.where(gx == 0)
    for i in range(0, idxs[0].shape[0]):
        gy[idxs[0][i]][idxs[1][i]] = 3.8
        gx[idxs[0][i]][idxs[1][i]] = 1

    angle_arr = np.degrees(np.arctan(gy / gx))
    # for i in range(0, idxs[0].shape[0]):
    #     angle_arr[idxs[0][i]][idxs[1][i]] = 90

    # # Calculating the angle of orientation of each pixel in img
    # gx, gy = gradient(crop_img)
    # # Find the indices of gx == 0, and set the value of gy to be 0 at the same indices
    # idxs = np.where(gx == 0)
    # for i in range(0, idxs[0].shape[0]):
    #     gy[idxs[0][i]][idxs[1][i]] = 0
    #     gx[idxs[0][i]][idxs[1][i]] = 1
    #
    # angle_arr = np.degrees(np.arctan(gy / gx))
    # mag_arr = np.sqrt(gx ** 2 + gy ** 2)
    # mag_arr[mag_arr < threshold] = 0  # Threshold for approach one

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
                        else:
                            curr_angle = curr_angle + 180

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

                    # Approach one
                    grid_arr[x][y][orientation_bin] += mag_arr[row][col]
                    # Approach two
                    # grid_arr[x][y][orientation_bin] += 1
    # grid_arr[grid_arr[:, :, :] < threshold] = 0  # Threshold for approach 2

    # Normalizing the grid_arr
    new_grid_arr = np.zeros((grid_size[0] - 1,  grid_size[1] - 1, 24))
    for x in range(0, grid_size[0] - 1):
        for y in range(0, grid_size[1] - 1):
            cancat_tup = (grid_arr[x][y], grid_arr[x][y + 1], grid_arr[x + 1][y], grid_arr[x + 1][y + 1])
            cancat_arr = np.concatenate(cancat_tup)
            l2_norm = math.sqrt((cancat_arr**2).sum() + 0.001)
            # l2_norm = np.linalg.norm(cancat_arr, 2)
            new_grid_arr[x][y] = cancat_arr / l2_norm

    # l2_norm = np.linalg.norm(grid_arr, 2, axis=2)
    # l2_norm = l2_norm.reshape((grid_size[0], grid_size[1], 1)).repeat(6, axis=2)
    # grid_arr = grid_arr / l2_norm  # divide l2_norm element-wise

    # Plot the HOG graph if visual is true
    if visual:
        graph_arr = np.zeros((grid_size[0], grid_size[1], 6))
        for x in range(0, grid_size[0] - 2):
            for y in range(0, grid_size[1] - 2):
                for i in range(0, 4):
                    x_p = x
                    y_p = y
                    if i == 1:
                        y_p = y + 1
                    elif i == 2:
                        x_p = x + 1
                    elif i == 3:
                        x_p = x + 1
                        y_p = y + 1

                    graph_arr[x_p][y_p][0] += new_grid_arr[x][y][i * 6]
                    graph_arr[x_p][y_p][1] += new_grid_arr[x][y][i * 6 + 1]
                    graph_arr[x_p][y_p][2] += new_grid_arr[x][y][i * 6 + 2]
                    graph_arr[x_p][y_p][3] += new_grid_arr[x][y][i * 6 + 3]
                    graph_arr[x_p][y_p][4] += new_grid_arr[x][y][i * 6 + 4]
                    graph_arr[x_p][y_p][5] += new_grid_arr[x][y][i * 6 + 5]
        graph_arr = graph_arr / 2
        # Plotting the graph
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
                        graph_u += [graph_arr[x][y][i] * 1]
                        graph_v += [graph_arr[x][y][i] * 0]
                    elif i == 1:
                        graph_u += [graph_arr[x][y][i] * math.sqrt(3) / 2]
                        graph_v += [graph_arr[x][y][i] * 1 / 2]
                    elif i == 2:
                        graph_u += [graph_arr[x][y][i] * 1 / 2]
                        graph_v += [graph_arr[x][y][i] * math.sqrt(3) / 2]
                    elif i == 3:
                        graph_u += [graph_arr[x][y][i] * 0]
                        graph_v += [graph_arr[x][y][i] * 1]
                    elif i == 4:
                        graph_u += [graph_arr[x][y][i] * -1 / 2]
                        graph_v += [graph_arr[x][y][i] * math.sqrt(3) / 2]
                    elif i == 5:
                        graph_u += [graph_arr[x][y][i] * -math.sqrt(3) / 2]
                        graph_v += [graph_arr[x][y][i] * 1 / 2]
        fig, ax = plt.subplots()
        ax.quiver(graph_x, graph_y,
                  graph_u,
                  graph_v,
                  angles='uv', scale_units='xy', scale=1., headaxislength=0)
        ax.set_xlim([0, grid_size[1]])
        ax.set_ylim([0, grid_size[0]])
        plt.show()

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
        ax.set_xlim([0, grid_size[1] - 1])
        ax.set_ylim([0, grid_size[0] - 1])
        plt.show()
    return new_grid_arr

def compute_eigenvalue(img, window, sigma = 3, visual = True):
    # Computing Ix, Ixx, Iy, Iyy
    blur = cv2.GaussianBlur(img, (window, window), 2)
    sobel_x = np.outer([1, 2, 1], [-1, 0, 1])
    sobel_y = np.outer([-1, 0, 1], [1, 2, 1])
    Ix = scipy.signal.convolve2d(blur, sobel_x, mode='same')
    Iy = scipy.signal.convolve2d(blur, sobel_y, mode='same')
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy

    # Gaussian bluring (our window (or weight) function)
    window_func = cv2.getGaussianKernel(ksize=window, sigma=sigma)
    window_func = np.outer(window_func, window_func)
    Ix2_blur = scipy.signal.convolve2d(Ix2, window_func, mode='same')
    Iy2_blur = scipy.signal.convolve2d(Iy2, window_func, mode='same')
    IxIy_blur = scipy.signal.convolve2d(IxIy, window_func, mode='same')

    # Getting eigenvalue of m over the whole image (lambda1 >> lambda0)
    lambda0 = np.zeros((img.shape[0], img.shape[1]))
    lambda1 = np.zeros((img.shape[0], img.shape[1]))
    for x in range(0, Ix2_blur.shape[0]):
        for y in range(0, Ix2_blur.shape[1]):
            #  Getting m at that position
            m = np.array([[Ix2_blur[x][y], IxIy_blur[x][y]],
                          [IxIy_blur[x][y], Iy2_blur[x][y]]])
            lambdas = np.linalg.eigvals(m)
            if lambdas[0] >= lambdas[1]:
                lambda1[x][y] = lambdas[0]
                lambda0[x][y] = lambdas[1]
            else:
                lambda1[x][y] = lambdas[1]
                lambda0[x][y] = lambdas[0]

    if visual:
        # Plot scatter plot
        lambda1_flat = lambda1.flatten()
        lambda0_flat = lambda0.flatten()
        plt.scatter(lambda1_flat, lambda0_flat, s=1)
        plt.title('Scatter Plot of λ1 and λ2')
        plt.xlabel('λ1')
        plt.ylabel('λ2')
        plt.show()

    return lambda1, lambda0



image = cv2.imread("Q3/uoft1.jpg", cv2.IMREAD_GRAYSCALE)
# hog = create_grid_cells(image, 5)
# flattened_array = hog.flatten()
# flattened_array.tofile('3.txt', sep=' ', format='%s')
lambda1, lambda0 = compute_eigenvalue(image, 3, 1)

# transform = transforms.CenterCrop((330, 300))
# image2 = transform(torch.from_numpy(image)).numpy()
# cv2.imwrite("test_img.jpg", image2)
