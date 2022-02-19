import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin
from image_reader import *
from os import path, mkdir
SQUARE_SIZE = 5


def centroid(intensity_map: np.ndarray) -> np.ndarray:
    sigma_xI = np.zeros((2,))
    sigma_I = 0
    for i in range(np.shape(intensity_map)[0]):
        for j in range(np.shape(intensity_map)[1]):
            sigma_xI += np.array([i, j]) * intensity_map[i, j]
            sigma_I += intensity_map[i, j]
    return np.rint((sigma_xI/sigma_I)).astype(int)


def drawCentroid(image, intensity_map, square_size):
    centre = centroid(intensity_map)
    for i in range(square_size):
        for j in range(square_size):
            try:
                image[centre[0] - int(square_size / 2) + i, centre[1] - int(square_size / 2) + j, :] = 0
                image[centre[0] - int(square_size / 2) + i, centre[1] - int(square_size / 2) + j, 2] = 1
            except IndexError:
                pass


def split(img, scaled_intensity_map, splits=0):
    if splits == 0:
        drawCentroid(img, scaled_intensity_map, SQUARE_SIZE)
        return img

    # Determine whether to split vertically or horizontally
    axis = 1 if np.shape(scaled_intensity_map)[0] < np.shape(scaled_intensity_map)[1] else 0

    # Find split column/row (as appropriate)
    one_d_intensity_map = np.sum(scaled_intensity_map, axis=1 - axis)
    cum_intensity_dist = np.cumsum(one_d_intensity_map)
    half_total_intensity = 0.5 * np.sum(scaled_intensity_map, axis=None)
    split_point = None
    for i, intensity in enumerate(cum_intensity_dist):
        if intensity > half_total_intensity:
            split_point = i
            break

    left_intensity_map, right_intensity_map = np.split(scaled_intensity_map, [split_point], axis=axis)
    left_img, right_img = np.split(img, [split_point], axis=axis)
    left_img = split(left_img, left_intensity_map, splits=splits-1)
    right_img = split(right_img, right_intensity_map, splits=splits-1)

    img = np.concatenate((left_img, right_img), axis=axis)
    # Draw division
    if axis == 1:
        img[:, split_point, :] = 1
    else:
        img[split_point, :, :] = 1
    return img


def scaleIntensityMap(env_map):
    # Number of steps in lattitude direction
    theta_steps = np.shape(env_map)[0]

    # Calculate intensity map from average of the three colours
    intensity_map = (env_map[:, :, 0] + env_map[:, :, 1] + env_map[:, :, 2]) / 3

    # Scale intensity map by sin(theta)
    scaled_intensity_map = np.zeros_like(intensity_map)
    for i in range(theta_steps):
        scaled_intensity_map[i, :] = intensity_map[i, :] * sin((i + 0.5) * pi / theta_steps)
    return scaled_intensity_map


if __name__ == "__main__":
    # Read in lat-long environment map
    env_map = read_pfm("./GraceCathedral/grace_latlong.pfm")
    scaled_intensity_map = scaleIntensityMap(env_map)

    env_map = split(read_ppm("./GraceCathedral/grace_latlong.ppm"), scaled_intensity_map, splits=2)

    write_ppm(env_map, "./envMap.ppm")

