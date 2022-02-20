from math import pi, sin
from os import path, mkdir
import numpy as np

from image_reader import read_pfm, write_pfm, write_ppm

SQUARE_SIZE = 7
GAMMA = 2.2


def centroid(intensity_map: np.ndarray) -> np.ndarray:
    sigma_xI = np.zeros((2,))
    sigma_I = 0
    for i in range(np.shape(intensity_map)[0]):
        for j in range(np.shape(intensity_map)[1]):
            sigma_xI += np.array([i, j]) * (intensity_map[i, j, :].sum())
            sigma_I += intensity_map[i, j, :].sum()
    return np.rint((sigma_xI/sigma_I)).astype(int)


def drawCentroid(image, intensity_map, square_size, colour=None):
    if colour is None:
        colour = [0, 0, 1]  # Blue
    centre = centroid(intensity_map)
    for i in range(square_size):
        for j in range(square_size):
            i_idx = centre[0] - int(square_size / 2) + i
            j_idx = centre[1] - int(square_size / 2) + j
            if i_idx >= 0 and j_idx >= 0:
                try:
                    image[i_idx, j_idx, 0] = colour[0]
                    image[i_idx, j_idx, 1] = colour[1]
                    image[i_idx, j_idx, 2] = colour[2]
                except IndexError:
                    pass


def medianSplit(img, scaled_intensity_map, splits=0, show_divisions=True):
    # Draw centroid if at a leaf in the tree
    if splits == 0:
        colour = scaled_intensity_map.sum(axis=0).sum(axis=0)
        drawCentroid(img, scaled_intensity_map, SQUARE_SIZE, colour=None if show_divisions else colour)
        return img

    # Determine whether to split vertically or horizontally
    axis = 1 if np.shape(scaled_intensity_map)[0] < np.shape(scaled_intensity_map)[1] else 0

    # Find split column/row (as appropriate)
    one_d_intensity_map = scaled_intensity_map.sum(axis=2).sum(axis=1-axis)
    cum_intensity_dist = np.cumsum(one_d_intensity_map)
    half_total_intensity = 0.5 * np.sum(scaled_intensity_map, axis=None)
    split_point = None
    for i, intensity in enumerate(cum_intensity_dist):
        if intensity > half_total_intensity:
            split_point = i
            break

    # Split maps at split point
    left_intensity_map, right_intensity_map = np.split(scaled_intensity_map, [split_point], axis=axis)
    left_img, right_img = np.split(img, [split_point], axis=axis)
    left_img = medianSplit(left_img, left_intensity_map, splits=splits-1, show_divisions=show_divisions)
    right_img = medianSplit(right_img, right_intensity_map, splits=splits-1, show_divisions=show_divisions)

    img = np.concatenate((left_img, right_img), axis=axis)

    # Draw division
    if show_divisions:
        if axis == 1:
            img[:, split_point, :] = 1
        else:
            img[split_point, :, :] = 1

    return img


def scaleIntensityMap(env_map):
    # Number of steps in latitude direction
    theta_steps = np.shape(env_map)[0]

    # Scale intensity map by sin(theta)
    scaled_intensity_map = np.zeros_like(env_map)
    for i in range(theta_steps):
        scaled_intensity_map[i, :, :] = env_map[i, :, :] * sin((i + 0.5) * pi / theta_steps)
    return scaled_intensity_map


def gammaCorrection(pfm_map):
    norm_pfm_map = pfm_map / np.max(pfm_map)
    return (np.power(norm_pfm_map, 1/GAMMA)*255).astype(np.uint8)


if __name__ == "__main__":
    # Read in lat-long environment map
    env_map = read_pfm("GraceCathedral/grace_latlong.pfm")
    scaled_intensity_map = scaleIntensityMap(env_map)

    if not path.exists("./images/"):
        mkdir("images")

    for i in range(7):
        env_map = read_pfm("GraceCathedral/grace_latlong.pfm")
        write_pfm(medianSplit(env_map, scaled_intensity_map, splits=i), f"./images/{pow(2,i)} partitions.pfm")

    empty_img = np.zeros_like(env_map)
    light_source_map = medianSplit(empty_img, scaled_intensity_map, splits=6, show_divisions=False)
    scaled_ls_map = gammaCorrection(light_source_map)
    write_pfm(light_source_map, "./images/light_source_map.pfm")
    write_ppm(scaled_ls_map, "./images/light_source_map.ppm")

