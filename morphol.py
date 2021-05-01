import numpy as np
import numpy.random


def erode(image, radius):
    return _filter(image, radius, 'erosion')


def dilate(image, radius):
    return _filter(image, radius, 'dilation')


def close(image, radius):
    dilated = dilate(image, radius)
    return erode(dilated, radius)


def open(image, radius):
    eroded = erode(image, radius)
    return dilate(eroded, radius)


def erode_random(image, r_min, r_max):
    return _filter_random(image, r_min, r_max, 'erosion')


def dilate_random(image, r_min, r_max):
    return _filter_random(image, r_min, r_max, 'dilation')


def close_random(image, r_min, r_max):
    dilated = dilate_random(image, r_min, r_max)
    return erode_random(dilated, r_min, r_max)


def open_random(image, r_min, r_max):
    eroded = erode_random(image, r_min, r_max)
    return dilate_random(eroded, r_min, r_max)


def _filter(image, radius, morph_type):
    m, n = image.shape
    pad = radius
    padded = np.pad(image, ((pad, pad), (pad, pad)), 'edge')
    tmp = np.zeros((m, n))
    if morph_type == 'erosion':
        for i, j in np.ndindex(tmp.shape):
            tmp[i, j] = np.min(padded[i:i + radius*2-1, j:j + radius*2-1])
    elif morph_type == 'dilation':
        for i, j in np.ndindex(tmp.shape):
            tmp[i, j] = np.max(padded[i:i + radius*2-1, j:j + radius*2-1])
    else:
        raise Exception('Incorrect morphological type given.')
    return tmp


def _filter_random(image, r_min, r_max, morph_type):
    m, n = image.shape
    padded = np.pad(image, ((r_max, r_max), (r_max, r_max)), 'edge')
    tmp = np.zeros((m, n))
    if morph_type == 'erosion':
        for i, j in np.ndindex(tmp.shape):
            i_idx, j_idx = _get_rand_idx(i, j, r_min, r_max)
            tmp[i, j] = np.min(padded[i_idx, j_idx])
    elif morph_type == 'dilation':
        for i, j in np.ndindex(tmp.shape):
            i_idx, j_idx = _get_rand_idx(i, j, r_min, r_max)
            tmp[i, j] = np.max(padded[i_idx, j_idx])
    else:
        raise Exception('Incorrect morphological type given.')
    return tmp


def _get_rand_idx(i, j, r_min, r_max):
    r_rand = numpy.random.randint(r_min, r_max)
    i_idx = slice(i - r_rand + r_max - 1, i + r_rand + r_max - 1)
    j_idx = slice(j - r_rand + r_max - 1, j + r_rand + r_max - 1)
    return i_idx, j_idx
