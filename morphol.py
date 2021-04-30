import math
import numpy as np


def erode(image, window):
    return _filter(image, window, 'erosion')


def dilate(image, window):
    return _filter(image, window, 'dilation')


def _filter(image, window, morph_type):
    m, n = image.shape
    pad = int(math.floor(window / 2))
    padded = np.pad(image, ((pad, pad), (pad, pad)), 'edge')
    tmp = np.zeros((m, n))
    if morph_type == 'erosion':
        for i, j in np.ndindex(tmp.shape):
            tmp[i, j] = np.min(padded[i:i + window, j:j + window])
    elif morph_type == 'dilation':
        for i, j in np.ndindex(tmp.shape):
            tmp[i, j] = np.max(padded[i:i + window, j:j + window])
    else:
        raise Exception('Incorrect morphological type given.')
    return tmp
