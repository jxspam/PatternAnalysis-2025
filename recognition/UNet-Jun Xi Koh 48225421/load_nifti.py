import numpy as np
import nibabel as nib
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Convert a label image into one-hot encoded channel representation."""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res


# --------------------- Load Medical Image Functions ---------------------

def load_data_2D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, early_stop=False):
    '''
    Load medical image data from names (2D slices).
    Pre-allocates 4D arrays for conv2D to avoid excessive memory usage.

    Parameters:
        normImage (bool): Normalize image to 0.0–1.0 range.
        categorical (bool): Convert labels to one-hot.
        dtype (np.dtype): Data type (float32 for images, uint8 for labels).
        getAffines (bool): Return affine transforms.
        early_stop (bool): Stop early for testing.
    '''
    affines = []

    # --- Get fixed size from first image ---
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    # --- Load all images ---
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # remove extra dim

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, orient=False,
                 early_stop=False):
    '''
    Load medical image data from names (3D volumes).
    Pre-allocates 5D arrays for conv3D to avoid excessive memory usage.

    Parameters:
        normImage (bool): Normalize image to 0.0–1.0 range.
        orient (bool): Apply orientation and resampling.
        categorical (bool): Convert labels to one-hot.
        dtype (np.dtype): Data type.
        getAffines (bool): Return affine transforms.
        early_stop (bool): Stop early for testing.
    '''
    affines = []
    interp = 'linear'
    if dtype == np.uint8:
        interp = 'nearest'  # for label maps

    # --- Get fixed size from first image ---
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

    first_case = niftiImage.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # remove extra dim

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    # --- Load all images ---
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)

        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)

        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]

        inImage = inImage[:, :, :depth]  # clip slices if needed
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :inImage.shape[0], :inImage.shape[1],
                   :inImage.shape[2], :inImage.shape[3]] = inImage
        else:
            images[i, :inImage.shape[0], :inImage.shape[1],
                   :inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images
