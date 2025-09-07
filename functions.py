### Script for some useful functions
### @author : Guozheng Xu
### @date   : 2024-07-06
############################################################################
from scipy.io import savemat
import os
import numpy as np
import h5py
import pandas as pd
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
def write_txt(file_path,content):
    try:
        file = open(file_path, 'a')
        file.write(content)
        file.close()
    except FileNotFoundError as e:
        print(f'An error occurred: {e}')
        
def curve_fitting_SAO(metrics,bias):
    '''
    Parabola Fitting method by 2N+1 observations
    '''
    org = metrics[0]
    c_cors = []
    for i in range(metrics.shape[0]//2):
        m_p = metrics[i*2+2]
        m_n = metrics[i*2+1]
        c_cor = -bias*(m_p-m_n)/(2*m_p-4*org+2*m_n)
        c_cors.append(c_cor)
    return np.array(c_cors)

def centralized_padding(original_array, target_size):
    """
    Apply centralized zero-padding to a 2D array.
    
    Parameters:
    original_array (np.ndarray): The original 2D input array.
    target_size (tuple): The desired output size after padding (height, width).
    
    Returns:
    np.ndarray: The zero-padded array with centralized padding.
    """
    # Calculate the padding needed for each axis
    pad_y = (target_size[0] - original_array.shape[0]) // 2
    pad_x = (target_size[1] - original_array.shape[1]) // 2
    
    # Apply centralized padding
    padded_array = np.pad(original_array, 
                          ((pad_y, target_size[0] - original_array.shape[0] - pad_y), 
                           (pad_x, target_size[1] - original_array.shape[1] - pad_x)),
                          mode='constant')
    
    return padded_array


def get_subarray(array, piece_number):
    """
    Divide a 2D array into 16 pieces and return the specified piece.

    Parameters:
    array (ndarray): The 2D input array (shape should be (n, n), where n is a power of 2)
    piece_number (int): The piece number (0 to 15) of the subarray to return

    Returns:
    ndarray: The corresponding subarray
    """
    # Check that the piece_number is within range
    if piece_number < 0 or piece_number > 15:
        raise ValueError("piece_number must be between 0 and 15")

    # Get the size of the original array
    size = array.shape[0]
    
    # Check if the size is a power of 2
    if not (size & (size - 1) == 0 and size != 0):
        raise ValueError("The array's length must be a power of 2")
    
    # Define the size of each subarray
    sub_size = size // 4
    
    # Get the row and column indices for the piece
    row_idx = piece_number // 4
    col_idx = piece_number % 4
    
    # Calculate the bounds for the subarray
    row_start = row_idx * sub_size
    row_end = row_start + sub_size
    col_start = col_idx * sub_size
    col_end = col_start + sub_size
    
    # Return the corresponding subarray
    return array[row_start:row_end, col_start:col_end]

def stitch_subarrays(subarrays):
    """
    Stitch 16 subarrays into a single 2D array, following the same number coding 
    used in the get_subarray function.

    Parameters:
    subarrays (list of ndarray): A list of 16 subarrays in the correct sequence (0 to 15)

    Returns:
    ndarray: The stitched 2D array
    """
    # Check that exactly 16 subarrays are provided
    if len(subarrays) != 16:
        raise ValueError("A list of 16 subarrays is required")

    # Get the size of each subarray
    sub_size = subarrays[0].shape[0]

    # Ensure all subarrays have the same size
    for subarray in subarrays:
        if subarray.shape != (sub_size, sub_size):
            raise ValueError("All subarrays must have the same dimensions")
    
    # Create an empty array to hold the final stitched result
    stitched_size = sub_size * 4
    stitched_array = np.zeros((stitched_size, stitched_size), dtype=subarrays[0].dtype)
    
    # Place each subarray in its correct location
    for piece_number in range(16):
        # Determine row and column position based on the piece number
        row_idx = piece_number // 4
        col_idx = piece_number % 4
        
        # Calculate where to place the subarray in the stitched array
        row_start = row_idx * sub_size
        row_end = row_start + sub_size
        col_start = col_idx * sub_size
        col_end = col_start + sub_size
        
        # Assign the subarray to the appropriate section of the stitched array
        stitched_array[row_start:row_end, col_start:col_end] = subarrays[piece_number]
    
    return stitched_array

def stitch_patches_row_major(patches, num_rows, num_cols):
    """
    Stitch a list of 2D arrays (patches) into a larger 2D array in row-major order.

    Args:
    - patches (list of np.ndarray): List of 2D subpatches to be stitched together.
    - num_rows (int): The number of subpatches along the vertical (row) direction.
    - num_cols (int): The number of subpatches along the horizontal (column) direction.

    Returns:
    - stitched_array (np.ndarray): The stitched 2D array.
    """
    # Assume all patches are of the same size
    patch_height, patch_width = patches[0].shape

    # Create an empty array to hold the stitched result
    stitched_array = np.zeros((num_rows * patch_height, num_cols * patch_width))

    # Iterate over each subpatch and place it in the correct position
    for i, patch in enumerate(patches):
        row_idx = i // num_cols  # Determine the row of the patch
        col_idx = i % num_cols   # Determine the column of the patch

        # Compute the position in the larger array
        start_row = row_idx * patch_height
        start_col = col_idx * patch_width

        # Place the patch in the correct position in the large array
        stitched_array[start_row:start_row + patch_height, start_col:start_col + patch_width] = patch

    return stitched_array


def zero_pad_image(image, target_size):
    """
    Zero-pad an image to the specified target size, centering the original image.

    Parameters:
        image (np.ndarray): The input 2D image array.
        target_size (list or tuple): Target size [s1, s2] as [height, width].

    Returns:
        np.ndarray: The zero-padded image.
    """
    # Ensure the target size is valid
    if len(target_size) != 2:
        raise ValueError("Target size must be a list or tuple with two elements [s1, s2].")
    
    s1, s2 = target_size
    h, w = image.shape

    if s1 < h or s2 < w:
        raise ValueError("Target size must be greater than or equal to the image size.")

    # Create a zero-padded array
    padded_image = np.zeros((s1, s2), dtype=image.dtype)

    # Calculate the offsets for centering the image
    offset_h = (s1 - h) // 2
    offset_w = (s2 - w) // 2

    # Insert the original image into the center of the padded array
    padded_image[offset_h:offset_h + h, offset_w:offset_w + w] = image

    return padded_image

def mat_img_load(filepath,keywords_slice = None,return_loc=False):
    files_org = [file for file in os.listdir(filepath) if file.endswith('.mat') and not file.startswith('._')]
    img_list = []
    #print(files)
    files = []
    if keywords_slice is not None:
        for word in keywords_slice:
            for file in files_org:
                if file[10:13] == word:
                    files.append(file)
        files_pd = pd.DataFrame(files)
        files_pd.to_csv(f'slice_names_{keywords_slice[0]}_to_{keywords_slice[-1]}.csv')
    else:
        files = files_org
    locs = []
    for file in files:
        hyphen_indices = find_hyphen_indices_loop(file)
        dot_indices = find_dot_indices_loop(file)
        v = int(file[7:9])
        s = int(file[int(hyphen_indices[0]-3):int(hyphen_indices[0])])
        x = int(file[int(hyphen_indices[0]+1):int(hyphen_indices[1])])
        y = int(file[int(hyphen_indices[1]+1):int(dot_indices[0])])

        fp = os.path.join(filepath,file)
        
        with h5py.File(fp, 'r') as mat_data:
        # List all datasets in the file
            keys = list(mat_data.keys())
            key = keys[0]
            # Access a specific dataset (replace 'your_dataset' with the actual dataset name)
            data1 = np.array(mat_data[key])#.reshape((513,512,512))
        # Print the keys of the dictionary to see the structure of the data
        #datax = data1[0]
        #data = data1#.reshape((513,512,512))
        data = data1.view(np.complex128)#[100:400,100:400,:]
        #data = np.transpose(data, (2,0,1))
        img_list.append(data)
        locs.append({'v': v, 's': s, 'x': loc_converter(x), 'y': loc_converter(y),'data':data})
    if return_loc:
        return locs, img_list
    return img_list


def mat_load(filepath):
    img_list = []
    files = [file for file in os.listdir(filepath) if file.endswith('.mat')]
    for file in files:
        fp = os.path.join(filepath, file)
        with h5py.File(fp, 'r') as mat_data:
            keys = list(mat_data.keys())
            key = keys[0]
            data1 = np.array(mat_data[key])
        data = data1.view(np.complex128)
        img_list.append(data)
    return img_list

def stitch_patches_with_overlap(patches, img_height, img_width, patch_size, step_size):
    stitched_img = np.zeros((img_height, img_width))
    weight_matrix = np.zeros((img_height, img_width))
    
    for patch, x_start, y_start in patches:
        x_end = x_start + patch_size
        y_end = y_start + patch_size
        stitched_img[x_start:x_end, y_start:y_end] += patch
        weight_matrix[x_start:x_end, y_start:y_end] += 1
    
    # Normalize by weights to account for overlaps
    stitched_img /= np.maximum(weight_matrix, 1)
    return stitched_img

def save_images_to_mat(image_array, output_filename):
    """
    Save a 3D NumPy array to a MATLAB .mat file.
    
    Args:
        image_array (numpy.ndarray): 3D array of shape (num_images, height, width).
        output_filename (str): Path to save the .mat file.
    """
    assert len(image_array.shape) == 3, "Input array must be 3D (num_images, height, width)."
    savemat(output_filename, {'image_sequence': image_array})

def loc_converter(In):
    if In == 1:
        return 0
    elif In == 129:
        return 1
    elif In ==257:
        return 2
    elif In == 385:
        return 3
    else:
        return None

def find_hyphen_indices_loop(s):
    indices = []
    for index, char in enumerate(s):
        if char == '-':
            indices.append(index)
    return indices
def find_dot_indices_loop(s):
    indices = []
    for index, char in enumerate(s):
        if char == '.':
            indices.append(index)
    return indices


def image_quality_Q(image, f_low=0.3, f_high=0.4, eps=1e-12):
    """
    Compute the Q metric for image quality based on Fourier power spectrum.

    Parameters
    ----------
    image : 2D numpy array
        Input grayscale image.
    f_low : float
        Cutoff for low-frequency band (cycles/pixel, relative to Nyquist = 0.5).
    f_high : float
        Cutoff for high-frequency band (cycles/pixel, relative to Nyquist = 0.5).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    Q : float
        Quality metric = (power in high-frequency band) / (power in low-frequency band).
    """

    # Ensure float image
    img = np.asarray(image, dtype=float)
    h, w = img.shape

    # Apply FFT and shift zero frequency to center
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2   # Power spectrum

    # Frequency coordinates (cycles per pixel)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)  # Radial frequency map
    #print(R)
    # Create masks for frequency bands
    low_mask = (R > 0) & (R <= f_low)
    high_mask = (R >= f_high) & (R <= 0.5)  # Nyquist = 0.5

    # Compute power in each band
    low_power = P[low_mask].sum()
    high_power = P[high_mask].sum()

    # Compute Q metric
    Q = high_power / (low_power + eps)
    return Q

from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
def register_images(images,upsample_factor = 20):
    """
    Register a list of images to the first one using DFT-based phase correlation.

    Parameters
    ----------
    images : list of 2D numpy arrays
        Input images.

    Returns
    -------
    registered : list of 2D numpy arrays
        Registered images (same order, first one unchanged).
    shifts : list of tuple
        (row_shift, col_shift) for each image relative to the first.
    """
    ref = images[0]
    registered = [ref]
    shifts = [(0.0, 0.0)]

    for img in images[1:]:
        # Compute shift with subpixel precision
        shift, error, diffphase = phase_cross_correlation(ref, img, upsample_factor=upsample_factor)

        # Apply shift in Fourier space (avoids interpolation artifacts)
        shifted = np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)).real

        registered.append(shifted)
        shifts.append(shift)

    return registered

