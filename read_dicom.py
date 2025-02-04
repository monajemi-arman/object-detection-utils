import numpy as np
import pydicom
from PIL import Image

def read_dicom(file_path):
    dicom_data = pydicom.dcmread(file_path, force=True)
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    rescale_slope = float(dicom_data.get("RescaleSlope", 1))
    rescale_intercept = float(dicom_data.get("RescaleIntercept", 0))

    # Handle MultiValue for WindowCenter and WindowWidth
    window_center = dicom_data.get("WindowCenter", np.mean(pixel_array))
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = float(window_center[0])
    else:
        window_center = float(window_center)

    window_width = dicom_data.get("WindowWidth", np.max(pixel_array) - np.min(pixel_array))
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = float(window_width[0])
    else:
        window_width = float(window_width)

    hu_pixels = (pixel_array * rescale_slope) + rescale_intercept
    min_window = window_center - (window_width / 2)
    max_window = window_center + (window_width / 2)

    normalized = np.clip((hu_pixels - min_window) / (max_window - min_window), 0, 1)

    image_array = (normalized * 255).astype(np.uint8)

    image = Image.fromarray(image_array)

    return image