"""
src/preprocessing.py
--------------------
Handles array downsampling, image normalization, histogram extraction,
and intelligent preprocessing for user-uploaded clothing photos.
"""

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = 14
HISTOGRAM_BINS = 16


def preprocess_array(images):
    """
    Convert a batch of 28x28 grayscale images to normalized 14x14 images.
    Takes every second pixel (subsampling by factor of 2) and normalizes to [0, 1].
    """
    images = np.asarray(images, dtype=np.float32)
    return images[:, ::2, ::2] / 255.0


def preprocess_uploaded_image(
    uploaded_file,
    invert_background: str = "Auto",
    auto_center: bool = True,
    noise_threshold: float = 0.05
):
    """
    Intelligently prepares any uploaded clothing image for Fashion-MNIST classification.

    Key Requirements of Fashion-MNIST:
      1. Background must be BLACK (pixel value 0.0).
      2. Clothing item must be LIGHT / WHITE (pixel values > 0.0).
      3. Object must be centered in a 28x28 frame.
      4. Subsampled to 14x14 resolution.

    Args:
        uploaded_file: Uploaded file buffer or PIL Image.
        invert_background: 'Auto', 'Yes' (force invert), or 'No' (keep original).
        auto_center: If True, crops around the clothing bounding box and centers it.
        noise_threshold: Values below this threshold are zeroed out to remove shadows.

    Returns:
        tuple: (array_14x14, display_28x28, was_inverted)
            array_14x14: Normalized float32 numpy array of shape (14, 14) in range [0, 1]
            display_28x28: PIL Image of the centered 28x28 intermediate image
            was_inverted: Boolean indicating whether color inversion was applied
    """
    if not isinstance(uploaded_file, Image.Image):
        image = Image.open(uploaded_file)
    else:
        image = uploaded_file

    # Step 1: Convert to Grayscale
    gray_img = image.convert("L")
    raw_arr = np.asarray(gray_img, dtype=np.float32)

    # Step 2: Detect whether background is Light or Dark
    # We sample the 1-pixel border along all four edges of the image
    edge_pixels = np.concatenate([
        raw_arr[0, :],      # Top row
        raw_arr[-1, :],     # Bottom row
        raw_arr[:, 0],      # Left col
        raw_arr[:, -1]      # Right col
    ])
    edge_mean = float(np.mean(edge_pixels))
    is_light_bg = edge_mean > 127.0

    # Determine inversion
    if invert_background == "Auto":
        should_invert = is_light_bg
    elif invert_background == "Yes":
        should_invert = True
    else:
        should_invert = False

    if should_invert:
        gray_img = ImageOps.invert(gray_img)

    # Step 3: Auto-contrast to enhance clothing edges against background
    gray_img = ImageOps.autocontrast(gray_img, cutoff=2)
    arr_255 = np.asarray(gray_img, dtype=np.float32)

    # Step 4: Clean subtle background shadows/noise
    threshold_val = noise_threshold * 255.0
    arr_255[arr_255 < threshold_val] = 0.0

    # Step 5: Bounding-box detection and Centering (matching Fashion-MNIST distribution)
    if auto_center:
        active_coords = np.argwhere(arr_255 > 25.0)
        if len(active_coords) > 20:
            ymin, xmin = active_coords.min(axis=0)
            ymax, xmax = active_coords.max(axis=0)

            # Crop tightly to the clothing object
            cropped = Image.fromarray(arr_255.astype(np.uint8)).crop((xmin, ymin, xmax + 1, ymax + 1))
            cw, ch = cropped.size

            # Scale to fit comfortably inside a 22x22 box (leaving 3px margins in a 28x28 canvas)
            scale = 22.0 / max(cw, ch)
            nw = max(1, int(cw * scale))
            nh = max(1, int(ch * scale))

            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized_crop = cropped.resize((nw, nh), resampling)

            # Paste into center of 28x28 black canvas
            canvas = Image.new("L", (28, 28), color=0)
            canvas.paste(resized_crop, ((28 - nw) // 2, (28 - nh) // 2))
            centered_img = canvas
        else:
            centered_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        centered_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)

    # Step 6: Downsample 28x28 to 14x14 (subsampling by factor of 2, exactly like training set)
    centered_arr = np.asarray(centered_img, dtype=np.float32)
    array_14x14 = (centered_arr[::2, ::2] / 255.0).astype(np.float32)

    return array_14x14, centered_img, should_invert


def extract_histograms(images, bins=HISTOGRAM_BINS):
    """
    Return one normalized grayscale-intensity histogram per image.
    Range [0.0, 1.0] divided into 16 bins.
    """
    features = []
    for image in images:
        hist, _ = np.histogram(image, bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-10)
        features.append(hist)
    return np.asarray(features, dtype=np.float32)
