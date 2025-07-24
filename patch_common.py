#!/usr/bin/env python3
"""
The above functions provide utilities for image preprocessing including resizing, cropping, and
color conversion.
:return: The `imagePreprocess` function returns the preprocessed image after running it through the
`imagePreprocessFromCoords` function, which includes resizing, cropping, and converting the image
based on the provided parameters such as scale, width, height, crop offsets, number of planes, and
color space.
"""
import cv2
import numpy as np


"""
Copyright © 2025 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

This file contains configuration utilities and definitions for patch processing.
"""


def expectedImageProcKeys():
    return [
        "scale",
        "width",
        "height",
        "crop_x_offset",
        "crop_y_offset",
        "frames_per_sample",
        "format",
    ]


def getCropCoords(
    src_width, src_height, scale, width, height, crop_x_offset, crop_y_offset
):
    """
    Compute the scaled dimensions and crop rectangle (left, top, right, bottom)
    entirely in the scaled-image space.
    """
    # 1) scaled full-image size
    scaled_w = int(round(src_width * scale))
    scaled_h = int(round(src_height * scale))

    # 2) center + offsets
    cx = scaled_w // 2 + crop_x_offset
    cy = scaled_h // 2 + crop_y_offset

    # 3) crop rectangle
    left = max(cx - width // 2, 0)
    top = max(cy - height // 2, 0)
    right = min(left + width, scaled_w)
    bottom = min(top + height, scaled_h)

    return scaled_w, scaled_h, (left, top, right, bottom)


def imagePreprocessFromCoords(
    image,
    scale_w,
    scale_h,
    crop_coords,
    out_width,
    out_height,
    planes=1,
    src="BGR",
    interp=cv2.INTER_CUBIC,
):
    """
    1) Resize the entire image to (scale_w, scale_h)
    2) Crop with integer coords in that space
    3) Resize the crop to (out_width, out_height)
    4) Convert to gray or RGB as requested
    """
    # 1) resize full image
    scaled = cv2.resize(image, (scale_w, scale_h), interpolation=interp)

    # 2) crop
    left, top, right, bottom = crop_coords
    cropped = scaled[top:bottom, left:right]

    # 3) final resize (in case we clipped at edges)
    patch = cv2.resize(cropped, (out_width, out_height), interpolation=interp)

    if planes == 1:
        # if it’s already single-channel, just return it:
        if patch.ndim == 2:
            return patch
        # if it has a singleton channel-dimension, squeeze:
        if patch.ndim == 3 and patch.shape[2] == 1:
            return patch[:, :, 0]
        # otherwise assume 3-channel BGR/RGB
        if src == "BGR" and patch.shape[2] == 3:
            return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if src == "RGB" and patch.shape[2] == 3:
            return cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        # fallback
        return patch.mean(axis=2).astype(patch.dtype)

    elif planes == 3:
        # color
        if src == "BGR":
            return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        else:
            # assume patch is already RGB
            return patch

    else:
        raise RuntimeError(f"Unhandled number of planes: {planes}")


def imagePreprocess(
    image, scale, width, height, crop_x_offset, crop_y_offset, planes=1, src="BGR"
):
    """
    Full pipeline:
      • compute scaled dims + crop box
      • run through imagePreprocessFromCoords
    """
    h, w = image.shape[:2]
    scale_w, scale_h, crop_coords = getCropCoords(
        w, h, scale, width, height, crop_x_offset, crop_y_offset
    )
    return imagePreprocessFromCoords(
        image,
        scale_w,
        scale_h,
        crop_coords,
        width,
        height,
        planes,
        src,
        interp=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
