import math


def vidSamplingCommonCrop(
    height, width, out_height, out_width, scale, x_offset, y_offset
):
    """
    Return the common cropping parameters used in dataprep and annotations.

    Arguments:
        height     (int): Height of the video
        width      (int): Width of the video
        out_height (int): Height of the output patch
        out_width  (int): Width of the output patch
        scale    (float): Scale applied to the original video
        x_offset   (int): x offset of the crop (after scaling)
        y_offset   (int): y offset of the crop (after scaling)
    Returns:
        out_width, out_height, crop_x, crop_y
    """

    if out_width is None:
        out_width = math.floor(width * scale)
    if out_height is None:
        out_height = math.floor(height * scale)

    crop_x = math.floor((width * scale - out_width) / 2 + x_offset)
    crop_y = math.floor((height * scale - out_height) / 2 + y_offset)

    return out_width, out_height, crop_x, crop_y
