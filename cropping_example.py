#!/usr/bin/env python3
"""
Example of how to use the updated SamplerFunctions.py with patch_common cropping functionality.

This demonstrates how to call sample_video with various cropping parameters.
"""

import pandas as pd
from SamplerFunctions import sample_video

# Example usage of the updated sample_video function with cropping
def example_video_sampling_with_cropping():
    """
    Example showing how to use the cropping functionality in SamplerFunctions.py
    """
    
    # Example parameters - adjust these to your needs
    video_input_path = "/path/to/your/videos"
    dataset_input_path = "/path/to/your/dataset"
    out_path = "/path/to/output"
    video_filename = "example_video.mp4"
    
    # Create example dataframe - replace with your actual dataframe
    # DataFrame should contain columns: data_file, class, begin frame, end frame
    example_df = pd.DataFrame({
        'data_file': ['example.csv'],
        'class': [1],
        ' begin frame': [0],
        ' end frame': [1000]
    })
    
    # Example 1: Basic sampling without cropping
    print("Example 1: Basic sampling without cropping")
    sample_video(
        video_input_path=video_input_path,
        dataset_input_path=dataset_input_path,
        out_path=out_path,
        video=video_filename,
        old_df=example_df,
        number_of_samples_max=10,
        frames_per_sample=5,
        normalize=True,
        out_channels=1,  # Grayscale
        sample_span=30,
        crop=False
    )
    
    # Example 2: Cropping with specific output dimensions
    print("Example 2: Cropping with specific output dimensions")
    sample_video(
        video_input_path=video_input_path,
        dataset_input_path=dataset_input_path,
        out_path=out_path,
        video=video_filename,
        old_df=example_df,
        number_of_samples_max=10,
        frames_per_sample=5,
        normalize=True,
        out_channels=1,  # Grayscale
        sample_span=30,
        crop=True,
        out_width=400,     # Crop to 400x300 pixels
        out_height=300,
        scale=1.2,         # Scale up by 20% before cropping
        x_offset=50,       # Shift crop region 50 pixels right
        y_offset=-25       # Shift crop region 25 pixels up
    )
    
    # Example 3: RGB cropping with different parameters
    print("Example 3: RGB cropping with different parameters")
    sample_video(
        video_input_path=video_input_path,
        dataset_input_path=dataset_input_path,
        out_path=out_path,
        video=video_filename,
        old_df=example_df,
        number_of_samples_max=5,
        frames_per_sample=3,
        normalize=False,
        out_channels=3,    # RGB
        sample_span=60,
        crop=True,
        out_width=224,     # Common size for neural networks
        out_height=224,
        scale=0.8,         # Scale down by 20%
        x_offset=0,        # Center crop
        y_offset=0
    )

def cropping_parameter_explanation():
    """
    Explanation of the cropping parameters and how they work together.
    """
    print("""
    Cropping Parameter Explanation:
    
    1. crop (bool): Enable/disable cropping functionality
       - False: No cropping, use original frame size
       - True: Apply cropping with the specified parameters
    
    2. scale (float): Scale factor applied to the original frame before cropping
       - 1.0: No scaling (default)
       - >1.0: Upscale (e.g., 1.2 = 120% of original size)
       - <1.0: Downscale (e.g., 0.8 = 80% of original size)
    
    3. out_width, out_height (int): Final dimensions of the cropped frame
       - These define the size of the output frame
       - If None, uses scaled dimensions
    
    4. x_offset, y_offset (int): Offset from center for crop region
       - 0, 0: Center crop (default)
       - Positive values: Move crop region right/down
       - Negative values: Move crop region left/up
    
    Processing Order:
    1. Frame is loaded from video
    2. Normalization and contrast/brightness adjustments applied
    3. If cropping enabled:
       a. Frame is conceptually scaled by 'scale' factor
       b. Crop region is calculated based on out_width/out_height and offsets
       c. Frame is cropped and resized to final dimensions
    4. Frame is converted to tensor format
    
    The patch_common.imagePreprocess function handles the scaling, cropping,
    and color space conversion efficiently in a single operation.
    """)

if __name__ == "__main__":
    cropping_parameter_explanation()
    # Uncomment the following line to run the examples (after setting correct paths)
    # example_video_sampling_with_cropping()
