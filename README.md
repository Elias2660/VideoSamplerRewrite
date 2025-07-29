# Video Sampler Rewrite

This project works in conjunction with the [Unified-bee-Runner](https://github.com/Elias2660/Unified-bee-Runner) and uses Python 3.8+.

## Installation

1. Clone the repository:

   ```sh  
   git clone https://github.com/Elias2660/VideoSamplerRewrite.git  
   cd VideoSamplerRewrite  
   ```


2. Install the required packages:

   ```sh  
   pip install -r requirements.txt  
   ```
   $$

## Usage

### Prepare Dataset

To sample video data and prepare the dataset, run:

```sh  
cd ..  # change to the working directory  
python Dataprep.py \
  --video-input-path <video_dir> \
  --dataset-input-path <csv_dir> \
  --out-path <output_dir> \
  --dataset-search-string "dataset_*.csv" \
  --number-of-samples <max_samples> \
  --max-workers <workers> \
  --frames-per-sample <frames_per_sample> \
  [--normalize True|False] \
  [--out-channels <channels>] \
  [--crop] \
  [--x-offset <px>] \
  [--y-offset <px>] \
  [--out-width <px>] \
  [--out-height <px>] \
  [--equalize-samples] \
  [--dataset-writing-batch-size <size>] \
  [--max-threads-pic-saving <threads>] \
  [--max-workers-tar-writing <workers>] \
  [--max-batch-size-sampling <size>] \
  [--debug]  
```

### Write Data

To write sampled frames into a WebDataset TAR, run:

```sh  
python WriteToDataset.py \
  <png_root> \
  <tar_file> \
  <dataset_path> \
  [--frames_per_sample <frames_per_sample>] \
  [--out_channels <channels>] \
  [--batch_size <batch_size>] \
  [--equalize] \
  [--max_workers <workers>]  
```

### Using sbatch

For sbatch execution, edit the provided settings in the script and run from the data directory:

```sh  
sbatch -x /[servers-currently-active] VideoSamplerRewrite/RunDataPrep.sh  
```

## Contributing

[Contributions](CONTRIBUTING.md) are welcome! Please follow the guidelines in SECURITY.md and ensure compliance with the project's license.

## License

This project is licensed under the [MIT License](LICENSE).

## Security

Please review our [Security Policy](SECURITY.md) for guidelines on reporting vulnerabilities.
$$
