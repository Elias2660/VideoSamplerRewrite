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

## Usage

### Prepare Dataset

To sample video data and prepare the dataset, run:

```sh
cd ..  # change to the working directory
python Dataprep.py --dataset_path <path-to-dataset> --dataset_name <dataset-name> --number_of_samples_max <max-samples> --max_workers <number-of-workers> --frames_per_sample <frames-per-sample> [--crop] [--normalize]
```

_Optional parameters such as --crop and --normalize can be added as needed._

### Write Data

To write data to the dataset, run:

```sh
python WriteToDataset.py
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
