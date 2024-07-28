# Data Sampler Rewrite

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

To prepare the dataset, run:

```sh
cd .. # into the working directory
python Dataprep.py --dataset_path <path-to-dataset> --dataset_name <dataset-name> --number_of_samples_max <max-samples> --max_workers <number-of-workers> --frames_per_sample <frames-per-sample>To write data
```

to write data to the dataset, run:

```sh
python WriteToDataset.py
```

To run in sbatch, go out to the data directory and run: 
```sh
sbatch -x /[servers that are currently running] VideoSamplerRewrite/RunDataPrep.sh
```
Edit that file with the recommended settings.