# Data Sampler Rewrite

## Installation

1. Clone the repository:

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To prepare the dataset, run:

```sh
python Dataprep.py --dataset_path <path-to-dataset> --dataset_name <dataset-name> --number_of_samples_max <max-samples> --max_workers <number-of-workers> --frames_per_sample <frames-per-sample>To write data
```

to write data to the dataset, run:

```sh
python WriteToDataset.py
```
