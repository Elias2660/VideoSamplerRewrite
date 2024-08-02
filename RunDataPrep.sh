#!/bin/bash
export OMP_NUM_THREADS=1

<<comment
RunDataPrep.sh

This script prepares the data for model training by executing the Dataprep.py script with specified options.

Usage:
    ./RunDataPrep.sh

Options:
    --max-workers <number-of-workers> : Specifies the maximum number of worker threads to use for data preparation. Default is 15.

Environment Variables:
    OMP_NUM_THREADS : Sets the number of threads to use for parallel regions. Default is 1.

Logging:
    The script logs the output of the data preparation process to dataprep.log.

Example:
    To run the script with the default settings:
        ./RunDataPrep.sh

    To run the script with a custom number of worker threads:
        python VideoSamplerRewrite/Dataprep.py --max-workers 10 >> dataprep.log 2>&1

Dependencies:
    - Python 3.x
    - Required Python packages listed in requirements.txt

Notes:
    - Ensure that the Dataprep.py script and the required dependencies are available and properly configured.
    - The script should be executed from the directory containing the VideoSamplerRewrite folder.
comment


# This script is used to prepare the data for the model training
python VideoSamplerRewrite/Dataprep.py --max-workers 15 >> dataprep.log 2>&1