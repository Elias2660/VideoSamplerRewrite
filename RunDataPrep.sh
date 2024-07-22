#!/bin/bash
export OMP_NUM_THREADS=1

# This script is used to prepare the data for the model training
python VideoSamplerRewrite/Dataprep.py --max-workers 15 >> dataprep.log 2>&1