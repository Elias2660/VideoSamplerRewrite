#!/bin/bash

# This script is used to prepare the data for the model training
python Dataprep.py --max-workers 40 >> dataprep.log 2>&1