#!/bin/bash

# This script is used to prepare the data for the model training
python DataPrep.py --max-workers 40 >> DataPrep.log 2>&1