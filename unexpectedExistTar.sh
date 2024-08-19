#!/bin/bash
export OMP_NUM_THREADS=1

python VideoSamplerRewrite/finishSamplerOnUnexpectedExit.py --max-workers 10 >> dataprep.log 2>&1