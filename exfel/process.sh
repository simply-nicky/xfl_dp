#!/bin/bash

source /etc/profile.d/modules.sh
module load anaconda/3

MODULE_PATH=$1
shift

cd $MODULE_PATH
python -m exfel $*