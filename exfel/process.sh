#!/bin/bash

source /etc/profile.d/modules.sh
module load anaconda/3

MODULE_PATH = $1
PACKAGE_NAME=$2
shift 2

cd $MODULE_PATH
python -m $PACKAGE_NAME $*