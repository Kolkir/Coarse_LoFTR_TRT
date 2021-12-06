#!/usr/bin/env bash

LOG_DIR=/home/kirill/development/models/LoFTR
# HOST=192.168.88.253
HOST=localhost

tensorboard --logdir $LOG_DIR --host $HOST
