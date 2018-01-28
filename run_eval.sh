#!/bin/sh

python src/learn_model.py config/svr.cfg
./evaluateWMT.pl ref.csv 100 predicted.csv







