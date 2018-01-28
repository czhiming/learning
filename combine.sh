#!/bin/sh

python combine.py train.src.vector train.mt.vector train.vector

python combine.py test.src.vector test.mt.vector test.vector

python combine.py train.src-tgt.txt train.vector train.feature

python combine.py test.src-tgt.txt test.vector test.feature





