#!/bin/bash
set -eu
cd /root/autodl-tmp/lrr/qa-math-recovery-20261003
/root/autodl-tmp/lrr/code2-merge-env/bin/python -u merge_qa.py > qa.log 2>&1
/root/autodl-tmp/lrr/code2-merge-env/bin/python -u merge_math.py > math.log 2>&1
