#!/bin/bash
set -eu
cd /root/autodl-tmp/lrr/code2-recovery-20261003
while [ ! -f /root/autodl-tmp/lrr/ParamAgent/models/Qwen3.5-2B/download_provenance.json ]; do
 if ! pgrep -f '[p]ython -u /root/autodl-tmp/lrr/code2-recovery-20261003/download.py' >/dev/null; then
  echo 'DOWNLOAD_STOPPED_WITHOUT_COMPLETE'; exit 1
 fi
 sleep 15
done
exec /root/autodl-tmp/lrr/code2-merge-env/bin/python -u merge.py
