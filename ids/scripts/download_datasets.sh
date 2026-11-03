#!/usr/bin/env bash
# Placeholder: fetch evaluation datasets into ids/data/ (git-ignored).
# CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
# NSL-KDD:     https://www.unb.ca/cic/datasets/nsl.html
set -euo pipefail
mkdir -p "$(dirname "$0")/../data"
echo "Download the datasets listed above into ids/data/ manually; licences require a form."
