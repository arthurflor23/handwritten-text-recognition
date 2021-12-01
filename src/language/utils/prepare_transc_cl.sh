#!/bin/bash
set -e
export LC_ALL=en_US.UTF-8

[ $# -lt 4 ] && {
    echo "Usage: ${0##*/} <gt-Dir> <gt-fileExt> <ou-tDir> <tagsLst1> [<tagsLst2> ...]" >&2;
    exit 1;
}

GT_DIR="$1"
GT_EXT="$2"
W_DIR="$3"
shift 3

mkdir -p ${W_DIR}
while [ "$1" != "" ]; do cat "$1"; shift 1; done | \
while read tag; do
    echo -n "${tag} "
    head -n 1 ${GT_DIR}/${tag}.${GT_EXT} | tr -d '\n' | sed 's/$/\n/'
done > ${W_DIR}/text
