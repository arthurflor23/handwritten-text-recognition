#!/bin/bash

SOURCE=$(dirname ${BASH_SOURCE[0]})
wip=0.0
lmw=10.0

. $SOURCE/parse_options.sh || exit 1

if [ $# -ne 4 ]; then
    cat <<EOF
Usage: ${0##*/} [options] <word-sym-table> <lattice-rspec> <ref-txt> <hyp-txt>
Options:
  --wip <float>  word insertion penalty
  --lmw <float>  language model weight
EOF
    exit 1
fi

wrdsym="$1"
lattice="$2"
reftxt="$3"
hyptxt="$4"

mkdir -p $(dirname ${hyptxt})

lattice-scale --acoustic-scale=${lmw} "${lattice}" ark:- | \
    lattice-add-penalty --word-ins-penalty=${wip} ark:- ark:- | \
    lattice-best-path ark:- ark,t:${hyptxt} || exit 1;

cat ${hyptxt} | int2sym.pl -f 2- ${wrdsym} | \
    compute-wer --text --mode=present \
    ark:${reftxt} ark,p:- || exit 1;
