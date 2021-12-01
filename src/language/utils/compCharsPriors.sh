#!/bin/bash

set -e
export LC_NUMERIC=C
NAME=${0##*/}

if [ $# -ne 5 ]; then
  echo "Usage: $NAME <GroundTruthTrain-File> <PstMatricesTrain-Dir> <PstMatrices-FileExt> <CharList-File> <blankSymb-Char>" 1>&2
  exit 1
fi

CURR_DIR=`pwd`
TMP=${CURR_DIR}/${0##*/}_$$; mkdir $TMP
trap "rm -rf $TMP 2>/dev/null" EXIT

TRAINGT=$1
TRAINPM=$2
EXT_PM_FILE=$3
CHARSF=$4
BLKSYM="$5"

[ -f $TRAINGT ] || { echo "ERROR: File \"$TRAINGT\" does not exist \!" 1>&2; exit 1; }
[ -d $TRAINPM ] || { echo "ERROR: Dir \"$TRAINPM\" does not exist \!" 1>&2; exit 1; }
[ -f $CHARSF ] || { echo "ERROR: File \"$CHARSF\" does not exist \!" 1>&2; exit 1; }
[ -z "$BLKSYM" ] && { echo "ERROR: \"BLKSYM\" is not set !" 1>&2; exit 1; }

# Make force alignment
if [ ! -f forceAlign_train.dat ]; then
  echo "Making force alignment on the training set ..." 1>&2
  force-alignmrnt-ext.awk $TRAINGT ${EXT_PM_FILE} $CHARSF $BLKSYM $TRAINPM/*.${EXT_PM_FILE} > forceAlign_train.dat
fi

# Compute character priors
if [ ! -f Prior.dat ]; then
  echo "Computing character priors" 1>&2
  awk -v cf="$CHARSF" '
     BEGIN{ 
        tot=0; cnt=0;
        while (getline < cf > 0) {
          C[++cnt]=$1; F[$1]=0
        } 
     }{ 
        tot+=NF;
	for (l=1;l<=NF;l++) F[$l]++;
     }END{ 
        for (i=1;i<=cnt;i++) 
	  printf("%d\t%s\t%d\t%d\t%.10e\n",i,C[i],F[C[i]],tot,F[C[i]]/tot) 
     }' forceAlign_train.dat > Priors.dat || { echo "ERROR: character priors" 1>&2; exit 1; }
fi

exit 0
