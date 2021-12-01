#!/bin/bash

set -e
export LC_NUMERIC=C
NAME=${0##*/}

if [ $# -ne 8 ]; then
  echo "Usage: $NAME <GroundTruthTrain-Dir> <PstMatricesTrain-Dir> <PstMatricesDevel-Dir> <PstMatrices-FileExt> <GroundTruth-FileExt> <CharList-File> <blankSymb-Char> <alpha>" 1>&2
  exit 1
fi

CURR_DIR=`pwd`
TMP=${CURR_DIR}/${0##*/}_$$; mkdir $TMP
trap "rm -rf $TMP 2>/dev/null" EXIT

TRAINGT=$1
TRAINPM=$2
DEVELPM=$3
EXT_PM_FILE=$4
EXT_GT_FILE=$5
CHARSF=$6
BLKSYM="$7"
ALPHA="$8"

[ -d $TRAINGT ] || { echo "ERROR: Dir \"$TRAINGT\" does not exist \!" 1>&2; exit 1; }
[ -d $TRAINPM ] || { echo "ERROR: Dir \"$TRAINPM\" does not exist \!" 1>&2; exit 1; }
[ -d $DEVELPM ] || { echo "ERROR: Dir \"$DEVELPM\" does not exist \!" 1>&2; exit 1; }
[ -f $CHARSF ] || { echo "ERROR: File \"$CHARSF\" does not exist \!" 1>&2; exit 1; }
[ -z "$BLKSYM" ] && { echo "ERROR: \"BLKSYM\" is not set \!" 1>&2; exit 1; }
[ -z "$ALPHA" ] && { echo "ERROR: \"ALPHA\" is not set \!" 1>&2; exit 1; }

# Make force alignment
if [ ! -f forceAlign_train.dat ]; then
  echo "Making force alignment on the training set ..." 1>&2
  force-alignmrnt-ext.awk $TRAINGT ${EXT_PM_FILE} ${EXT_GT_FILE} $CHARSF $BLKSYM $TRAINPM/*.${EXT_PM_FILE} > forceAlign_train.dat
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
     }' forceAlign_train.dat > Prior.dat || { echo "ERROR: character priors" 1>&2; exit 1; }
fi

# Compute Log Likelihood Matrices
a=$ALPHA
ODIR=${DEVELPM##*/}_alp${a};
if [ ! -d "$ODIR" ]; then
  echo "Computing log-likelihood matrices ..." 1>&2
  mkdir $ODIR;
  for f in $DEVELPM/*.${EXT_PM_FILE}; do
    F=$(basename $f .${EXT_PM_FILE});
    echo -n -e "Processing $F\r" 1>&2;
    awk -v alpha=$a '
      BEGIN{while (getline < "Prior.dat" > 0) P[$1]=$5+0.0}
      {
        for (i=1;i<=NF;i++) {
	  lgLkh=(P[i]==0)?-743.747:log($i)-log(P[i])*alpha;
	  printf("%f ",lgLkh); 
	} 
	print""
      }' $f > $ODIR/$F.fea;
  done;
  echo "" 1>&2
fi

exit 0
