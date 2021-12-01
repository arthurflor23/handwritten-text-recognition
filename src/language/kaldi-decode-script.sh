#!/bin/bash

# ------------------------------------------------------------------------------------------------------
# Compile and install Kaldi with SRILM and OpenBLAS
# ------------------------------------------------------------------------------------------------------
# Download:
#	Kaldi: https://github.com/kaldi-asr/kaldi
#   SRILM: http://www.speech.sri.com/projects/srilm/download.html

# Extract files:
#   unzip kaldi-*.zip && mv kaldi-master kaldi
#	cp srilm-*.tar.gz kaldi/tools/srilm.tgz

# Compile Kaldi with SRILM and OpenBLAS (needs ``python2 sox subversion gcc-fortran`` packages):
#   cd ./kaldi/tools
#   ./install_srilm.sh <user> <organization> <email>
#   ./extras/install_openblas.sh
#   make -j $(nproc)

#   cd ../src
#   ./configure --shared --mathlib=OPENBLAS
#   make -j clean depend
#   make -j $(nproc)
# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------
# Setting parameters
# ------------------------------------------------------------------------------------------------------
ROOT_DIR=$(pwd)/$(dirname "$0")

# Default path (root dir)
if [ -z $1 ]; then TARGET_DIR=${ROOT_DIR}; else TARGET_DIR=$1; fi

# set mode action
if [ -z $2 ]; then ACTION='TEST'; else ACTION=$2; fi

# Remove folder if exist and run in train mode
if [ -d "$TARGET_DIR" ] && [ "$ACTION" == "TRAIN" ]; then rm -Rf $TARGET_DIR/data; fi

# N-Gram Language Model Order
if [ -z $3 ]; then NGRAM_ORDER=2; else NGRAM_ORDER=$3; fi
# ======================================================================================================


# ------------------------------------------------------------------------------------------------------
# Settings and Environment Variables
# ------------------------------------------------------------------------------------------------------
# Data settings
# ------------------------------------------------------------------------------------------------------
DEVEL_PM_DIR=conf_mats.ark            # Directory of Validation Confidence Matrices
GT_FILE=ground_truth.lst		      # File containing ground-truth in Kaldi format
TRAIN_ID_LST=ID_train.lst		      # List of line IDs of Training set
DEVEL_ID_LST=ID_test.lst		      # List of line IDs of Validation set
CHRS_LST=chars.lst                    # File containing the list of chars
# ------------------------------------------------------------------------------------------------------
# Special symbols
# ------------------------------------------------------------------------------------------------------
BLANK_SYMB="<ctc>"                    # BLSTM non-character symbol
WHITESPACE_SYMB="<space>"             # White space symbol
DUMMY_CHAR="<DUMMY>"                  # Especial HMM used for modelling "</s>" end-sentence
# ------------------------------------------------------------------------------------------------------
# Feature processing settings
# ------------------------------------------------------------------------------------------------------
LOGLKH_ALPHA_FACTOR=0.3               # p(x|s) = P(s|x) / P(s)^LOGLKH_ALPHA_FACTOR
# ------------------------------------------------------------------------------------------------------
# Modelling settings
# ------------------------------------------------------------------------------------------------------
HMM_LOOP_PROB=0.1   		          # Self-Loop HMM-state probability
HMM_NAC_PROB=0.5			          # BLSTM-NaC HMM-state probability
GSF=1.0 	 			  		      # Grammar Scale Factor
WIP=-1.0		 			  	      # Word Insertion Penalty
ASF=1.0					  		      # Acoustic Scale Factor
# ------------------------------------------------------------------------------------------------------
# Decoding settings
# ------------------------------------------------------------------------------------------------------
MAX_NUM_ACT_STATES=2007483647		  # Maximum number of active states
BEAM_SEARCH=30						  # Beam search
LATTICE_BEAM=30					  	  # Lattice generation beam
# ------------------------------------------------------------------------------------------------------
# System settings
# ------------------------------------------------------------------------------------------------------
N_CORES=$(nproc)					  # Number of cores
# ======================================================================================================


# Check for required files
# ======================================================================================================
[ -f ${TARGET_DIR}/$DEVEL_PM_DIR ] || 
{
	echo "No file: ${TARGET_DIR}/${DEVEL_PM_DIR}"
	exit 1
}
[ -f ${TARGET_DIR}/$GT_FILE ] || 
{
	echo "No file: ${TARGET_DIR}/${GT_FILE}"
	exit 1
}
[ -f ${TARGET_DIR}/$TRAIN_ID_LST ] || 
{
	echo "No file: ${TARGET_DIR}/${TRAIN_ID_LST}"
	exit 1
}
[ -f ${TARGET_DIR}/$DEVEL_ID_LST ] || 
{
	echo "No file: ${TARGET_DIR}/${DEVEL_ID_LST}"
	exit 1
}
[ -f ${TARGET_DIR}/$CHRS_LST ] || 
{
	echo "No file: ${TARGET_DIR}/${CHRS_LST}"
	exit 1
}
# ======================================================================================================


# Export PATH
# ======================================================================================================
# SRILM and KALDI Stuff
export KALDI_ROOT=$ROOT_DIR/kaldi
export LIBLBFGS=${KALDI_ROOT}/tools/liblbfgs-1.10 && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:${LIBLBFGS}/lib/.libs && export SRILM=${KALDI_ROOT}/tools/srilm && export PATH=${PATH}:${SRILM}/bin:${SRILM}/bin/i686-m64 && export PATH=${KALDI_ROOT}/tools/python:${PATH} && export PATH=$PATH:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$ROOT_DIR/utils && export PATH
######################################################################################################


# Splitting reference transcripts according to the train and devel ID lists
######################################################################################################
[ -f ${TARGET_DIR}/data/train/text ] ||
{
	echo "Processing training transcripts into Kaldi format ..." 1>&2
	[ -d ${TARGET_DIR}/data/train/ ] || mkdir -p ${TARGET_DIR}/data/train/
	awk -v idf="${TARGET_DIR}/$TRAIN_ID_LST" 'BEGIN{while (getline < idf > 0) IDs[$1]=""} {if ($1 in IDs) print}' ${TARGET_DIR}/$GT_FILE > ${TARGET_DIR}/data/train/text
}
[ -f ${TARGET_DIR}/data/test/text ] ||
{
	echo "Processing development transcripts into Kaldi format ..." 1>&2
	[ -d ${TARGET_DIR}/data/test/ ] || mkdir -p ${TARGET_DIR}/data/test/
	awk -v idf="${TARGET_DIR}/$DEVEL_ID_LST" 'BEGIN{while (getline < idf > 0) IDs[$1]=""} {if ($1 in IDs) print}' ${TARGET_DIR}/$GT_FILE > ${TARGET_DIR}/data/test/text
}
######################################################################################################


# Processing development feature samples into Kaldi format
######################################################################################################
[ -e ${TARGET_DIR}/data/test/${DEVEL_PM_DIR}_alp${LOGLKH_ALPHA_FACTOR}.scp ] ||
{
	echo "Processing development samples into Kaldi format ..." 1>&2
	mkdir -p ${TARGET_DIR}/data/test
	copy-matrix "ark:${TARGET_DIR}/${DEVEL_PM_DIR}" "ark,scp:${TARGET_DIR}/data/test/${DEVEL_PM_DIR}_alp${LOGLKH_ALPHA_FACTOR}.ark,${TARGET_DIR}/data/test/${DEVEL_PM_DIR}_alp${LOGLKH_ALPHA_FACTOR}.scp"
}
######################################################################################################


# Prepare Kaldi's lang directories
######################################################################################################
# Preparing Lexic (L)
[ -d ${TARGET_DIR}/data/train/lang ] ||
{
	echo "Generating lexic model ..." 1>&2
	prepare_lang_cl-ds.sh ${TARGET_DIR}/data/train ${TARGET_DIR}/${CHRS_LST} "${BLANK_SYMB}" "${WHITESPACE_SYMB}" "${DUMMY_CHAR}"
}
######################################################################################################


# Preparing LM (G)
[ -f ${TARGET_DIR}/data/train/lang/LM.arpa ] ||
{
	echo "Generating ${NGRAM_ORDER} character-level language model ..." 1>&2
	cat ${TARGET_DIR}/data/train/text | cut -d " " -f 2- |
	ngram-count -text - -lm ${TARGET_DIR}/data/train/lang/LM.arpa -order ${NGRAM_ORDER} -kndiscount8 -kndiscount7 -kndiscount6 -kndiscount5 -kndiscount4 -kndiscount3 -kndiscount2 -interpolate
	prepare_lang_test-ds.sh ${TARGET_DIR}/data/train/lang/LM.arpa ${TARGET_DIR}/data/train/lang ${TARGET_DIR}/data/train/lang_test "$DUMMY_CHAR"
}
######################################################################################################


# Prepare HMM models
######################################################################################################
# Create HMM topology file
[ -d ${TARGET_DIR}/data/train/hmm ] ||
{
	echo "Creating character HMM topologies ..." 1>&2
	mkdir -p ${TARGET_DIR}/data/train/hmm
	phones_list=( $(cat ${TARGET_DIR}/data/train/lang_test/phones/{,non}silence.int) )
	featdim=$(feat-to-dim scp:${TARGET_DIR}/data/test/${DEVEL_PM_DIR}_alp${LOGLKH_ALPHA_FACTOR}.scp - 2>/dev/null)
	dummyID=$(awk -v d="$DUMMY_CHAR" '{if (d==$1) print $2}' ${TARGET_DIR}/data/train/lang/phones.txt)
	blankID=$(awk -v bs="${BLANK_SYMB}" '{if (bs==$1) print $2}' ${TARGET_DIR}/data/train/lang/pdf_blank.txt)
	create_proto_rnn-ds.sh $featdim ${HMM_LOOP_PROB} ${HMM_NAC_PROB} ${TARGET_DIR}/data/train/hmm ${dummyID} ${blankID} ${phones_list[@]}
}
######################################################################################################


# Compose FSTs
######################################################################################################
[ -d ${TARGET_DIR}/data/test/hmm ] ||
{
	echo "Creating global SFS automaton for decoding ..." 1>&2
	mkdir -p ${TARGET_DIR}/data/test/hmm
	mkgraph.sh --transition-scale 1.0 --self-loop-scale 1.0 ${TARGET_DIR}/data/train/lang_test ${TARGET_DIR}/data/train/hmm/new.mdl ${TARGET_DIR}/data/train/hmm/new.tree ${TARGET_DIR}/data/test/hmm/graph
}
######################################################################################################


# Lattice Generation
######################################################################################################
[ -f ${TARGET_DIR}/data/lat.gz ] ||
{
	echo "Generating lattices ..." 1>&2
	split -d -n l/${N_CORES} -a 3 ${TARGET_DIR}/data/test/${DEVEL_PM_DIR}_alp${LOGLKH_ALPHA_FACTOR}.scp ${TARGET_DIR}/data/part-
	mkdir -p ${TARGET_DIR}/data/lattices
	for n in $(seq -f "%03.0f" 0 1 $[N_CORES-1]); do
		echo "launching subprocess in core $n ..." 1>&2
		latgen-faster-mapped --verbose=2 --allow-partial=true --acoustic-scale=${ASF} --max-active=${MAX_NUM_ACT_STATES} --beam=${BEAM_SEARCH} --lattice-beam=${LATTICE_BEAM} ${TARGET_DIR}/data/train/hmm/new.mdl ${TARGET_DIR}/data/test/hmm/graph/HCLG.fst scp:${TARGET_DIR}/data/part-$n "ark:|gzip -c > ${TARGET_DIR}/data/lattices/lat_$n.gz" ark,t:${TARGET_DIR}/data/lattices/RES_$n 2>${TARGET_DIR}/data/lattices/LOG-Lats-$n &
	done
	echo "Waiting for finalization of the ${N_CORES} subprocesses ..." 1>&2
	wait
	lattice-copy "ark:gunzip -c ${TARGET_DIR}/data/lattices/lat_*.gz |" "ark:|gzip -c > ${TARGET_DIR}/data/lat.gz"
	rm -rf ${TARGET_DIR}/data/lattices/ ${TARGET_DIR}/data/part-*
}
######################################################################################################


# Final Evaluation
######################################################################################################
echo "Computing ..." 1>&2
score.sh --wip $WIP --lmw $ASF ${TARGET_DIR}/data/test/hmm/graph/words.txt "ark:gzip -c -d ${TARGET_DIR}/data/lat.gz |" ${TARGET_DIR}/data/test/text ${TARGET_DIR}/data/predicts 2>${TARGET_DIR}/data/log
echo -e "\nGenerating file of predicts: predicts_t" 1>&2
int2sym.pl -f 2- ${TARGET_DIR}/data/test/hmm/graph/words.txt ${TARGET_DIR}/data/predicts > ${TARGET_DIR}/data/predicts_t
######################################################################################################
