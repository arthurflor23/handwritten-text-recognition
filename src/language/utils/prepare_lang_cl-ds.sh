#!/bin/bash
set -e

export LC_ALL=en_US.UTF-8

[ $# -lt 3 ] && {
    echo "Usage: ${0##*/} <work-dir> <charMap-File> <blank-symb> [<whiteSpace-symb>] [<Dummy-symb>]" >&2;
    exit 1;
}

W_DIR="$1"
CM_FILE="$2"
BLANK="$3"
shift 3
if [ $# -gt 0 ]; then WSPACE=$1; shift 1; fi
if [ $# -gt 0 ]; then DUMMY=$1; shift 1; fi
WSPACE=${WSPACE:="<space>"}
DUMMY=${DUMMY:="<DUMMY>"}
#echo $W_DIR $CM_FILE $WSPACE $DUMMY; exit

srcdir=${W_DIR}/local/dict
tmpdir=${W_DIR}/local/lang
dir=${W_DIR}/lang

rm -rf ${srcdir} ${tmpdir} ${dir} ${dir}/phones
mkdir -p ${srcdir} ${tmpdir} ${dir} ${dir}/phones

# Create lexicon with probabilities
#awk '{ p=substr($1,1,1); print p"\t1.0\t"p }' ${CM_FILE} > ${srcdir}/lexiconp.txt || exit 1
awk -v bs="$BLANK" -v ws="$WSPACE" -v dm="$DUMMY" '
    BEGIN{dflg=0; sflg=0}
    { p=$1; if (p==dm) dflg=1; if (p==ws) sflg=1; if (p!=bs) print p"\t1.0\t"p }
    END{if (sflg==0) {print ws" is not in the vocabulary!" > "/dev/stderr"; exit 1;}
        if (dflg==0) print dm"\t1.0\t"dm}' ${CM_FILE} > ${srcdir}/lexiconp.txt || exit 1

# Create temporal pdf-BLANK file
awk -v bs="$BLANK" '
    BEGIN{flg=0}
    {if ($1==bs) flg=NR-1}
    END{print bs"\t"flg}' ${CM_FILE} > ${srcdir}/pdf_blank.txt || exit 1

# Create temporal phones lists
echo -e "$WSPACE" > ${srcdir}/silence_phones.txt
awk -v bs="$BLANK" -v ws="$WSPACE" -v dm="$DUMMY" '
    BEGIN{flg=0}
    {if ($1==dm) flg=1; if (ws!=$1 && bs!=$1) print $1}
    END{if (flg==0) print dm}' ${CM_FILE} > ${srcdir}/nonsilence_phones.txt || exit 1
cp ${srcdir}/silence_phones.txt ${dir}/phones/silence.txt
cp ${srcdir}/nonsilence_phones.txt ${dir}/phones/nonsilence.txt
cp ${srcdir}/pdf_blank.txt ${dir}/
#cat ${srcdir}/{,non}silence_phones.txt > ${srcdir}/phones.txt
awk '{print $1}' ${srcdir}/lexiconp.txt > ${srcdir}/phones.txt

# Add disambiguation symbols to the lexicon
ndisambig=$(add_lex_disambig.pl --pron-probs ${srcdir}/lexiconp.txt ${srcdir}/lexiconp_disambig.txt)
#ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence disambiguation words in lexicon FST.
echo $ndisambig > ${srcdir}/lex_ndisambig

for n in $(seq 0 $ndisambig); do echo "#$n"; done > ${dir}/phones/disambig.txt

# Create phones.txt table
echo "<eps>" | cat - ${srcdir}/phones.txt ${dir}/phones/disambig.txt | \
    awk '{n=NR - 1; print $1, n;}' > ${dir}/phones.txt

# Create words.txt table
awk '{print $1}' ${srcdir}/lexiconp.txt | sort | uniq | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" > "/dev/stderr"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" > "/dev/stderr"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > ${dir}/words.txt || exit 1;

# Create lexicon FST
# Create the basic L.fst without disambiguation symbols, for use
# in training.
#utils/make_lexicon_fst.pl --pron-probs ${srcdir}/lexiconp.txt 0.5 0x20 | \
#    fstcompile --isymbols=${dir}/phones.txt --osymbols=${dir}/words.txt \
#    --keep_isymbols=false --keep_osymbols=false | \
#    fstarcsort --sort_type=olabel > ${dir}/L.fst || exit 1;

# Create the lexicon FST with disambiguation symbols.
# There is an extra step where we create a loop to "pass through" the
# disambiguation symbols from G.fst.
# See http://vpanayotov.blogspot.com.ar/2012/06/kaldi-decoding-graph-construction.html
# See also: http://www.gavo.t.u-tokyo.ac.jp/~novakj/wfst-algorithms.pdf
# http://stackoverflow.com/questions/2649474/how-to-perform-fst-finite-state-transducer-composition
phone_disambig_symbol=$(grep \#0 ${dir}/phones.txt | awk '{print $2}')
word_disambig_symbol=$(grep \#0 ${dir}/words.txt | awk '{print $2}')
make_lexicon_fst.pl \
    --pron-probs ${srcdir}/lexiconp_disambig.txt | \
    fstcompile --isymbols=${dir}/phones.txt --osymbols=${dir}/words.txt \
    --keep_isymbols=false --keep_osymbols=false |   \
    fstaddselfloops  \
    "echo ${phone_disambig_symbol} |" "echo ${word_disambig_symbol} |" | \
    fstarcsort --sort_type=olabel > ${dir}/L_disambig.fst || exit 1;

# Prepare final lists of phones
for f in silence nonsilence disambig; do
    sym2int.pl ${dir}/phones.txt < ${dir}/phones/${f}.txt \
        > ${dir}/phones/${f}.int
    sym2int.pl ${dir}/phones.txt < ${dir}/phones/${f}.txt | \
        awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// \
        > ${dir}/phones/${f}.csl
done

exit 0
