#!/bin/bash
set -e
export LC_NUMERIC=C

[ $# -lt 6 ] && {
    echo "Usage: ${0##*/} <PstMatricesTrain-Dir> <PstMatrices-FileExt> <out-Dir> <charPriors-File> <alpha> <tagLst1-File> [<tagLst2-File> ...]"
    exit 1
}

pm_dir="$1";
pmFile_ext="$2";
out_dir="$3";
priors_file=$4
ALPHA="$5"
shift 5;

[ -d $pm_dir ] || { echo "ERROR: Dir \"$pm_dir\" does not exist !" 1>&2; exit 1; }
[ -f $priors_file ] || { echo "ERROR: File \"$priors_file\" does not exist !" 1>&2; exit 1; }
[ -z "$ALPHA" ] && { echo "ERROR: \"ALPHA\" is not set !" 1>&2; exit 1; }

mkdir -p ${out_dir}
bdir=$(echo "${pm_dir}" | xargs basename)"_alp$ALPHA"
scp_file="${out_dir}/$bdir.scp";
ark_file="${out_dir}/$bdir.ark";

echo "Computing log-likelihood matrices ..." 1>&2
a=$ALPHA
while [ "$1" != "" ]; do cat "${1}"; shift 1; done | \
while read tag; do
    echo -n -e "Processing ${tag}\r" 1>&2;
    echo -n "${tag} [ "
    awk -v alpha=$a -v chPrs="$priors_file" '
        BEGIN{while (getline < chPrs > 0) P[$1]=$5+0.0}
        {
          for (i=1;i<=NF;i++) {
            lgLkh=(P[i]==0)?-743.747:log($i)-log(P[i])*alpha;
          printf("%f ",lgLkh); 
        } 
        print""
        }' ${pm_dir}/${tag}.${pmFile_ext}
    echo "]"
done | copy-matrix 'ark,t:-' "ark,scp:${ark_file},${scp_file}"
echo "" 1>&2

exit 0
