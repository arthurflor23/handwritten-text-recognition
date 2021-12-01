#!/bin/bash

[ $# -lt 7 ] && {
    echo "Usage: ${0##*/} <featDim> <LoopTransProb> <NaC-Prob> <dir-out> <dummyPhone> <blankPDF> <phonesList>" >&2
    exit 1
}

DIM=${1}; shift 1;
PrLoo=$(echo "scale=2; (${1})/1" | bc -l); shift 1;
PrNaC=$(echo "scale=2; (${1})/1" | bc -l); shift 1;
DIROUT=${1}; shift 1;
[ -d $DIROUT ] || mkdir -p $DIROUT
dmPhone=${1}; shift 1;
iblnkPDF=${1}; shift 1;

cad=""; na=$#; idmPhone=""; ns=0
for ((i=1;i<=na;i++)); do
  if [ $dmPhone -eq $1 ]; then idmPhone=$i;
  else cad=$cad" "$1; ns=$[ns+1];
  fi
  shift 1;
done
[ -z "$idmPhone" ] && { echo "ERROR: Dummy symbol \"$dmPhone\" has not been found"; exit 1; }

NPrLoo=$(echo "scale=2; (1 - $PrLoo)/1" | bc -l)
NPrNaC=$(echo "scale=2; (1 - $PrNaC)/1" | bc -l)

cat <<EOF > ${DIROUT}/topo
<Topology> 
<TopologyEntry> 
<ForPhones> 
${cad} 
</ForPhones> 
$(
  echo "<State> 0 <Transition> 1 0${PrNaC} <Transition> 2 0${NPrNaC} </State>"
  echo "<State> 1 <PdfClass>  0 <Transition> 1 0${PrLoo} <Transition> 2 0${NPrLoo} </State>"
  echo "<State> 2 <PdfClass>  1 <Transition> 2 0${PrLoo} <Transition> 3 0${NPrLoo} </State>"
  echo "<State> 3 </State>"
)
</TopologyEntry> 
<TopologyEntry> 
<ForPhones> 
${dmPhone}
</ForPhones> 
$(
  echo "<State> 0 <PdfClass>  0 <Transition> 0 0${PrLoo} <Transition> 1 0${NPrLoo} </State>"
  echo "<State> 1 </State>"
)
</TopologyEntry>
</Topology> 
EOF

gmm-init-mono --print-args=false ${DIROUT}/topo ${DIM} auxMdl auxTree

{ 
  echo "<TransitionModel> "
  cat ${DIROUT}/topo
  echo "<Triples> $[ns*2+1]";
  npdf=0
  for ((i=1;i<=$ns;i++)); do
    [ ${iblnkPDF} -eq $npdf ] && npdf=$[npdf+1]
    echo -e "$i 1 ${iblnkPDF}\n$i 2 $npdf";
    npdf=$[npdf+1]
  done;
  echo -e "$i 0 ${iblnkPDF}";
  echo "</Triples>"; 
  gmm-copy --print-args=false --binary=false auxMdl - 2>/dev/null |
  sed -n "/<LogProbs>/,/<\/LogProbs>/p"
  echo "</TransitionModel> "
} > ${DIROUT}/auxMDL

vv=$(perl -E "say ' 1' x ${DIM}")
{
echo -n "<DIMENSION> ${DIM} <NUMPDFS> ${DIM} "
for ((i=1;i<=DIM;i++)); do
cat << EOF
<DiagGMM> 
<GCONSTS>  [ -120.6098 ]
<WEIGHTS>  [ 1 ]
<MEANS_INVVARS>  [
  $vv ]
<INV_VARS>  [
  $vv ]
</DiagGMM> 
EOF
done
} >> ${DIROUT}/auxMDL
gmm-copy --print-args=false --binary=true ${DIROUT}/auxMDL ${DIROUT}/new.mdl 2>/dev/null

{ 
  echo "ContextDependency 1 0 ToPdf TE 0 $[ns+2] ( NULL";
  for ((i=0;i<=$ns;i++)); do
    [ ${iblnkPDF} -eq $i ] && continue
    echo "TE -1 2 ( CE ${iblnkPDF} CE $i )";
  done;
  echo "TE -1 1 ( CE ${iblnkPDF} )";
  echo -e ")\nEndContextDependency";
} > ${DIROUT}/auxTree
copy-tree --print-args=false --binary=true ${DIROUT}/auxTree ${DIROUT}/new.tree 2>/dev/null

rm auxMdl auxTree

gmm-info --print-args=false ${DIROUT}/new.mdl
tree-info --print-args=false ${DIROUT}/new.tree

