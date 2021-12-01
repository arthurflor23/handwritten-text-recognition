#!/usr/bin/gawk -f

#awk '

function traceBackPath(i,j, aux) {
    split(Tb[i,j],aux," ");
    #print aux[1],aux[2],aux[3]
    #return;
    if (aux[1]==0 && aux[2]==0) {
	#printf("%s",aux[3]);
	printf("%s ",aux[3]);
    }
    else {
	traceBackPath(aux[1],aux[2]);
	#printf("%s",aux[3]);
	printf("%s ",aux[3]);
    }
    return;
}

BEGIN{
    if (ARGC < 6) {
 	print "USAGE: force-alignmrnt.awk <GT-File> <PstMatrices-FileExt> <Chars-File> <BLANK-Symb> <file1> <file2> ..." > "/dev/stderr";
	exit 1;
    }
    #print ARGC,ARGV[0],ARGV[1],ARGV[2],ARGV[3],ARGV[4] > "/dev/stderr"
    gtF=ARGV[1]; delete ARGV[1];
    ePM=ARGV[2]; delete ARGV[2];
    chrsF=ARGV[3]; delete ARGV[3];
    blkSym=ARGV[4]; delete ARGV[4];

    while (getline < gtF > 0) {
      id=$1; nwrds[id]=NF-1;
      for (i=2;i<=NF;i++) wrds[id,i-1]=$i
    }
    if (ERRNO) {
      print "ERROR: "gtF": "ERRNO > "/dev/stderr";
      exit 1;
    }

    cnt=0; bBLKSymbFound=0
    while (getline < chrsF > 0) {
	chr=$1;
	C[++cnt]=chr; I[chr]=cnt;
	if (chr==blkSym) bBLKSymbFound=1;
    }
    if (ERRNO) {
	print "ERROR: "chrsF": "ERRNO > "/dev/stderr";
	exit 1;
    }
    if (!bBLKSymbFound) {
	print "ERROR: BLANK Symbol \""blkSym"\" has not been found !" > "/dev/stderr";
	exit 1;
    }
    lzero = -1000;
}

BEGINFILE{
    idL=gensub(/^.*\//,"","g",FILENAME);
    eAUX="."ePM"$"
    sub(eAUX,"",idL);
    if (!(idL in nwrds)) {
      print "WARNNING: "idL" is not in "gtF > "/dev/stderr";
      nextfile;
    }

    Len=nwrds[idL]; cnt=0;
    for (i=1; i<=Len; i++) {
	Gt[++cnt]=I[blkSym]; 
	Gt[++cnt]=I[wrds[idL,i]];
    }
    Gt[++cnt]=I[blkSym]; 
    Len=cnt;
    #print idL,Len;
    #for (l=1;l<=Len;l++) print l,Gt[l], C[Gt[l]];
    printf("Force aligning file: %s\r",FILENAME) > "/dev/stderr"
    #printf(".") > "/dev/stderr"
}

{
    if (FNR==1) {
	Vt[1,FNR] = log($(Gt[1]));
	Tb[1,FNR] = 0" "0" "C[Gt[1]];
	Vt[2,FNR] = log($(Gt[2]));
	Tb[2,FNR] = 0" "0" "C[Gt[2]];
	for (i=3; i<=Len; i++) {
	    #Vt[i,FNR] = Vt[i-1,FNR] + log($(Gt[i]));
	    Vt[i,FNR] = lzero;
	    #Tb[i,FNR] = i-1" "FNR" "C[Gt[i]];
	    Tb[i,FNR] = 0" "0" "C[Gt[i]];
	}
    } else {
	Vt[1,FNR] = Vt[1,FNR-1] + log($(Gt[1]));
	Tb[1,FNR] = "1 "FNR-1" "C[Gt[1]];
	for (i=2; i<=Len; i++) {
	    if (Vt[i-1,FNR-1] > Vt[i,FNR-1]) {
		Vt[i,FNR] = Vt[i-1,FNR-1];
		Tb[i,FNR] = i-1" "FNR-1" "C[Gt[i]];
	    } else {
		Vt[i,FNR] = Vt[i,FNR-1];
		Tb[i,FNR] = i" "FNR-1" "C[Gt[i]];
	    }
	    if ((i%2==0 && i>3) && Vt[i-2,FNR-1] > Vt[i,FNR] && Gt[i-2] != Gt[i]) {
                Vt[i,FNR] = Vt[i-2,FNR-1];
	        Tb[i,FNR] = i-2" "FNR-1" "C[Gt[i]];
	    }
	    Vt[i,FNR] += log($(Gt[i]));
	}
    }
}

ENDFILE{
    #print "\nFile:"FILENAME,"#Frms:"FNR,"\nGTF:"gtF,"GT:"gtL,"\nMxScr:"Vt[Len,FNR];
    #printf("%s ",idL);
    cnt=0;
    if ((Len>1) && (Vt[Len-1,FNR] > Vt[Len,FNR]))
      traceBackPath(Len-1,FNR);
    else
      traceBackPath(Len,FNR);
    print ""; fflush(); close(FILENAME);
    delete Gt; delete Vt; delete Tb;
    gtF=""; gtL=""; Len=0;
}

END{
    printf("\n") > "/dev/stderr"
}

#' confmats_train/096_014_002_7144.txt
