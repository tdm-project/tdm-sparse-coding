#!/bin/bash

#===============================================================================

if [ $# -ne 4 ]; then
    echo Wrong parameter count: Usage test_volume_sparse_codec.sh VOLUME_CODEC_APP INPUT_DIR INPUT_FILE OUTPUT_DIR 
    exit
fi

VICCOMPRESS=$1
INPUT_DIR=$2
INPUT_FILE=$3
STATS_DIR=$4

#===============================================================================

for ROCKDVR_CONDITION in "-K 8" "-a 8" "-T 0.01"
do
    for ROCKDVR_DICT in 2048
    do
	for ROCKDVR_ALLOC in fixed greedygrow
	do
	    for ROCKDVR_CORESET in 16
	    do
                for QUANTIZATION in none covra q-esc q-mtv
                do
		    VICARGS="-I 30 -C $ROCKDVR_CORESET -D $ROCKDVR_DICT -P 4 -B 6 -b 6 -Q 1 -e $ROCKDVR_ALLOC -q $QUANTIZATION $ROCKDVR_CONDITION"
		    VICARGSTRING=`echo "$VICARGS" | tr ' \.' '_'  | tr -d '\-'`
		    
		    VICOUTPUT_FILE=$INPUT_FILE"_vic_"$VICARGSTRING
		    VICOUTPUT_FILE_STATS=$VICOUTPUT_FILE".stats"
		    VICOUTPUT_FILE_LOG=$VICOUTPUT_FILE".log"
		    
		    echo "================================================="    
		    echo "-------- Running " $VICCOMPRESS $VICARGS " on " $INPUT_FILE
		    
		    echo "########### STATS FOR " $INPUT_FILE " ######" > $STATS_DIR/$VICOUTPUT_FILE_STATS
		    echo "########### LOG FOR " $INPUT_FILE " ######" > $STATS_DIR/$VICOUTPUT_FILE_LOG
		    
		    echo "#" $VICARGS $INPUT_DIR/$INPUT_FILE | tee -a $STATS_DIR/$VICOUTPUT_FILE_STATS
                    $VICCOMPRESS $VICARGS $INPUT_DIR/$INPUT_FILE 2>>$STATS_DIR/$VICOUTPUT_FILE_LOG | tee -a $STATS_DIR/$VICOUTPUT_FILE_STATS

                    RET_VAL=$?
                    echo
                    if [ $RET_VAL -eq 0 ]; then
	                echo OK
                    else
                        echo =========== FAILED ==============
                    fi
		done
	    done
	done
    done
done
