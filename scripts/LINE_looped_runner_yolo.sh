STARTLINE=$1 
ENDLINE=$2 

export CUDA_VISIBLE_DEVICES=$3
echo "Running on GPU: $CUDA_VISIBLE_DEVICES from [ $STARTLINE , $ENDLINE )"

CURLINE=$STARTLINE
CURENDLINE=0
RESOLUTION_BS="416 32\n448 32\n480 16\n512 16\n544 16\n576 16\n608 16"

while [ $CURLINE -lt  $ENDLINE ]
do
    # choose resolution and batch-size combination
    resbatch=$( shuf <(echo -e $RESOLUTION_BS) | head -n1 )
    resolution=$( echo $resbatch | cut -c 1-3 )
    batchsize=$( echo $resbatch | cut -c 5- )

    # CURLINE, CURENDLINE
    CURENDLINE=$( expr $CURLINE + $batchsize )
    if [ $CURENDLINE -gt $ENDLINE ]
    then
        CURENDLINE=$ENDLINE
        batchsize=$( expr $CURENDLINE - $CURLINE )
    fi 
    echo "lines: [$CURLINE - $CURENDLINE ) | batchsize: $batchsize | resolution: $resolution"
    
    # extract subset ngc_5k lines [$CURLINE - $CURENDLINE) 
    # randstring=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    SUBSETFILE="subset_${CURLINE}_${CURENDLINE}_bs${batchsize}_res${resolution}.txt"
    OUTDIR="subset_${CURLINE}_${CURENDLINE}_bs${batchsize}_res${resolution}"
    cat /tmp/coco/ngc_train_82k.txt | head -n $( expr $CURENDLINE - 1 ) | tail -n $batchsize > /tmp/$SUBSETFILE 

    # Check that number of lines in file == batchsize 
    nlines=$( cat /tmp/$SUBSETFILE | wc -l )
    if [ $nlines -ne $batchsize ]
    then
        echo "Note: bs:${batchsize} doesn't match nlines:$nlines" 
    fi

    python main_yolo.py --resolution=${resolution} --bs=${batchsize} \
    --jitter=30 --do_flip --mean_var_clip --rand_brightness --rand_contrast --random_erase \
    --path="/result/$OUTDIR" \
    --train_txt_path="/tmp/$SUBSETFILE" \
    --iterations=4000 \
    --r_feature=0.002 \
    --first_bn_coef=10.0 \
    --main_loss_multiplier=1.0 \
    --alpha_img_stats=0.0 \
    --tv_l1=100.0 \
    --tv_l2=0.0 \
    --lr=0.015 \
    --wd=0 \
    --save_every=100 \
    --seeds="0,0,123456" \
    --display_every=100 --init_scale=0.28661 --init_bias=0.48853 --fp16 > /dev/null

    mv /tmp/$SUBSETFILE /result/$OUTDIR 
    cat /result/$OUTDIR/losses.log | grep "\[VERIFIER\] Real image mAP"
    cat /result/$OUTDIR/losses.log | grep "mAP VERIFIER" | tail -n1

    # loop increment
    CURLINE=$CURENDLINE
done 
