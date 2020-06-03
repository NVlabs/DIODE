if [ "${CUDA_VISIBLE_DEVICES}" = "auto" ]
then
    # number of gpus 
    NUMGPUS=`nvidia-smi -q -d MEMORY | grep "Attached GPU" | grep -P -o "\d"`
    echo "NUMGPUS: $NUMGPUS"
    
    # extract free-memory for each gpu
    MEMLIST="ID FREEMEM"
    for (( DEVICE=0; DEVICE<${NUMGPUS}; DEVICE++ ))
    do
        echo "RUNNING for GPU: ${DEVICE}"
        FREEMEM=`nvidia-smi -q -d MEMORY -i ${DEVICE} | grep "Free" | head -n1 | grep -E -o "[0-9]+"`
        MEMLIST="${MEMLIST}\n${DEVICE} ${FREEMEM}"
    done
    echo "####################"
    echo -e $MEMLIST
    echo "####################"

    # MEMLIST --> remove first line --> sort on gpumem --> pick first line --> pick first GPU device-id 
    export CUDA_VISIBLE_DEVICES=`echo -e ${MEMLIST} | tail -n +2 | sort -n -r -k2 | head -n1 | grep -E -o "^[0-9]"`

fi 
echo "CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
