now=$(date +"day_%m_%d_%Y_time_%H_%M_%S")
echo "CURDATETIME: ${now}"
export CUDA_VISIBLE_DEVICES="auto"
source auto_gpu.sh
echo "${CUDA_VISIBLE_DEVICES}"
python main_yolo.py --bs=64 \
--jitter=10 --do_flip \
--path="./yoloResults/test_metrics/${now}" \
--iterations=8000 \
--r_feature=0.01 \
--first_bn_coef=0.0 \
--main_loss_multiplier=1.0 \
--tv_l1=0.0 \
--tv_l2=0.005 \
--lr=0.005 \
--wd=0.0000001 \
--save_every=100 \
--display_every=1 # --cache_batch_stats 
