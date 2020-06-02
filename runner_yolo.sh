now=$(date +"day_%m_%d_%Y_time_%H_%M_%S")
echo "CURDATETIME: ${now}"
export CUDA_VISIBLE_DEVICES="auto"
source auto_gpu.sh
echo "${CUDA_VISIBLE_DEVICES}"
python main_yolo.py --resolution=320 --bs=64 \
--jitter=30 --do_flip --mean_var_clip --rand_brightness --rand_contrast --random_erase \
--path="./yoloResults/test/quicktest/${now}" \
--train_txt_path="/tmp/coco/ngc_5k.txt" \
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
--seeds="0,0,0" \
--display_every=100 --init_scale=0.28661 --init_bias=0.48853 --fp16 # --cache_batch_stats 
# "/dataset/coco/ngc_5k.txt"
# "models/yolo/5k_fullpath.txt"
