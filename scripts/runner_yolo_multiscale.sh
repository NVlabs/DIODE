now=$(date +"day_%m_%d_%Y_time_%H_%M_%S")
echo "CURDATETIME: ${now}"
export CUDA_VISIBLE_DEVICES="auto"
source scripts/auto_gpu.sh
echo "${CUDA_VISIBLE_DEVICES}"

##############################################################
# This command generates 128 images of resolution 160x160
##############################################################

rootlocation="/akshayws/finalcodetest/"
python -u main_yolo.py --resolution=160 --bs=128 \
--jitter=20 --do_flip --rand_brightness --rand_contrast --random_erase \
--path="${rootlocation}/${now}_res160" \
--train_txt_path="/tmp/onebox/manifest.txt" \
--iterations=2500 \
--r_feature=0.1 --p_norm=2 --alpha-mean=1.0 --alpha-var=1.0 --num_layers=-1 \
--first_bn_coef=2.0 \
--main_loss_multiplier=0.5 \
--alpha_img_stats=0.0 \
--tv_l1=75.0 \
--tv_l2=0.0 \
--lr=0.2 --min_lr=0.0 --beta1=0.0 --beta2=0.0 \
--wd=0.0 \
--save_every=100 --display_every=100 \
--seeds="0,0,23456" \
--init_scale=1.0 --init_bias=0.0 --nms_conf_thres=0.05 --alpha-ssim=0.00 --save-coco

##############################################################
# to enable fp-sampling add the following args to above:
# --box-sampler --box-sampler-warmup=800 --box-sampler-conf=0.2 \
# --box-sampler-overlap-iou=0.35 --box-sampler-minarea=0.01 --box-sampler-maxarea=0.85 --box-sampler-earlyexit=2800
##############################################################

##############################################################
# Uncomment the command block below to upsample and fine-tune 
# images generated by previous command to resolution 320x320
##############################################################

# # create manifest file
# find ${rootlocation}/${now}_res160/coco/images -name '*.png' > ${rootlocation}/${now}_res160/coco/manifest.txt 
# # upsample images from 160 --> 320 
# # WARNING: imagemagick must be installed before the next line is run
# # to install imagemagick: $sudo apt-get update && sudo apt-get install -y imagemagick 
# find ${rootlocation}/${now}_res160/coco/images -name '*.png' | xargs -I {} convert {} -resize 320x320 {} 

# python main_yolo.py --resolution=320 --bs=96 \
# --jitter=40 --do_flip --rand_brightness --rand_contrast --random_erase \
# --path="${rootlocation}/${now}_res320" \
# --train_txt_path="${rootlocation}/${now}_res160/coco/manifest.txt" \
# --iterations=1500 \
# --r_feature=0.1 --p_norm=2 --alpha-mean=1.0 --alpha-var=1.0 --num_layers=51 \
# --first_bn_coef=0.0 \
# --main_loss_multiplier=1.0 \
# --alpha_img_stats=0.0 \
# --tv_l1=75.0 \
# --tv_l2=0.0 \
# --lr=0.002 --min_lr=0.0005 \
# --wd=0.0 \
# --save_every=100 \
# --seeds="0,0,23456" \
# --display_every=100 --init_scale=1.0 --init_bias=0.0 --nms_conf_thres=0.1 --alpha-ssim=0.0 --save-coco --real_mixin_alpha=1.0