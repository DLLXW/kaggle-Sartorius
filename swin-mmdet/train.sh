#nohup python tools/train.py ./qyl_config/swin_t.py --gpu-ids 3 &
# CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh ./qyl_config/swin_s_coco.py
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh ./qyl_config/swin_s_coco.py 2