export PYTHONPATH=/home/user/chanho/HandAvatar:/home/user/chanho/HandAvatar/leap:$PYTHONPATH
CUDA_VISIBLE_DEVICES=3 NVIDIA_TF32_OVERRIDE=0 python handavatar/train.py \
    --cfg 'handavatar/configs/interhand/val_cap0.yaml'
