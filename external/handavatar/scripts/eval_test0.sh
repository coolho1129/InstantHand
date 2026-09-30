export PYTHONPATH=/home/user/chanho/HandAvatar:/home/user/chanho/HandAvatar/leap:$PYTHONPATH
CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 python handavatar/train.py \
    --cfg 'handavatar/configs/interhand/test_cap0.yaml'




CUDA_VISIBLE_DEVICES=2 NVIDIA_TF32_OVERRIDE=0 python handavatar/train.py \
    --cfg '/home/user/chanho/LAM_hand/handavatar/configs/hanco/191_scale1.yaml'