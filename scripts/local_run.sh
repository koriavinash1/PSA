CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name SSAU_zp_gauss --model SSAU --hps clevr_hans3 --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name SSAU_zp_learngauss --learn_prior --model SSAU --hps clevr_hans3 --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name SSAU_zp_gmm --learn_prior --zprior gmm --model SSAU --hps clevr_hans3 --ckpt_dir ../checkpoints_rgb_correction &

CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name SSA_zp_gauss --model SSA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name SSA_zp_learngauss --learn_prior --model SSA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name SSA_zp_gmm --learn_prior --zprior gmm --model SSA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &

CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name VASA_zp_gauss --model VASA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name VASA_zp_learngauss --learn_prior --model VASA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name VASA_zp_gmm --learn_prior --zprior gmm --model VASA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &

CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name VSA_zp_gauss --model VSA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name VSA_zp_learngauss --learn_prior --model VASA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name VSA_zp_gmm --learn_prior --zprior gmm --model VSA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &

CUDA_VISIBLE_DEVICES=0 python ../main.py --exp_name ASA --model ASA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &
CUDA_VISIBLE_DEVICES=1 python ../main.py --exp_name SA --model SA --hps clevr --ckpt_dir ../checkpoints_rgb_correction &



