CUDA_VISIBLE_DEVICES=0 python main.py --exp_name HAE_s8 --z_max_res 1 --slot_arch '8s7' &
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name HAE_8s7_32s2 --z_max_res 1 --slot_arch '8s7,32s2' &

CUDA_VISIBLE_DEVICES=0 python main.py --exp_name HVAE_s8 --slot_arch '8s7' &
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name HVAE_8s7_32s2 --slot_arch '8s7,32s2' &

CUDA_VISIBLE_DEVICES=0 python main.py --exp_name variational_s8 --variational_slots --slot_arch '8s7' &
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name variational_8s7_32s2 --variational_slots --slot_arch '8s7,32s2' &

CUDA_VISIBLE_DEVICES=0 python main.py --exp_name qcorrection_variational_s8 --qcorrection --variational_slots --slot_arch '8s7' &
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name qcorrection_variational_s32 --qcorrection --variational_slots --slot_arch '32s7' &
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name qcorrection_variational_8s7_32s2 --qcorrection --variational_slots --slot_arch '8s7,32s2' &
