# CUDA_VISIBLE_DEVICES=3 python train.py  \
# --configs defaults metaworld  \
# --task metaworld_button_press  \
# --logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press_compare_encout_viewpointcorrected1" \
# --run.from_offline_checkpoint '/data/ytzheng/log_icml25/checkpoints/offline_button_press_checkpoint.ckpt'  \
# --run.from_online_checkpoint '/data/ytzheng/log_icml25/metaworld/metaworld_button_press_s1/checkpoint.ckpt' \
# --run.script eval_only2


CUDA_VISIBLE_DEVICES=3 python train.py  \
--configs defaults metaworld  \
--task metaworld_button_press  \
--logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press_compare_online2seeds_offlineinput125" \
--run.from_offline_checkpoint '/data/ytzheng/log_icml25/metaworld/metaworld_button_press_s1/checkpoint.ckpt'  \
--run.from_online_checkpoint '/data/ytzheng/log_icml25/metaworld/metaworld_button_press_s0/checkpoint.ckpt' \
--run.script eval_only2 \
--run.offline_directory '/data/ytzheng/dataset/metaworld_200eps_501/button_press/eval_eps'

# CUDA_VISIBLE_DEVICES=3 python train.py  \
# --configs defaults metaworld  \
# --task metaworld_button_press  \
# --logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press_compare_offline2seeds_onlineinput124" \
# --run.from_offline_checkpoint '/data/ytzheng/log_icml25/checkpointsoffline_button_press_checkpoint_s2.ckpt'  \
# --run.from_online_checkpoint '/data/ytzheng/log_icml25/checkpoints/offline_button_press_checkpoint_s1.ckpt' \
# --run.script eval_only2 
# # --run.offline_directory '/data/ytzheng/dataset/metaworld_200eps_501/button_press/eval_eps'