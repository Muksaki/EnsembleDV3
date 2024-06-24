
# offline copycat
CUDA_VISIBLE_DEVICES=2 python train.py  \
--configs defaults metaworld  \
--task metaworld_button_press  \
--logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press_ensemble_test" \
--run.from_offline_checkpoint '/data/ytzheng/log_icml25/metaworld/ensemble_offlinemt/button_press_train_eval_5embody/checkpoint.ckpt'  \
--run.from_online_checkpoint '/data/ytzheng/log_icml25/metaworld/metaworld_button_press_s0/checkpoint.ckpt' \
--run.script eval_only2 \
--run.offline_directory '/data/ytzheng/dataset/metaworld_200eps_501/button_press_copycat/eval_eps' \
--replay kuniform \
--ensemble_number 5
# CUDA_VISIBLE_DEVICES=3 python train.py  \
# --configs defaults metaworld  \
# --task metaworld_button_press  \
# --logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press_ensemble_test" \
# --run.from_offline_checkpoint '/data/ytzheng/log_icml25/metaworld/ensemble_offlinemt/button_press_train_eval_5embody/checkpoint.ckpt'  \
# --run.from_online_checkpoint '/data/ytzheng/log_icml25/metaworld/metaworld_button_press_s0/checkpoint.ckpt' \
# --run.script eval_only2 \
# --run.offline_directory '' \
# --replay uniform