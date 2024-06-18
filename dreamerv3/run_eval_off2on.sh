CUDA_VISIBLE_DEVICES=3 python train.py  \
--configs defaults metaworld  \
--task metaworld_button_press  \
--logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press_compare_encout" \
--run.from_offline_checkpoint '/data/ytzheng/log_icml25/checkpoints/offline_button_press_checkpoint.ckpt'  \
--run.from_online_checkpoint '/data/ytzheng/log_icml25/checkpoints/online_button_press_checkpoint.ckpt' \
--run.script eval_only2