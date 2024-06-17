#!/bin/sh
#python train.py --configs defaults dmc_vision --task dmc_quadruped_walk --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_walk_s0 --seed 0
#wait
#python train.py --configs defaults dmc_vision --task dmc_quadruped_walk --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_walk_s1 --seed 1
#wait
#python train.py --configs defaults dmc_vision --task dmc_quadruped_walk --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_walk_s2 --seed 2
#wait
#python train.py --configs defaults dmc_vision --task dmc_quadruped_run --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_run_s0 --seed 0
#wait
#python train.py --configs defaults dmc_vision --task dmc_quadruped_run --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_run_s1 --seed 1
#wait
#python train.py --configs defaults dmc_vision --task dmc_quadruped_run --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_run_s2 --seed 2
#wait

seed = 3

CUDA_VISIBLE_DEVICES=0 python train.py \
 --configs defaults metaworld \
 --task metaworld_button_press \
 --logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press" \
 --seed $seed \
 --from_offline_checkpoint '/data/ytzheng/log_nips2024/metaworld_baseline/button_press_full_replay_501_s1/checkpoint.ckpt' \
 --from_online_checkpoint '/data/ytzheng/log_eccv2024/metaworld/metaworld_button_press_s1/checkpoint.ckpt'

CUDA_VISIBLE_DEVICES=0 python train.py  --configs defaults metaworld  --task metaworld_button_press  --logdir "/data/ytzheng/log_icml2025/metaworld_off2on_test/metaworld_button_press"  --seed $seed  --from_offline_checkpoint '/data/ytzheng/log_nips2024/metaworld_baseline/button_press_full_replay_501_s1/checkpoint.ckpt'  --from_online_checkpoint '/data/ytzheng/log_eccv2024/metaworld/metaworld_button_press_s1/checkpoint.ckpt'