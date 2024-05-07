#!/bin/sh
python train.py --configs defaults dmc_vision --task dmc_quadruped_walk --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_walk_s0 --seed 0
wait
python train.py --configs defaults dmc_vision --task dmc_quadruped_walk --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_walk_s1 --seed 1
wait
python train.py --configs defaults dmc_vision --task dmc_quadruped_walk --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_walk_s2 --seed 2
wait
python train.py --configs defaults dmc_vision --task dmc_quadruped_run --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_run_s0 --seed 0
wait
python train.py --configs defaults dmc_vision --task dmc_quadruped_run --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_run_s1 --seed 1
wait
python train.py --configs defaults dmc_vision --task dmc_quadruped_run --logdir /data/ytzheng/log_eccv2024/dmc/dmc_quadruped_run_s2 --seed 2
wait
python train.py --configs defaults metaworld --task metaworld_button_press --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_button_press_s0 --seed 0
wait
python train.py --configs defaults metaworld --task metaworld_button_press --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_button_press_s1 --seed 1
wait
python train.py --configs defaults metaworld --task metaworld_button_press --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_button_press_s2 --seed 2
wait
python train.py --configs defaults metaworld --task metaworld_coffee_push --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_coffee_push_s0 --seed 0
wait
python train.py --configs defaults metaworld --task metaworld_coffee_push --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_coffee_push_s1 --seed 1
wait
python train.py --configs defaults metaworld --task metaworld_coffee_push --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_coffee_push_s2 --seed 2
wait
python train.py --configs defaults metaworld --task metaworld_drawer_open --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_drawer_open_s0 --seed 0
wait
python train.py --configs defaults metaworld --task metaworld_drawer_open --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_drawer_open_s1 --seed 1
wait
python train.py --configs defaults metaworld --task metaworld_drawer_open --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_drawer_open_s2 --seed 2
wait
python train.py --configs defaults metaworld --task metaworld_window_open --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_window_open_s0 --seed 0
wait
python train.py --configs defaults metaworld --task metaworld_window_open --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_window_open_s1 --seed 1
wait
python train.py --configs defaults metaworld --task metaworld_window_open --logdir /data/ytzheng/log_eccv2024/metaworld/metaworld_window_open_s2 --seed 2
wait
