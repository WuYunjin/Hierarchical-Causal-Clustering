#!/bin/bash

# run this script with command: nohup bash run.sh &


# for n_sub in 10 15 20 25 30
# do
#         for seed in 0 1 2 3 5 8 11 17 27 30
#         do
#                 nohup python -u main.py  --seed $seed --num_variables 5  --num_subjects_per_group $n_sub --num_samples 100 --num_groups 2 > nohup.txt 2>&1 &
#                 sleep 3
#         done
#         # wait
# done
# wait

# for n_sample in 100 120 140 160 180 
# do
#         for seed in  0 3 5 8 11 17 18 26 27 28
#         do
#                 nohup python -u main.py  --seed $seed  --num_variables 5 --num_subjects_per_group 15 --num_samples $n_sample --num_groups 2 > nohup.txt 2>&1 &
#                 sleep 3
#         done
#         # wait
# done
# wait

# for n_group in 2 3 4 5 6
# do
#         for seed in 0 3 5 8 11 17 18 26 27 28
#         do
#                 nohup python -u main.py  --seed $seed  --num_variables 5 --num_subjects_per_group $((60 / $n_group)) --num_samples 100 --num_groups $n_group  > nohup.txt 2>&1 &
#                 sleep 3
#         done
#         # wait
# done
# wait

for n_variable in 6 8 10 12 14
do
        for seed in  11 17 26 30 33 34 38 41 48 49
        do
                nohup python -u main.py  --seed $seed  --num_variables $n_variable --num_samples 100 --num_groups 2 --num_subjects_per_group 15 > nohup.txt 2>&1 &
                sleep 3
        done
        # wait
done
wait