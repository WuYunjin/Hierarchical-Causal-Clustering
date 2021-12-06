#!/bin/bash

# run this script with command: nohup bash run.sh &

for n_sub in 10 
do
        for seed in 2 3 4 6 7 8 9 10 11 12
        do
                nohup python -u main.py  --num_iterations_clustering 200 --seed $seed --num_variables 5  --num_subjects_per_group $n_sub --num_samples 60 --num_groups 2 > nohup.txt 2>&1 &
                sleep 5
        done
done
for n_sub in 15 20 25 30
do
        for seed in 2 3 4 6 7 8 9 10 11 12
        do
                nohup python -u main.py  --num_iterations_clustering $[$n_sub*10] --seed $seed --num_variables 5  --num_subjects_per_group $n_sub --num_samples 60 --num_groups 2 > nohup.txt 2>&1 &
                sleep 5
        done
        # wait
done
wait

for n_sample in 40 60 80 100 120
do
        for seed in 2 3 4 6 7 8 9 10 11 12
        do
                nohup python -u main.py  --seed $seed  --num_variables 5 --num_subjects_per_group 15 --num_samples $n_sample --num_groups 2 > nohup.txt 2>&1 &
                sleep 5
        done
        # wait
done
wait


for n_group in 2 3
do
        for seed in 0 1 2 3 4 6 7 8 9 10
        do
                nohup python -u main.py  --seed $seed  --num_iterations_clustering 200 --num_variables 5 --num_subjects_per_group $((60 / $n_group)) --num_samples 60 --num_groups $n_group  > nohup.txt 2>&1 &
                sleep 5
        done
done
for n_group in 4 5 6
do
        for seed in 0 1 2 3 4 6 7 8 9 10
        do
                nohup python -u main.py  --seed $seed  --num_iterations_clustering 250 --num_variables 5 --num_subjects_per_group $((60 / $n_group)) --num_samples 60 --num_groups $n_group  > nohup.txt 2>&1 &
                sleep 5
        done
done
wait



for n_variable in 6 8
do
        for seed in 0 1 2 3 5 6 7 8 9 10
        do
                nohup python -u main.py  --seed $seed --num_variables $n_variable --num_samples 60 --num_groups 2 --num_subjects_per_group 15 > nohup.txt 2>&1 &
                sleep 5
        done
done
for n_variable in 10 12 14
do
        for seed in 0 1 2 3 5 6 7 8 9 10
        do
                nohup python -u main.py  --seed $seed --num_iterations_clustering $[$n_variable*10] --num_variables $n_variable --num_samples 60 --num_groups 2 --num_subjects_per_group 15 > nohup.txt 2>&1 &
                sleep 5
        done
done
wait