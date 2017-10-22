for seed in 1 2 3
do
  for fac in 0.333 3.0
  do
    echo ${seed} ${fac}
    python main.py --logdir=./results/YF_seed_1_boost_fac_${fac} --opt_method=YF --seed=${seed} --lr_factor=${fac}
  done
done



#python main.py --lr=1.0 --mu=0.0 --logdir=./results/YF_seed_1_h_max_log_test_slow_start_10_win --opt_method=YF --seed=1 --lr_thresh=2.0
#python main.py --lr=1.0 --mu=0.0 --logdir=./results/YF_seed_2_h_max_log_test_slow_start_10_win --opt_method=YF --seed=2 --lr_thresh=2.0
#python main.py --logdir=./results/YF_seed_1_h_max_log_slow_start_10_win_h_max_clip --opt_method=YF --seed=1
#python main.py --logdir=./results/YF_seed_3_h_max_log_slow_start_10_win_h_max_clip --opt_method=YF --seed=3
