python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_2_h_max_log_test_slow_start_10_win --cuda --seed=2

python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_h_max_log_test_slow_start_10_win --cuda --seed=1

python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_3_h_max_log_test_slow_start_10_win --cuda --seed=3
#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_fast_view_clamp_0.01 --cuda --seed=1 --lr_thresh=0.01

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_min_lr_t_lr_max_mu_t_mu_h_max_linear_smooth_exploding_clip_fac_2 --cuda --seed=1 --lr_thresh=2.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_2_lr_t_mu_t_h_max_log --cuda --seed=2 --lr_thresh=2.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_lr_t_mu_t_h_max_log_no_clip --cuda --seed=1 --lr_thresh=2.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_3_lr_t_mu_t_h_max_log_no_clip --cuda --seed=3 --lr_thresh=2.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=Adam --logdir=./results/Adam_seed_1_no_clip_lr_0.001 --cuda --seed=1 --lr_thresh=2.0 --lr=0.001

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=Adam --logdir=./results/Adam_seed_3_no_clip_lr_0.001 --cuda --seed=3 --lr_thresh=2.0 --lr=0.001

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=Adam --logdir=./results/Adam_seed_1_no_clip_lr_0.01 --cuda --seed=1 --lr_thresh=2.0 --lr=0.01

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=Adam --logdir=./results/Adam_seed_1_no_clip_lr_0.1 --cuda --seed=1 --lr_thresh=2.0 --lr=0.1

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=Adam --logdir=./results/Adam_seed_1_no_clip_lr_0.00001 --cuda --seed=1 --lr_thresh=2.0 --lr=0.00001

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=Adam --logdir=./results/Adam_seed_1_no_clip_lr_0.001 --cuda --seed=1 --lr_thresh=2.0 --lr=0.001

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_clip_grad_fac_10.0 --cuda --seed=1 --lr_thresh=10.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_clamp_1.0 --cuda --seed=1 --lr_thresh=1.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_2_clamp_2.0 --cuda --seed=1 --lr_thresh=2.0

#python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_3_clamp_5.0 --cuda --seed=1 --lr_thresh=5.0
