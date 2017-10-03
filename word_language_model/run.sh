python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_1_clamp_1.0 --cuda --seed=1 --lr_thresh=1.0

python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_2_clamp_2.0 --cuda --seed=1 --lr_thresh=2.0

python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=./results/YF_seed_3_clamp_5.0 --cuda --seed=1 --lr_thresh=5.0
