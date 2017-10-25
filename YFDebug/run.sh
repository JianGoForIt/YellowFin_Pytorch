#for ((seed=1;seed<=100;seed++));
#do
#  python train_mnist_torch.py --log_dir='./results_stress/seed_${seed}_h_max_log_h_max_clip_100_hard_stress_test' --fast_bound_const=1.0 --seed=${seed}
#done
unzip train.txt.zip
unzip test.txt.zip
python train_mnist_torch.py --log_dir='./results/seed_5_test_memory_ckpt_loop' --seed=5
