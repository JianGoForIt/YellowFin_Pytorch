unzip train.txt.zip
unzip test.txt.zip
python train_mnist_torch.py --log_dir='./results/seed_5_test_memory_ckpt_clean_up' --seed=5
