unzip yf_data.dat.zip
python long_stress_test.py --seed=1 --nhidden=1000  --init_range=0.1 --log_dir=./results/seed_${1}_long_stress_test --debug  --use_cuda --use_lstm

