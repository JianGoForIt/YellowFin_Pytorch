echo seed ${1}
echo init range ${2}
echo nhidden ${3}
python YF_batch_50-Oct-1-test-master.py --seed=${1} --nhidden=${3}  --init_range=${2}  --log_dir=./results/full_master_test_seed_${1}_init_${2}_nhid_${3} --debug --use_lstm --use_cuda
