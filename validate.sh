#work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name=resnet18_20e_class5 \
python -u validate.py --bs 256 --model SN --num_classes 5 --checkpoint kdSN20e_resnet152_zm/checkpoint.pth.tar

#--config $work_path/config.yaml \
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
