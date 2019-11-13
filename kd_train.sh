#work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=8 --job-name tb_kd_atp_dn121_e20_c5_a75 \
python -u new_train_kd.py --bs 800 --model anytime_prediction_model_densenet121 --lr 0.001 --name tb_kd_atp_dn121_e20_c5_a75_lr10 --epoch 20 --teacher_checkpoint /mnt/lustre/tangzhejun/CE/Datascience/experiments/tb_atp_dn121_e20_c5/model_best.pth.tar

#--config $work_path/config.yaml \
 #--load-path=$work_path/ckpt.pth.tar \
 #--recover
