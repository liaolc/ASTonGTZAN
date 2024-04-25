#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
##SBATCH -p sm
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
# source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=GTZAN
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=1e-4
epoch=10
freqm=48
timem=200
mixup=0.7
batch_size=8
fstride=10
tstride=10

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_length=1024
noise=False

metrics=acc
loss=BCE
warmup=True
wa=True
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.5


#+ tr_data=/content/drive/MyDrive/Rana_Draytonii/Rana7/train_data.json
#+ te_data=/content/drive/MyDrive/Rana_Draytonii/Rana7/val_data.json
tr_data=/content/drive/MyDrive/ASTonGTZAN-main/egs/GTZAN/data/GTZAN_train_data.json
val_data=/content/drive/MyDrive/ASTonGTZAN-main/egs/GTZAN/data/GTZAN_valid_data.json
eval_data=/content/drive/MyDrive/ASTonGTZAN-main/egs/GTZAN/data/GTZAN_eval_data.json
#tr_data=./data/datafiles/GTZAN_train_data.json
#val_data=./data/datafiles/GTZAN_valid_data.json
#eval_data=./data/datafiles/GTZAN_eval_data.json
exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-decoupe

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

# python ./prep_GTZAN.py

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv /content/drive/MyDrive/ASTonGTZAN-main/egs/GTZAN/data/GTZAN_class_labels_indices.csv --n_class 10 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain > $exp_dir/log.txt
