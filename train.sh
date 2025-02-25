LOCAL_DIR=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune
cd $LOCAL_DIR

MODEL_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio
# TRIAN_DATA_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multitask_asr/train
# EVAL_DATA_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multitask_asr/dev
# TRIAN_DATA_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/train
# EVAL_DATA_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/dev
TRAIN_STRATEGY=deepspeed # ddp deepspeed
DEVICE_TYPE=npu # npu cuda
#parameters
num_workers=8
prefetch_factor=2
if [[ $TRAIN_STRATEGY == ddp ]]
then
# export ASCEND_VISIBLE_DEVICES=0,1 #CPU
# export CUDA_VISIBLE_DEVICES=0,1 #GPU
	torchrun \
		--nnodes 1 \
		--nproc_per_node 8 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH \
		++data.train_data_path=$TRIAN_DATA_PATH \
		++data.eval_data_path=$EVAL_DATA_PATH \
		++data.num_workers=$num_workers \
		++data.prefetch_factor=$prefetch_factor

else
export DEEPSPEED_CONFIG=./config/deepspeed.json
	deepspeed \
		--num_nodes 1 \
		--num_gpus 8 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++train.deepspeed_config=$DEEPSPEED_CONFIG \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH \
		++data.train_data_path=$TRIAN_DATA_PATH \
		++data.eval_data_path=$EVAL_DATA_PATH \


fi