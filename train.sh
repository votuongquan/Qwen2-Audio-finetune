LOCAL_DIR=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune
cd $LOCAL_DIR



MODEL_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio

TRAIN_STRATEGY=ddp # ddp deepspeed
DEVICE_TYPE=npu # npu gpu

if [[ $TRAIN_STRATEGY == ddp ]]
then
export ASCEND_VISIBLE_DEVICES=0,1 #CPU
export CUDA_VISIBLE_DEVICES=0,1 #GPU
	torchrun \
		--nnodes 1 \
		--nproc_per_node 2 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH 

else
export DEEPSPEED_CFG_PATH=./config/deepspeed.json
	deepspeed \
		--num_nodes 1 \
		--num_gpus 2 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH 
fi