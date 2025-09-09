LOCAL_DIR=/kaggle/working/Qwen2-Audio-finetune
cd $LOCAL_DIR

MODEL_PATH=/kaggle/input/qwen2audio7b/Qwen2-Audio-7B
TRAIN_DATA_PATH=/kaggle/input/vivos-newformat/vivos/train
EVAL_DATA_PATH=/kaggle/input/vivos-newformat/vivos/eval
TRAIN_STRATEGY=ddp # ddp deepspeed
DEVICE_TYPE=cuda # npu or cuda
#parameters
num_workers=4
prefetch_factor=2
if [[ $TRAIN_STRATEGY == ddp ]]
then
	torchrun \
		--nnodes 1 \
		--nproc_per_node 8 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH \
		++data.train_data_path=$TRAIN_DATA_PATH \
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
		++data.train_data_path=$TRAIN_DATA_PATH \
		++data.eval_data_path=$EVAL_DATA_PATH \


fi