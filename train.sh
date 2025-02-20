export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOCAL_DIR=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune

## env
export model_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio"
export train_data_path="./data/aishell-1/asr/test"
export eval_data_path="./data/aishell-1/asr/test"
export lr=1e-4
export seed=1314
export batch_size=1
export device_type="npu"
export eval_step=2
export train_epoch=5
export prompt_path="./data/multiprompt.jsonl"
export save_path="./exp/$(date +"%H%M")"
## 
cd $LOCAL_DIR
torchrun \
	--nnodes 1 \
	--nproc_per_node 2 \
	--master_port=29502 \
	train.py