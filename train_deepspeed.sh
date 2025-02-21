export ASCEND_VISIBLE_DEVICES=0,1
LOCAL_DIR=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune

## env
export model_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio"
export train_data_path="./data/aishell-1/asr/test"
export eval_data_path="./data/aishell-1/asr/test"
export lr=1e-4
export seed=1314
export batch_size=1
export device_type="npu" # npu cuda
export eval_step=2000
export train_epoch=5
export prompt_path="./data/multiprompt.jsonl"
export save_path="./exp/$(date +"%H%M")"
export total_train_steps=100000
export warmup_steps=1000
export wav_type=ark # ark wav
# export train_strategy=fsdp # ddp fsdp
## run
cd $LOCAL_DIR
deepspeed \
	./src/train.py