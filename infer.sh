LOCAL_DIR=/kaggle/working/Qwen2-Audio-finetune
cd $LOCAL_DIR

MODEL_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio
TRAIN_DATA_PATH=/kaggle/input/vivos-newformat/vivos/train
EVAL_DATA_PATH=/kaggle/input/vivos-newformat/vivos/eval
TRAIN_STRATEGY=ddp # ddp deepspeed
DEVICE_TYPE=cuda # npu or cuda
PEFT_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune/exp/15-00-13/15-05/
#parameters
num_workers=4
prefetch_factor=2
python \
    ./infer.py \
    ++env.device_type=$DEVICE_TYPE \
    ++env.model_path=$MODEL_PATH \
    ++data.eval_data_path=$EVAL_DATA_PATH \
    ++data.num_workers=$num_workers \
    ++data.prefetch_factor=$prefetch_factor \
    ++eval.peft_path=$PEFT_PATH

