# Qwen2-Audio-finetune

[中文](./README_zh.md)
This is a repository prepared for fine-tuning Qwen2-Audio (Note: Not Instruct), supporting both GPU and NPU, with data reading and writing in ark and wav formats, and supporting DDP and LoRA.

# Requires
The following are the required dependencies for running this project:
```
plaintext
numpy==1.26.0
torch==2.1.0
torch-npu==2.1.0.post10 (if using NPU)
torchaudio==2.1.0
torchvision==0.16.0
soundfile
transformers==4.46.3
```

# Train
## Data Preparation
Please refer to the example data path /data/aishell-1 to prepare your data.
```
multitask.jsonl 
my_wav.scp
multiprompt.jsonl
```

## Configuration Preparation
Set the following necessary environment variables in train.sh:
```
LOCAL_DIR=
MODEL_PATH=
```

## Running the Code
Run the following command to start training:
```
bash train.sh
```

# RoadMap

## Notes
Data Path: Ensure that train_data_path and eval_data_path point to the correct data directories.
Device Selection: Set device_type to npu or cuda based on your hardware environment.
Dependency Installation: Ensure all dependencies are correctly installed. If not, you can use the following command:
```
pip install numpy==1.26.0 torch==2.1.0 torch-npu==2.1.0.post10 torchaudio==2.1.0 torchvision==0.16.0
```