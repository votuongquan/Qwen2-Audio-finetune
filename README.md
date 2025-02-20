
# Qwen2-Audio-finetune
[中文Readme](./README_zh.md)
This is a repository prepared for fine-tuning Qwen2-Audio (note: no Instruct). It supports GPU and NPU, and the data supports reading and writing in ark and wav formats.

---

# Requires

The following dependencies are required to run this project:

```plaintext
numpy==1.26.0  
torch==2.1.0  
torch-npu==2.1.0.post10  
torchaudio==2.1.0  
torchvision==0.16.0  
soundfile  
transformers==4.46.3
```
# Train
## Data Preparation
Prepare the data by referring to the example data path /data/aishell-1.
## Configuration Preparation
Set the following environment variables in train.sh:
```
export model_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio"  
export train_data_path="./data/aishell-1/asr/train"  
export eval_data_path="./data/aishell-1/asr/test"  
export device_type="npu"  # Options: npu or cuda  
```
## Running the Code
Start the training by running the following command:
```
bash run.sh
```
## RoadMap

# Notes
Data Path: Ensure that train_data_path and eval_data_path point to the correct data directories.
Device Selection: Set device_type to npu or cuda based on your hardware environment.
Dependency Installation: Ensure all dependencies are correctly installed. If not, you can use the following command:
```
pip install numpy==1.26.0 torch==2.1.0 torch-npu==2.1.0.post10 torchaudio==2.1.0 torchvision==0.16.0
```
