import os

def generate_scp_file(source_folder, output_scp, prefix):
    wav_files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]
    wav_files.sort()
    
    # Generate .scp file
    with open(output_scp, 'w') as f:
        for wav in wav_files:
            utterance_id = os.path.splitext(wav)[0]  # Remove .wav extension
            wav_path = os.path.join(prefix, source_folder, wav).replace('\\', '/')
            f.write(f"{utterance_id} {wav_path}\n")

def concatenate_scp_files(train_scp, eval_scp, output_scp):
    with open(output_scp, 'w') as outfile:
        with open(train_scp, 'r') as infile:
            outfile.write(infile.read())
        with open(eval_scp, 'r') as infile:
            outfile.write(infile.read())

dest_folder = r'E:\School\Qwen2-Audio-finetune\scp_wav'  

train_folder = r'E:\School\Qwen2-Audio-finetune\vivos\train' 
eval_folder = r'E:\School\Qwen2-Audio-finetune\vivos\eval'    
   
train_prefix = '/kaggle/input/vivos-newformat/vivos/train'    
eval_prefix = '/kaggle/input/vivos-newformat/vivos/eval'      
train_scp = os.path.join(dest_folder, 'train_wav.scp')        
eval_scp = os.path.join(dest_folder, 'eval_wav.scp')          
my_wav_scp = os.path.join(dest_folder, 'my_wav.scp')         

os.makedirs(dest_folder, exist_ok=True)

generate_scp_file(train_folder, train_scp, train_prefix)

generate_scp_file(eval_folder, eval_scp, eval_prefix)

concatenate_scp_files(train_scp, eval_scp, my_wav_scp)