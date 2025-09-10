import os

def generate_kaldi_ark_scp(wav_dir, output_dir='.', scp_prefix='./data/'):
    """
    Generate data_wav.1.ark and my_wav.scp from a directory of wav files.
    
    Args:
        wav_dir (str): Directory containing the .wav files.
        output_dir (str): Directory to save the generated files (default: current directory).
        scp_prefix (str): Prefix for paths in my_wav.scp (default: './data/').
    """
    # Collect wav files and derive utt_ids from filenames (without .wav extension)
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    utt_wav_pairs = {os.path.splitext(f)[0]: os.path.join(wav_dir, f) for f in wav_files}
    
    # Sort by utt_id
    sorted_pairs = sorted(utt_wav_pairs.items())
    
    # File names
    ark_name = 'data_wav.1.ark'
    my_scp_name = 'my_wav.scp'
    
    ark_file = os.path.join(output_dir, ark_name)
    my_scp_file = os.path.join(output_dir, my_scp_name)
    
    with open(ark_file, 'wb') as ark_f:
        with open(my_scp_file, 'w') as my_scp_f:
            offset = 0
            for utt_id, wav_path in sorted_pairs:
                with open(wav_path, 'rb') as wav_f:
                    wav_binary = wav_f.read()
                
                # Write utt_id and space to ark
                ark_f.write(utt_id.encode('utf-8'))
                ark_f.write(b' ')
                
                # Calculate data offset
                data_offset = offset + len(utt_id) + 1
                
                # Write wav binary to ark
                ark_f.write(wav_binary)
                
                # Write to my_wav.scp
                my_scp_f.write(f"{utt_id} {scp_prefix}{ark_name}:{data_offset}\n")
                
                # Update total offset
                offset += len(utt_id) + 1 + len(wav_binary)
                
generate_kaldi_ark_scp(
    wav_dir=r'E:\School\Qwen2-Audio-finetune\vivos\eval',
    output_dir=r'E:\School\Qwen2-Audio-finetune\code\test_folder',
    scp_prefix='/kaggle/input/vivos-ark/vivos_ark/eval'
)