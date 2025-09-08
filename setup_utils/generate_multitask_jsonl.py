import json
import os

def convert_txt_to_jsonl(input_file, output_file, wav_folder):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                key, target = parts
                # Construct the WAV file path
                wav_file = os.path.join(wav_folder, f"{key}.wav")
                # Check if the WAV file exists
                if os.path.exists(wav_file):
                    json_obj = {
                        "key": key,
                        "task": "ASR",
                        "target": target
                    }
                    json.dump(json_obj, outfile, ensure_ascii=False)
                    outfile.write('\n')

if __name__ == "__main__":
    input_file = r"E:\School\Capstone\vivos\test\prompts.txt"
    output_file = r"E:\School\Qwen2-Audio-finetune\scp_wav\test_multitask.jsonl"
    wav_folder = r"E:\School\Qwen2-Audio-finetune\vivos\test"
    convert_txt_to_jsonl(input_file, output_file, wav_folder)