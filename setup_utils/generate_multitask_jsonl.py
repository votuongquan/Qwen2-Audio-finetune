import json

def convert_txt_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                key, target = parts
                json_obj = {
                    "key": key,
                    "task": "ASR",
                    "target": target
                }
                json.dump(json_obj, outfile, ensure_ascii=False)
                outfile.write('\n')

if __name__ == "__main__":
    input_file = r"E:\School\Capstone\vivos\train\prompts.txt"
    output_file = r"E:\School\Qwen2-Audio-finetune\data\vivos\multitask.jsonl"
    convert_txt_to_jsonl(input_file, output_file)