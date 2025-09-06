import os
import shutil
import random

# Define source and destination folders
source_folder = r"E:\School\Qwen2-Audio-finetune\vivos\train"
destination_folder = r"E:\School\Qwen2-Audio-finetune\vivos\eval"

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get list of all .wav files in the source folder
wav_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.wav')]

# Calculate number of files to move (5% of total)
num_files_to_move = max(1, int(len(wav_files) * 0.05))

# Randomly select files to move
files_to_move = random.sample(wav_files, num_files_to_move)

# Move selected files
for file in files_to_move:
    source_file = os.path.join(source_folder, file)
    destination_file = os.path.join(destination_folder, file)
    
    # Check for duplicate filenames
    if os.path.exists(destination_file):
        base, extension = os.path.splitext(file)
        counter = 1
        while os.path.exists(destination_file):
            new_filename = f"{base}_{counter}{extension}"
            destination_file = os.path.join(destination_folder, new_filename)
            counter += 1
    
    # Move the file
    shutil.move(source_file, destination_file)
    print(f"Moved: {file} to {destination_folder}")

print(f"Moved {len(files_to_move)} .wav files to {destination_folder}")