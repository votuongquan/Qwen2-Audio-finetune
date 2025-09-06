import os
import shutil

# Define source and destination folders
source_folder = r"E:\School\Qwen2-Audio-finetune\vivos_old\train\waves"
destination_folder = r"E:\School\Qwen2-Audio-finetune\vivos\train"

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Walk through the source folder
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # Check if the file is a .wav file
        if file.lower().endswith('.wav'):
            # Get the full path of the source file
            source_file = os.path.join(root, file)
            # Get the full path of the destination file
            destination_file = os.path.join(destination_folder, file)
            
            # Check for duplicate filenames
            if os.path.exists(destination_file):
                # If file exists, append a number to avoid overwriting
                base, extension = os.path.splitext(file)
                counter = 1
                while os.path.exists(destination_file):
                    new_filename = f"{base}_{counter}{extension}"
                    destination_file = os.path.join(destination_folder, new_filename)
                    counter += 1
            
            # Move the file
            shutil.move(source_file, destination_file)
            print(f"Moved: {file} to {destination_folder}")

print("All .wav files have been moved to", destination_folder)