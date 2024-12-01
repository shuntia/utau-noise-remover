import subprocess
import os
import argparse
import concurrent.futures
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process .wav files to extract vocals using Demucs.")
parser.add_argument("--input_dir", type=str, default=os.path.expanduser("~/tmp/denoise"), help="Input directory containing .wav files")
parser.add_argument("--output_dir", type=str, default=os.path.expanduser("\0"), help="Output directory for processed files")
parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent workers")

args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir if args.output_dir != "\0" else input_dir
max_workers = args.max_workers
def process_file(mapped_name):
    input_file = os.path.join(input_dir, mapped_name)
    print(f"Processing (vocals only): {mapped_name}")

    # Run Demucs with the older model (`mdx_extra_q`) to extract only vocals
    subprocess.run([
        "demucs", 
        "-n", "mdx_extra_q",  # Use the older model
        "--two-stems", "vocals",  # Extract only vocals
        "--out", output_dir,  # Set the output directory
        input_file  # Input file
    ])

    print(f"Completed processing: {mapped_name}")

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all .wav files in the input directory
files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

# Process each file concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(process_file, files)

# Remove .wav files in the input directory if input and output directories are the same
if input_dir == output_dir:
    for file in files:
        file_path = os.path.join(input_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")

# Rename and move the processed files
for file in files:
    original_vocals_path = os.path.join(output_dir, "mdx_extra_q", file[:-4], "vocals.wav")
    new_vocals_path = os.path.join(output_dir, file)
    
    if os.path.exists(original_vocals_path):
        os.rename(original_vocals_path, new_vocals_path)
        print(f"Renamed and moved: {original_vocals_path} to {new_vocals_path}")
    else:
        print(f"File not found: {original_vocals_path}")