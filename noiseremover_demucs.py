import os
import argparse
import concurrent.futures
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process .wav files to extract vocals using Demucs.")
parser.add_argument("--input_dir", type=str, default=os.path.expanduser("~/tmp/denoise"), help="Input directory containing .wav files")
parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/tmp/denoise_output"), help="Output directory for processed files")
parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent workers")
parser.add_argument("--model", type=str, default="mdx_extra_q", help="Model name to use for processing")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
parser.add_argument("--leave-artifacts", action="store_true", help="Leave processed files in the input directory")
parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")

args = parser.parse_args()
verbose = args.verbose
input_dir = args.input_dir
if verbose: print(f"Options: {args}")
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory not found: {input_dir}")
if not os.path.isdir(input_dir):
    raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
if args.output_dir == "~/tmp/denoise_output" and verbose:
    print("Output directory not specified. Using the input directory as the output directory.")
output_dir = args.output_dir if args.output_dir != "~/tmp/denoise_output" else input_dir
max_workers = args.max_workers
model_name = args.model

# Load the Demucs model
model = get_model(model_name)
model.eval()
device = torch.device("cpu" if args.no_cuda else "cuda")
model.to(device)

def process_file(mapped_name):
    input_file = os.path.join(input_dir, mapped_name)
    # Load audio
    wav, sr = torchaudio.load(input_file)
    wav = wav.to(device)
    
    # Convert mono to stereo if necessary
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    
    # Apply the model
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device)[0]
    
    # Get the vocals
    vocals = sources[model.sources.index('vocals')]
    
    # Convert vocals to mono
    vocals_mono = torch.mean(vocals, dim=0, keepdim=True)
    
    # Construct output path
    relative_path = os.path.relpath(os.path.dirname(input_file), input_dir)
    output_subdir = os.path.join(output_dir, relative_path)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, os.path.basename(input_file))
    
    # Save vocals
    torchaudio.save(output_path, vocals_mono.cpu(), sr)
    if verbose: print(f"Saved vocals to {output_path}")

# Recursively get all .wav files in the input directory
files = []
for root, _, filenames in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.wav'):
            relative_path = os.path.relpath(os.path.join(root, filename), input_dir)
            files.append(relative_path)
if verbose:
    print(f"Files: {files}")
print(f"Found {len(files)} .wav files in the input directory. starting processing...")

# Process each file concurrently with a progress bar
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_file, files), total=len(files)))
    executor.shutdown(wait=True)
print("Processing complete. Cleaning up...")
# Remove .wav files in the input directory if input and output directories are the same and leave_artifacts is False
if input_dir == output_dir and not args.leave_artifacts:
    for file in files:
        file_path = os.path.join(input_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            if verbose: print(f"Removed: {file_path}")

    # Remove files from input directory and its subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                if verbose: print(f"Removed: {file_path}")

# Rename and move the processed files
if input_dir == output_dir:
    for root, dirs, files in os.walk(os.path.join(output_dir, model_name)):
        for dir_name in dirs:
            original_vocals_path = os.path.join(root, dir_name, "vocals.wav")
            relative_path = os.path.relpath(os.path.join(root, dir_name), os.path.join(output_dir, model_name))
            new_vocals_path = os.path.join(output_dir, f"{relative_path}.wav")
            
            if os.path.exists(original_vocals_path):
                os.makedirs(os.path.dirname(new_vocals_path), exist_ok=True)
                os.rename(original_vocals_path, new_vocals_path)
                if verbose: print(f"Renamed and moved: {original_vocals_path} to {new_vocals_path}")
            else:
                print(f"File not found: {original_vocals_path}")

    # Remove empty directories
    for root, dirs, files in os.walk(os.path.join(output_dir, model_name), topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                if verbose: print(f"Removed empty directory: {dir_path}")
print("Processing complete.")