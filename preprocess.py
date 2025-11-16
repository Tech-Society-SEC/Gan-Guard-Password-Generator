import torch
from data.utils import CHARS, MAX_LEN, char_to_int
import time
import argparse

def preprocess_data(source_file, dest_file):
    """
    Reads a source password file, filters and processes the passwords,
    and saves them as a PyTorch tensor to the destination file.
    """
    print(f"Starting preprocessing for: {source_file}")
    start_time = time.time()

    try:
        with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
            passwords = [
                line.strip() for line in f
                if len(line.strip()) <= MAX_LEN and all(c in CHARS for c in line.strip())
            ]
    except FileNotFoundError:
        print(f"Error: The file '{source_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    print(f"Found {len(passwords)} valid passwords.")
    
    if not passwords:
        print("No valid passwords found. Aborting.")
        return

    print("Converting to tensors...")
    tensor_list = []
    for password in passwords:
        int_sequence = [char_to_int[char] for char in password]
        padded_sequence = int_sequence + [0] * (MAX_LEN - len(int_sequence))
        tensor_list.append(torch.tensor(padded_sequence, dtype=torch.long))

    processed_data = torch.stack(tensor_list)

    print(f"Final tensor shape: {processed_data.shape}")
    print(f"Saving to {dest_file}...")

    torch.save(processed_data, dest_file)

    end_time = time.time()
    print(f"✅ Preprocessing complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a password dataset for the GAN.")
    
    parser.add_argument(
        "source_file", 
        type=str, 
        help="Path to the source .txt password file (e.g., rockyou.txt)"
    )
    
    parser.add_argument(
        "--output", 
        "-o",
        type=str, 
        default="processed_data.pt",
        help="Path to save the processed .pt tensor file (e.g., rockyou_common.pt)"
    )
    
    args = parser.parse_args()
    
    preprocess_data(args.source_file, args.output)