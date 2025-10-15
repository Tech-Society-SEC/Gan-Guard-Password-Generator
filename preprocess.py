# preprocess.py

import torch
from data.utils import CHARS, MAX_LEN, char_to_int
import time

SOURCE_FILE = 'rockyou_expert.txt'
DEST_FILE = 'rockyou_expert_processed.pt'

print("Starting preprocessing...")
start_time = time.time()

# Read all lines from the source file
with open(SOURCE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    passwords = [
        line.strip() for line in f
        if len(line.strip()) <= MAX_LEN and all(c in CHARS for c in line.strip())
    ]

print(f"Found {len(passwords)} valid passwords.")
print("Converting to tensors...")

# Create a list to hold all the tensor data
tensor_list = []
for password in passwords:
    int_sequence = [char_to_int[char] for char in password]
    padded_sequence = int_sequence + [0] * (MAX_LEN - len(int_sequence))
    tensor_list.append(torch.tensor(padded_sequence, dtype=torch.long))

# Stack all individual tensors into a single large tensor
# Shape will be [num_passwords, MAX_LEN]
processed_data = torch.stack(tensor_list)

print(f"Final tensor shape: {processed_data.shape}")
print(f"Saving to {DEST_FILE}...")

# Save the final tensor to a file
torch.save(processed_data, DEST_FILE)

end_time = time.time()
print(f"✅ Preprocessing complete in {end_time - start_time:.2f} seconds.")