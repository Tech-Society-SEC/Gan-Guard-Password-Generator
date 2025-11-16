import torch
import argparse
import os
from zxcvbn import zxcvbn
from models import Generator, LATENT_DIM  # Generator class is the same
from data.utils import int_to_char, MAX_LEN, VOCAB_SIZE

# --- Helper Function to Generate Passwords ---

def get_generated_passwords(model_path, num_to_find, device):
    """
    Generates a list of passwords from a model, filtering out
    blank strings and number-only strings.
    """
    generator = Generator()
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.to(device)
        generator.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return []

    passwords = []
    
    # We will try up to 10x as hard to find the number of passwords
    # you asked for, in case the model is collapsing.
    max_attempts = num_to_find * 10
    
    with torch.no_grad():
        for _ in range(max_attempts):
            noise = torch.randn(1, LATENT_DIM).to(device)
            fake_password_tensor = generator(noise)
            _, indices = torch.max(fake_password_tensor, dim=2)
            indices = indices.squeeze(0)
            
            password_str = ""
            for idx in indices:
                char_index = idx.item()
                if char_index == 0: # Stop at padding
                    break
                password_str += int_to_char.get(char_index, '?')
            
            # ‼️ --- FILTER 1: IGNORE BLANK STRINGS --- ‼️
            if not password_str:
                continue
                
            # ‼️ --- FILTER 2: IGNORE "NUMBER-ONLY" STRINGS --- ‼️
            if password_str.isdigit():
                continue
            
            # If it passes, add it to the list
            passwords.append(password_str)
            
            # Stop if we've found enough
            if len(passwords) >= num_to_find:
                break

    return passwords

# --- Evaluation Functions ---

def check_uniqueness(passwords):
    """
    Calculates the percentage of unique passwords in a list.
    """
    if not passwords:
        return 0
    unique_passwords = set(passwords)
    uniqueness_rate = len(unique_passwords) / len(passwords)
    return uniqueness_rate * 100

def check_novelty(passwords, training_data_path):
    """
    Calculates the percentage of generated passwords that are NOT in the training data.
    """
    if not passwords:
        return 0
    
    try:
        with open(training_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            training_passwords = set(line.strip() for line in f)
    except FileNotFoundError:
        print(f"❌ Error: Training data file not found at {training_data_path}")
        return 0
    except Exception as e:
        print(f"❌ Error reading training data: {e}")
        return 0

    if not training_passwords:
        print("❌ Error: Training data is empty.")
        return 0

    novel_count = 0
    for p in passwords:
        if p not in training_passwords:
            novel_count += 1
            
    novelty_rate = novel_count / len(passwords)
    return novelty_rate * 100

def check_strength(passwords):
    """
    Calculates the average ZXCVBN score for a list of passwords.
    Score ranges from 0 (terrible) to 4 (excellent).
    """
    if not passwords:
        return 0
    
    total_score = 0
    for p in passwords:
        try:
            analysis = zxcvbn(p)
            total_score += analysis['score']
        except Exception as e:
            # Handle any zxcvbn errors on weird strings
            continue
        
    average_score = total_score / len(passwords)
    return average_score

# --- Main Evaluation Function ---

def evaluate_model(model_path, num_samples, training_data_path):
    """
    Runs a full evaluation on a trained generator model.
    """
    print(f"📊 Starting evaluation for: {model_path}")
    print(f"Attempting to generate {num_samples} valid (filtered) samples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate filtered passwords
    passwords = get_generated_passwords(model_path, num_samples, device)
    
    if not passwords:
        print("❌ Evaluation failed: No valid (non-blank, non-numeric) passwords were generated.")
        return

    print(f"✅ Generated {len(passwords)} valid passwords.")
    
    # 2. Run evaluations
    print("\n--- Evaluation Results ---")
    
    # Uniqueness
    uniqueness = check_uniqueness(passwords)
    print(f"📈 Uniqueness: {uniqueness:.2f}%")
    print("   (Percentage of valid passwords that are unique)")
    
    # Novelty
    novelty = check_novelty(passwords, training_data_path)
    print(f"✨ Novelty: {novelty:.2f}%")
    print(f"   (Percentage of valid passwords NOT found in {training_data_path})")

    # Strength
    strength = check_strength(passwords)
    print(f"💪 Average Strength (zxcvbn): {strength:.2f} / 4.0")
    print("   (Average password strength score from 0 to 4)")
    
    print("\n--- Sample of 10 Valid Passwords ---")
    for p in passwords[:10]:
        print(p)
    print("---------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained GAN Generator.")
    
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the generator .pth file."
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1000,
        help="Number of valid passwords to generate for evaluation."
    )
    
    parser.add_argument(
        "--training_data", 
        type=str, 
        # ‼️ --- IMPORTANT --- ‼️
        # Make sure this is the default dataset we are training on
        default="rockyou_common_subset.txt", 
        help="Path to the original training data for novelty check."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
    elif not os.path.exists(args.training_data):
        print(f"❌ Error: Training data file not found at {argsAtts.training_data}")
    else:
        evaluate_model(args.model_path, args.num_samples, args.training_data)