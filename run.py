import torch
import argparse
from models import Generator, LATENT_DIM
from data.utils import int_to_char, MAX_LEN, VOCAB_SIZE

def generate_passwords(model_path, num_passwords, device):
    """
    Loads a trained generator model and generates a specified number of passwords.
    """
    generator = Generator()
    
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.to(device)
        generator.eval()
        print(f"✅ Loaded model: {model_path}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"--- {num_passwords} Filtered Passwords ---")
    
    generated_passwords = []
    
    with torch.no_grad():
        # We'll ask for many more passwords than we need,
        # since we will be filtering out the bad ones.
        passwords_to_generate = num_passwords * 5 
        
        for _ in range(passwords_to_generate):
            # 1. Generate random noise
            noise = torch.randn(1, LATENT_DIM).to(device)
            
            # 2. Generate password with the model
            fake_password_tensor = generator(noise)
            
            # 3. Convert tensor to password string
            _, indices = torch.max(fake_password_tensor, dim=2)
            indices = indices.squeeze(0)
            
            password_str = ""
            for idx in indices:
                char_index = idx.item()
                if char_index == 0:  # Stop at the first padding character
                    break
                password_str += int_to_char.get(char_index, '?')
            
            # ‼️ --- NEW FILTER 1: IGNORE BLANK STRINGS --- ‼️
            if not password_str:
                continue
                
            # ‼️ --- NEW FILTER 2: IGNORE "NUMBER-ONLY" STRINGS --- ‼️
            if password_str.isdigit():
                continue
                
            # If the password passed the filters, add it to our list
            print(password_str)
            generated_passwords.append(password_str)
            
            # Stop once we have enough good passwords
            if len(generated_passwords) >= num_passwords:
                break
            
    print("-------------------------------")
    
    if len(generated_passwords) < num_passwords:
        print(f"⚠️ Warning: Model collapsed. Could only find {len(generated_passwords)} valid passwords.")
    
    return generated_passwords

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate passwords using a trained GAN Generator.")
    
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the trained generator .pth file."
    )
    
    parser.add_argument(
        "--num", 
        "-n",
        type=int, 
        default=10,
        help="Number of passwords to generate."
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generate_passwords(args.model_path, args.num, device)