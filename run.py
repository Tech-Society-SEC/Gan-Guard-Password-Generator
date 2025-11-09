import torch
import argparse
from models import Generator, LATENT_DIM
from data.utils import int_to_char, MAX_LEN, VOCAB_SIZE

def generate_passwords(model_path, num_passwords, device):
    """
    Loads a trained generator model and generates a specified number of passwords.
    """
    # Initialize the generator
    generator = Generator()
    
    try:
        # Load the trained weights
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.to(device)
        generator.eval()  # Set the model to evaluation mode
        print(f"✅ Loaded model: {model_path}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"--- {num_passwords} Generated Passwords ---")
    
    generated_passwords = []
    
    with torch.no_grad():  # Turn off gradient calculations for inference
        for _ in range(num_passwords):
            # 1. Generate random noise
            noise = torch.randn(1, LATENT_DIM).to(device)
            
            # 2. Generate password with the model
            # Output shape is (1, MAX_LEN, VOCAB_SIZE)
            fake_password_tensor = generator(noise)
            
            # 3. Convert tensor to password string
            # Get the index of the most likely character for each position
            # Shape of 'indices' will be (1, MAX_LEN)
            _, indices = torch.max(fake_password_tensor, dim=2)
            
            # Squeeze to remove the batch dimension, shape becomes (MAX_LEN)
            indices = indices.squeeze(0)
            
            # Convert indices to characters
            password_str = ""
            for idx in indices:
                char_index = idx.item()
                if char_index == 0:  # Stop at the first padding character
                    break
                password_str += int_to_char.get(char_index, '?')
                
            print(password_str)
            generated_passwords.append(password_str)
            
    print("-------------------------------")
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generate_passwords(args.model_path, args.num, device)