import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

# Import our custom modules
from data.utils import get_dataloader, VOCAB_SIZE, MAX_LEN
from models import Generator, Critic, LATENT_DIM # Imports the Critic

# --- CONTROL RESUMING ---
RESUME = True
START_EPOCH = 13
# -------------------------

# --- WGAN-GP Hyperparameters ---
LEARNING_RATE = 0.0001
BETA1 = 0.5
BETA2 = 0.9
BATCH_SIZE = 128
EPOCHS = 100 
# ‼️ --- (THE FIX) --- ‼️
# Pointing to our new, fast, "common" dataset
DATASET_PATH = "rockyou_common_subset_processed.pt" 
CHECKPOINT_DIR = 'models_wgan_common' # New folder for this experiment

# --- WGAN-GP Specific Parameters ---
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP"""
    alpha = torch.randn(real_samples.size(0), 1, 1).to(device)
    alpha = alpha.expand(real_samples.size())
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    
    # --- (FIX for CuDNN double backward error) ---
    with torch.backends.cudnn.flags(enabled=False):
        critic_interpolates = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # --- (FIX for .view() runtime error) ---
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_GP
    return gradient_penalty

# --- Main Training ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    # --- Initialization ---
    dataloader = get_dataloader(DATASET_PATH, BATCH_SIZE)
    generator = Generator().to(device)
    critic = Critic().to(device)

    # --- Setup for Training ---
    optimizerC = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizerG = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # --- RESUME LOGIC ---
    if RESUME:
        if START_EPOCH == 0:
            print("⚠️ Warning: RESUME is True but START_EPOCH is 0. Starting from scratch.")
        else:
            print(f"Resuming training from epoch {START_EPOCH}...")
            gen_path = os.path.join(CHECKPOINT_DIR, f'generator_epoch_{START_EPOCH}.pth')
            crit_path = os.path.join(CHECKPOINT_DIR, f'critic_epoch_{START_EPOCH}.pth')
            
            try:
                generator.load_state_dict(torch.load(gen_path))
                critic.load_state_dict(torch.load(crit_path))
                print(f"✅ Successfully loaded checkpoints from epoch {START_EPOCH}")
            except Exception as e:
                print(f"❌ Error loading checkpoints: {e}")
                print("Starting from scratch instead.")
                START_EPOCH = 0 # Reset to 0 if loading failed

    print("✅ Setup complete. Starting the WGAN-GP training process...")
    print(f"   Critic will train {CRITIC_ITERATIONS} times for every 1 Generator update.")

    # --- Main Training Loop ---
    for epoch in range(START_EPOCH, EPOCHS):
        for i, real_passwords_int in enumerate(dataloader):
            real_passwords_int = real_passwords_int.to(device)
            current_batch_size = real_passwords_int.size(0)

            # --- Train Critic ---
            for _ in range(CRITIC_ITERATIONS):
                critic.zero_grad()
                real_passwords_one_hot = F.one_hot(real_passwords_int, num_classes=VOCAB_SIZE).float()
                noise = torch.randn(current_batch_size, LATENT_DIM).to(device)
                fake_passwords = generator(noise)
                critic_real = critic(real_passwords_one_hot)
                critic_fake = critic(fake_passwords.detach())
                gradient_penalty = compute_gradient_penalty(critic, real_passwords_one_hot, fake_passwords, device)
                lossC = torch.mean(critic_fake) - torch.mean(critic_real) + gradient_penalty
                lossC.backward()
                optimizerC.step()

            # --- Train Generator ---
            generator.zero_grad()
            noise_g = torch.randn(current_batch_size, LATENT_DIM).to(device)
            fake_passwords_g = generator(noise_g)
            critic_fake_g = critic(fake_passwords_g)
            lossG = -torch.mean(critic_fake_g)
            lossG.backward()
            optimizerG.step()

            # --- Print Logs ---
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], "
                    f"Loss C: {lossC.item():.4f}, Loss G: {lossG.item():.4f}"
                )

        print(f"--- End of Epoch {epoch+1} ---")

        # --- SAVE CHECKPOINT ---
        gen_save_path = os.path.join(CHECKPOINT_DIR, f'generator_epoch_{epoch+1}.pth')
        critic_save_path = os.path.join(CHECKPOINT_DIR, f'critic_epoch_{epoch+1}.pth')
        
        torch.save(generator.state_dict(), gen_save_path)
        torch.save(critic.state_dict(), critic_save_path)
        print(f"✅ Checkpoint saved for epoch {epoch+1} to {CHECKPOINT_DIR}")

    print("🏁 Training Finished!")