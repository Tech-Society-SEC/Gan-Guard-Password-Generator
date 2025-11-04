import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Import our custom modules
from data.utils import get_dataloader, VOCAB_SIZE, MAX_LEN
from models import Generator, Discriminator, LATENT_DIM

# --- CONTROL RESUMING ---
RESUME = False 
START_EPOCH = 50
# -------------------------

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # --- Hyperparameters ---
    LEARNING_RATE_D = 0.0001
    LEARNING_RATE_G = 0.0001
    BETA1 = 0.5
    BATCH_SIZE = 512 # Increased batch size for the smaller dataset
    EPOCHS = 50 
    DATASET_PATH = "rockyou_expert_processed.pt" # <-- Using the new expert dataset

    # --- Initialization ---
    dataloader = get_dataloader(DATASET_PATH, BATCH_SIZE)
    generator = Generator()
    discriminator = Discriminator()

    if RESUME:
        print(f"Resuming training from epoch {START_EPOCH}...")
        generator.load_state_dict(torch.load(f'generator_epoch_{START_EPOCH}.pth'))
        discriminator.load_state_dict(torch.load(f'discriminator_epoch_{START_EPOCH}.pth'))

    generator.to(device)
    discriminator.to(device)

    # --- Setup for Training ---
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))

    print("✅ Setup complete. Starting the training process...")

    # --- Main Training Loop ---
    for epoch in range(START_EPOCH, EPOCHS):
        for i, real_passwords_int in enumerate(dataloader):
            real_passwords_int = real_passwords_int.to(device)
            
            # --- Train Discriminator ---
            discriminator.zero_grad()
            real_passwords_one_hot = F.one_hot(real_passwords_int, num_classes=VOCAB_SIZE).float()
            real_labels = torch.full((real_passwords_int.size(0), 1), 0.9, device=device)
            output_real = discriminator(real_passwords_one_hot)
            lossD_real = criterion(output_real, real_labels)
            lossD_real.backward()
            
            noise = torch.randn(real_passwords_int.size(0), LATENT_DIM).to(device)
            fake_passwords_for_D = generator(noise)
            fake_labels = torch.zeros(real_passwords_int.size(0), 1).to(device)
            output_fake = discriminator(fake_passwords_for_D.detach())
            lossD_fake = criterion(output_fake, fake_labels)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # --- Train Generator (2 times) ---
            for _ in range(2):
                real_labels_for_G = torch.ones(real_passwords_int.size(0), 1).to(device)
                generator.zero_grad()
                
                noise = torch.randn(real_passwords_int.size(0), LATENT_DIM).to(device)
                fake_passwords = generator(noise)
                output_fooled = discriminator(fake_passwords)
                lossG = criterion(output_fooled, real_labels_for_G)
                lossG.backward()
                optimizerG.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], "
                    f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}"
                )

        print(f"--- End of Epoch {epoch+1} ---")

        # --- SAVE CHECKPOINT AFTER EVERY EPOCH ---
        torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')
        print(f"✅ Checkpoint saved for epoch {epoch+1}")

    print("🏁 Training Finished!")