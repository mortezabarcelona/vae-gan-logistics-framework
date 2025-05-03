import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import sys

print("Starting train.py execution...")
sys.path.append('/Users/mortezapourjahangiri/PycharmProjects/PythonProject/vae_gan_logistics_framework/models')
from vae_gan_hybrid import VAEGAN

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Training function
def train_vae_gan(model, data_loader, num_epochs=10, learning_rate=1e-3, device='cpu'):
    print(f"Starting training on device: {device}")
    model.to(device)

    # Optimizers
    optimizer_vae = optim.Adam(model.vae.parameters(), lr=learning_rate)
    optimizer_discriminator = optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    optimizer_generator = optim.Adam(model.generator.parameters(), lr=learning_rate)

    # Loss functions
    reconstruction_loss = nn.MSELoss()
    adversarial_loss = nn.BCELoss()

    # Scenario vector as a 1D tensor
    scenario_vector = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss_vae = 0
        total_loss_discriminator = 0
        total_loss_generator = 0

        for batch_idx, (data, _) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)} with data shape: {data.shape}")
            data = data.to(device)
            batch_size = data.size(0)

            # Real and fake labels
            real_label = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
            fake_label = torch.zeros(batch_size, 1, dtype=torch.float32).to(device)

            # --- VAE-GAN Forward Pass ---
            recon, mu, logvar, validity = model(data, scenario_vector)

            # --- VAE Loss (Reconstruction + KL Divergence) ---
            recon_loss = reconstruction_loss(recon, data)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + kl_div
            total_loss_vae += vae_loss.item()

            # --- Discriminator Loss ---
            real_validity = model.discriminator(data.clone().detach())  # Detach to avoid backprop through VAE
            fake_validity = validity.clone().detach()  # Detach to avoid backprop through generator
            d_loss_real = adversarial_loss(real_validity, real_label)
            d_loss_fake = adversarial_loss(fake_validity, fake_label)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            total_loss_discriminator += d_loss.item()

            # --- Generator Loss ---
            g_loss = adversarial_loss(validity, real_label)  # Use original validity for generator loss
            total_loss_generator += g_loss.item()

            # --- Update Weights ---
            optimizer_vae.zero_grad()
            vae_loss.backward(retain_graph=True)
            optimizer_vae.step()

            optimizer_discriminator.zero_grad()
            d_loss.backward()
            optimizer_discriminator.step()

            optimizer_generator.zero_grad()
            g_loss.backward()
            optimizer_generator.step()

            # --- Feedback Loop: Update model with historical data ---
            model.update_feedback(data.clone().detach())

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{num_epochs}] VAE Loss: {total_loss_vae / len(data_loader):.4f}, "
              f"Discriminator Loss: {total_loss_discriminator / len(data_loader):.4f}, "
              f"Generator Loss: {total_loss_generator / len(data_loader):.4f}")

# Example usage (for testing)
if __name__ == "__main__":
    print("Loading data...")
    try:
        # Find the latest data file in the data directory
        data_dir = "../data/"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        data_files = [f for f in os.listdir(data_dir) if f.startswith("logistics_data_baseline_")]
        if not data_files:
            raise FileNotFoundError("No logistics_data_baseline_ files found in data directory")
        latest_file = max(data_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        data_path = os.path.join(data_dir, latest_file)
        print(f"Found data file: {data_path}")

        # Load sample data
        df = pd.read_csv(data_path)
        print(f"Data loaded with shape: {df.shape}")

        # Drop non-numeric columns
        non_numeric_cols = ['shipment_id', 'origin', 'destination', 'route', 'delivery_deadline', 'day_of_week', 'road_incident']
        df = df.drop(columns=non_numeric_cols)

        # Simple preprocessing (normalize numeric columns)
        numeric_cols = ['time_step', 'hour_of_day', 'demand_level', 'volume', 'weight', 'distance',
                        'weather_severity', 'traffic_congestion', 'port_congestion', 'transit_time',
                        'fuel_price', 'co2_emissions', 'fuel_cost', 'emission_penalty', 'shipment_urgency',
                        'customer_satisfaction', 'driver_fatigue']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            df[col] = df[col].fillna(0)

        # Convert categorical to one-hot
        categorical_cols = ['weather_condition', 'transport_mode', 'port_status']
        df = pd.get_dummies(df, columns=categorical_cols, dtype=float)

        # Verify all columns are numeric
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Convert to tensor
        data = torch.tensor(df.values, dtype=torch.float32)
        dataset = TensorDataset(data, torch.zeros(len(data)))
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"DataLoader created with {len(data_loader)} batches")

        # Initialize model
        model = VAEGAN(input_dim=data.shape[1], hidden_dim=128, latent_dim=10, scenario_dim=4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model initialized, using device: {device}")

        # Train
        train_vae_gan(model, data_loader, num_epochs=5, device=device)
        print("Training completed.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")