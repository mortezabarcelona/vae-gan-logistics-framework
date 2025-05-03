import torch
import torch.nn as nn
from vae import VAE
from gan import Generator, Discriminator

class VAEGAN(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, latent_dim=10, scenario_dim=4):
        super(VAEGAN, self).__init__()

        # VAE components
        self.vae = VAE(input_dim, hidden_dim, latent_dim)

        # GAN components
        self.generator = Generator(latent_dim + scenario_dim, hidden_dim, input_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)

        # Feedback loop buffer (placeholder for historical data)
        self.feedback_buffer = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, scenario_vector):
        # Ensure scenario_vector is a 1D tensor and broadcast to match batch size
        batch_size = x.size(0)
        scenario_tensor = scenario_vector.clone().detach().to(dtype=torch.float32, device=x.device)
        if len(scenario_tensor.shape) == 2 and scenario_tensor.shape[0] == 1:
            scenario_tensor = scenario_tensor.squeeze(0)  # Remove extra dimension if present
        scenario_tensor = scenario_tensor.expand(batch_size, -1)  # Broadcast to match batch size

        # VAE encoding
        h = self.vae.encoder(x)
        mu = self.vae.fc_mu(h)
        logvar = self.vae.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Concatenate scenario vector to latent space
        z_conditioned = torch.cat([z, scenario_tensor], dim=1)

        # Generate reconstructed data
        x_recon = self.generator(z_conditioned)

        # Discriminator output
        validity = self.discriminator(x_recon.clone().detach())  # Avoid in-place and detach

        return x_recon, mu, logvar, validity

    def update_feedback(self, historical_data):
        # Simple feedback loop: store historical data for next iteration
        self.feedback_buffer = historical_data.detach().clone()

    def get_feedback(self):
        # Retrieve feedback for training adjustment
        return self.feedback_buffer if self.feedback_buffer is not None else torch.zeros(1, historical_data.size(1), device=historical_data.device)