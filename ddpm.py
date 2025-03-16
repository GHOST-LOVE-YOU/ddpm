import torch
from torch.nn import functional as F
from tqdm.auto import tqdm


def linear_beta_schedule(timesteps) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPMSampler:
    def __init__(self, timesteps=300):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def ddim_p_sample(self, model, x, t, t_index, t_next=None, eta=0.0):
        device = x.device
        b = x.shape[0]

        # Extract values for current timestep
        alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # Predict noise
        predicted_noise = model(x, t)

        # Predict x0 from xt and predicted noise
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / torch.sqrt(
            alpha_cumprod_t
        )

        # Compute the timestep for the next sampling step
        if t_next is None:
            t_next = torch.full((b,), t_index - 1, device=device, dtype=torch.long)

        # If we're at the last step, just return the predicted x0
        if t_index == 0:
            return pred_x0

        # Get alpha values for the next timestep
        alpha_cumprod_next = self.extract(self.alphas_cumprod, t_next, x.shape)

        # Compute sigma for current step (controls stochasticity)
        # When eta=0, sigma=0 (deterministic)
        # When eta=1, equivalent to DDPM
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_next)
            / (1 - alpha_cumprod_t)
            * (1 - alpha_cumprod_t / alpha_cumprod_next)
        )

        coeff_x0 = torch.sqrt(alpha_cumprod_next)
        coeff_eps = torch.sqrt(1 - alpha_cumprod_next - sigma_t**2)

        # DDIM sampling
        if eta > 0:
            noise = torch.randn_like(x)
            x_next = coeff_x0 * pred_x0 + coeff_eps * predicted_noise + sigma_t * noise
        else:
            # Deterministic DDIM (eta=0)
            x_next = coeff_x0 * pred_x0 + coeff_eps * predicted_noise

        return x_next

    @torch.no_grad()
    def ddim_p_sample_loop(self, model, shape, n_steps=None, eta=0.0):
        device = next(model.parameters()).device
        b = shape[0]

        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        # If n_steps is not provided, use all timesteps
        if n_steps is None:
            n_steps = self.timesteps
            time_steps = list(range(self.timesteps))
        else:
            # Uniformly spaced steps for DDIM
            skip = self.timesteps // n_steps
            time_steps = list(range(0, self.timesteps, skip))
            time_steps = sorted(time_steps, reverse=True)

        for i, step in enumerate(tqdm(time_steps, desc="DDIM sampling loop")):
            # Store only the final predicted image
            if i == len(time_steps):
                return imgs
            t = torch.full((b,), step, device=device, dtype=torch.long)

            next_step = time_steps[i + 1] if i < len(time_steps) - 1 else 0
            t_next = torch.full((b,), next_step, device=device, dtype=torch.long)
            img = self.ddim_p_sample(model, img, t, step, t_next=t_next, eta=eta)
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def ddim_sample(
        self, model, image_size, batch_size=16, channels=3, n_steps=50, eta=0.0
    ):
        return self.ddim_p_sample_loop(
            model,
            shape=(batch_size, channels, image_size, image_size),
            n_steps=n_steps,
            eta=eta,
        )
