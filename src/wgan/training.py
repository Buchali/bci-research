import torch
from src.data import DATA_DIR
from src.wgan.Critic import Critic
from src.wgan.Generator import Generator, make_noise
from src.wgan.gradient import get_gradient, gradient_penalty
from src.wgan.weights_init import weights_init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

z_dim = 100
batch_size = 28
# Load and Initialize the Networks
gen = Generator(z_dim=z_dim, hidden_dim=64, img_chan=1)
critic = Critic(img_chan=1, hidden_dim=32)

gen.apply(weights_init)
critic.apply(weights_init)

# Data Loading
root = DATA_DIR / 'graz/stft_image_data'
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5, inplace=False)])
graz_dataset = datasets.ImageFolder(root=root, transform=transform)
dataloader = DataLoader(graz_dataset, batch_size=batch_size, shuffle=False)

# Learning Parameters
epochs = 20
lr_gen = 0.002
lr_critic = 0.001
beta_1, beta_2 = 0.5, 0.999
c_lambda = 10

# Optimizers
gen_opt = torch.optim.Adam(params=gen.parameters(), lr=lr_gen, betas=(beta_1, beta_2))
critic_opt = torch.optim.Adam(params=critic.parameters(), lr=lr_critic, betas=(beta_1, beta_2))

## Training
device = 'cpu'
cur_batch_size = batch_size
cur_step = 0
gen_loss_mean = 0
critic_loss_mean = 0
display_step = 30

for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device=device)

        ## Genrator Training
        gen.zero_grad()
        z_1 = make_noise(batch_size=cur_batch_size, z_dim=z_dim, device=device)
        fake = gen(z_1) # G(z)
        fake_score = critic(fake) # C(G(z))
        gen_loss = -fake_score.mean() # Generator Loss
        gen_loss.backward()
        gen_opt.step()

        ## Critic Training
        critic.zero_grad()

        z_2 = make_noise(batch_size=cur_batch_size, z_dim=z_dim, device=device)
        fake_2 = gen(z_2) # G(z)
        fake_score = critic(fake_2.detach()) # C(G(z))
        real_score = critic(real) # C(x)
        # gradient penalty
        epsilon = torch.rand(cur_batch_size, 1, 1, 1, device=device, requires_grad=True)
        gradient = get_gradient(critic=critic, real=real, fake=fake_2.detach(), epsilon=epsilon)

        critic_loss = fake_score.mean() - real_score.mean() + c_lambda * gradient_penalty(gradient) # Critic Loss
        critic_loss.backward()
        critic_opt.step()

        # mean losses
        gen_loss_mean += gen_loss.item() / display_step
        critic_loss_mean += critic_loss.item() / display_step

        cur_step += 1
        if cur_step % display_step == 0:
            print(f'Epoch: {epoch}, Step: {cur_step}, Generator Mean Loss: {gen_loss_mean:.2f}, Critic Loss: {critic_loss_mean:.2f}')

