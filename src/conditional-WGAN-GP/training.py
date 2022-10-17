import matplotlib.pyplot as plt
import torch
from src.data import DATA_DIR
from src.wgan.Critic import Critic
from src.wgan.Generator import Generator, make_noise
from src.wgan.gradient import get_gradient, gradient_penalty
from src.wgan.one_hot import make_one_hot_labels
from src.wgan.visualize import plot_tensor
from src.wgan.weights_init import weights_init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


z_dim = 18
n_classes = 2
batch_size = 28
# Load and Initialize the Networks
gen = Generator(z_dim=z_dim+n_classes, hidden_dim=64, img_chan=1)
critic = Critic(img_chan=3, hidden_dim=32)

gen = gen.apply(weights_init)
critic = critic.apply(weights_init)

# Data Loading
root = DATA_DIR / 'graz/stft_image_data'
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(0.5, 0.5, inplace=False)])
graz_dataset = datasets.ImageFolder(root=root, transform=transform)
dataloader = DataLoader(graz_dataset, batch_size=batch_size, shuffle=True)

# Learning Parameters
lr_gen = 0.002
lr_critic = 0.001
beta_1, beta_2 = 0.5, 0.999
c_lambda = 10

# Optimizers
gen_opt = torch.optim.Adam(params=gen.parameters(), lr=lr_gen, betas=(beta_1, beta_2))
critic_opt = torch.optim.Adam(params=critic.parameters(), lr=lr_critic, betas=(beta_1, beta_2))

## Training
epochs = 20
device = 'cpu'
cur_step = 0
gen_loss_mean = 0
critic_loss_mean = 0
display_step = 30

for epoch in range(epochs):
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device=device)

        # one_hot
        one_hot_labels = make_one_hot_labels(labels, n_classes=2)
        one_hot_images = one_hot_labels[:, :, None, None].repeat(1, 1, real.shape[2], real.shape[3])

        ## Genrator Training
        gen.zero_grad()
        z_1 = make_noise(batch_size=cur_batch_size, z_dim=z_dim, device=device)
        noise_and_labels = torch.cat([z_1, one_hot_labels], dim=1)
        fake = gen(noise_and_labels) # G(z)
        fake_images_and_labels = torch.cat([fake, one_hot_images], dim=1)
        fake_score = critic(fake_images_and_labels) # C(G(z))
        gen_loss = -fake_score.mean() # Generator Loss
        gen_loss.backward()
        gen_opt.step()

        ## Critic Training
        critic.zero_grad()

        z_2 = make_noise(batch_size=cur_batch_size, z_dim=z_dim, device=device)
        noise_and_labels = torch.cat([z_2, one_hot_labels], dim=1)
        fake = gen(noise_and_labels) # G(z)

        fake_images_and_labels = torch.cat([fake, one_hot_images], dim=1)
        fake_score = critic(fake_images_and_labels.detach()) # C(G(z))

        real_images_and_labels = torch.cat([real, one_hot_images], dim=1)
        real_score = critic(real_images_and_labels) # C(x)

        # gradient penalty
        epsilon = torch.rand(cur_batch_size, 1, 1, 1, device=device, requires_grad=True)
        gradient = get_gradient(critic=critic, real=real_images_and_labels, fake=fake_images_and_labels.detach(), epsilon=epsilon)

        critic_loss = fake_score.mean() - real_score.mean() + c_lambda * gradient_penalty(gradient) # Critic Loss
        critic_loss.backward()
        critic_opt.step()

        # mean losses
        gen_loss_mean += gen_loss.item() / display_step
        critic_loss_mean += critic_loss.item() / display_step

        cur_step += 1
        if cur_step % display_step == 0:
            print(f'Epoch: {epoch}, Step: {cur_step}, Generator Mean Loss: {gen_loss_mean:.2f}, Critic Loss: {critic_loss_mean:.2f}')
            gen_loss_mean = 0
            critic_loss_mean = 0

# Test
num_img = 20
z_1 = make_noise(batch_size=num_img, z_dim=z_dim, device=device)
one_hot_labels = make_one_hot_labels(torch.zeros([num_img]), n_classes=2)

noise_and_labels = torch.cat([z_1, one_hot_labels], dim=1)
fake_1 = gen(noise_and_labels)

z_2 = make_noise(batch_size=num_img, z_dim=z_dim, device=device)
one_hot_labels = make_one_hot_labels(torch.ones([num_img]), n_classes=2)
noise_and_labels = torch.cat([z_2, one_hot_labels], dim=1)
fake_2 = gen(noise_and_labels)

fake_image_1 = plot_tensor(fake_1)
fake_image_2 = plot_tensor(fake_2)

plt.subplot(121)
plt.imshow(fake_image_1)
plt.subplot(122)
plt.imshow(fake_image_2)
plt.show()
