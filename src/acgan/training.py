import torch
from src.acgan.nets import Classifier, Critic, Generator, weights_init
from src.acgan.utils import (combine_vectors, make_label, make_noise,
                             make_one_hot_labels)
from src.data import DATA_DIR
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


# Create Networks
z_dim = 20
n_repeats = 2
n_classes = 2
batch_size = 28
device = 'cpu'

gen = Generator(z_dim=(z_dim + n_classes * n_repeats), hidden_dim=64, im_chan=1)
classifier = Classifier(im_chan=1, hidden_dim=32)
critic = Critic()

# Optimizers
lr_gen = 0.002
lr_critic = 0.001
lr_classifier = 0.001
beta_1, beta_2 = 0.5, 0.999

## create optimizers
gen_opt = torch.optim.Adam(params=gen.parameters(), lr=lr_gen, betas=(beta_1, beta_2))
classifier_opt = torch.optim.Adam(params=classifier.parameters(), lr=lr_classifier, betas=(beta_1, beta_2))
critic_opt = torch.optim.Adam(params=critic.parameters(), lr=lr_critic, betas=(beta_1, beta_2))

# criterion
criterion = nn.BCEWithLogitsLoss()

# load data
root = DATA_DIR / 'graz/stft_image_data'
transform =  transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(0, 1, inplace=False)
    ]
)
graz_dataset = datasets.ImageFolder(root=root, transform=transform)
dataloader = DataLoader(graz_dataset, batch_size=batch_size, shuffle=None)

# apply init values
gen = gen.apply(weights_init)
classifier = classifier.apply(weights_init)
critic = critic.apply(weights_init)

# Training loop
epochs = 30
c_lambda = 10
display_step = 30
device = 'cpu'

cur_step = 0
gen_loss_mean = 0
classifier_loss_mean = 0
critic_loss_mean = 0
classifier_mean_error = 0


for epoch in range(epochs):
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device=device)

        # gen_labels
        gen_labels = make_label(cur_batch_size, n_classes)
        # one_hot
        one_hot_labels = make_one_hot_labels(gen_labels, n_classes=n_classes, n_repeats=n_repeats)

        #-----------
        ## Critic
        #-----------
        critic.zero_grad()
        z = make_noise(batch_size=cur_batch_size, z_dim=z_dim, device=device)
        gen_input = combine_vectors(one_hot_labels, z)
        fake = gen(gen_input) # G(z)

        fake_pred_class, fake_feature_map = classifier(fake.detach())
        real_pred_class, real_feature_map = classifier(real)

        fake_pred_source = critic(fake_feature_map.detach())
        real_pred_source = critic(real_feature_map.detach())

        critic_loss = criterion(fake_pred_source, torch.zeros_like(fake_pred_source)) + criterion(real_pred_source, torch.ones_like(fake_pred_source))
        critic_loss.backward(retain_graph=True)
        critic_opt.step()

        #-----------
        ## Classifier
        #-----------
        classifier.zero_grad()

        classifier_loss = criterion(fake_pred_class, gen_labels.float()) + criterion(real_pred_class, labels.float()) 

        classifier_loss.backward(retain_graph=True)
        classifier_opt.step()

        #-----------
        ## Genrator
        #-----------
        gen.zero_grad()
        z = make_noise(batch_size=cur_batch_size, z_dim=z_dim, device=device)
        gen_input = combine_vectors(one_hot_labels, z)
        fake = gen(gen_input) # G(z)
        fake_pred_class, fake_feature_map = classifier(fake) # Cl(z)
        fake_pred_source = critic(fake_feature_map)

        gen_loss = criterion(fake_pred_source, torch.ones_like(fake_pred_source)) + criterion(fake_pred_class, gen_labels.float())

        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        #-------------
        # batch accuracy
        classifier_mean_error += (abs(real_pred_class - labels).sum() / cur_batch_size) * 100

        ## mean losses
        gen_loss_mean += gen_loss.item() / display_step
        classifier_loss_mean += classifier_loss.item() / display_step
        critic_loss_mean += critic_loss.item() / display_step

        cur_step += 1
        if cur_step % display_step == 0:
            print(
                f'Epoch: {epoch} / {epochs}, Step: {cur_step} \n' +
                f'Classifier Mean error: {classifier_mean_error / display_step:.2f} % \n' +
                f'Generator Mean Loss: {gen_loss_mean:.2f} \n' +
                f'Classifier Mean Loss: {classifier_loss_mean:.2f} \n' +
                f'Critic Loss: {critic_loss_mean:.2f}'
            )
            # show_tensor_images(fake)
            # show_tensor_images(real)
            gen_loss_mean = 0
            classifier_loss_mean = 0
            critic_loss_mean = 0
            classifier_mean_error = 0
