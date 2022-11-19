import streamlit as st
import torch
from src.acgan.nets import Classifier, Critic, Generator, weights_init
from src.acgan.utils import (combine_vectors, load_dataset, make_label,
                             make_noise, make_one_hot_labels, plot_tensor)
from src.data import DATA_DIR
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


class Model():
    """
    A class for training and testing the acgan network.
    """
    def __init__(self, n_classes=2, z_dim=20, n_repeats=2):

        st.title('ACGAN')
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.n_repeats = n_repeats

        self.load_model()
        self.create_optimizers()
        # criterion
        self.criterion = nn.BCEWithLogitsLoss()

    def load_model(self):
        """
        Load the generator, critic and classifier networks.
        """

        self.gen = Generator(z_dim=(self.z_dim + self.n_classes * self.n_repeats), hidden_dim=64, im_chan=1)
        self.classifier = Classifier(im_chan=1, hidden_dim=32)
        self.critic = Critic()

    def create_optimizers(
        self,
        lr_gen=0.002, lr_critic=0.001, lr_classifier=0.001,
        beta_1=0.5, beta_2=0.999
        ):
        """
        Create and initials the adam optimizers for the networks.

        Args:
            lr_gen (float, optional): generator learning rate. Defaults to 0.002.
            lr_critic (float, optional): critic learning rate. Defaults to 0.001.
            lr_classifier (float, optional): classifier learning rate. Defaults to 0.001.
            beta_1 (float, optional): momentum beta 1 param. Defaults to 0.5.
            beta_2 (float, optional): momentum beta 2. Defaults to 0.999.
        """
        ## create optimizers
        self.gen_opt = torch.optim.Adam(params=self.gen.parameters(), lr=lr_gen, betas=(beta_1, beta_2))
        self.classifier_opt = torch.optim.Adam(params=self.classifier.parameters(), lr=lr_classifier, betas=(beta_1, beta_2))
        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(), lr=lr_critic, betas=(beta_1, beta_2))

    def init_weights(self):
        """
        Initialize the networks weights.
        """
        # apply init values
        self.gen = self.gen.apply(weights_init)
        self.classifier = self.classifier.apply(weights_init)
        self.critic = self.critic.apply(weights_init)

    def train(self, train_dataset, batch_size=28, epochs=30, display_step=30, device='cpu'):
        """
        Train the model based on train_dataset.

        Args:
            train_dataset: Training dataset.
            batch_size (int, optional): batch size. Defaults to 28.
            epochs (int, optional): number of epochs. Defaults to 30.
            display_step (int, optional): display the progress on display_step. Defaults to 30.
            device (str, optional): computation device. Defaults to 'cpu'.
        """

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=None)
        self.init_weights()

        # Training loop
        cur_step = 0
        gen_loss_mean = 0
        classifier_loss_mean = 0
        critic_loss_mean = 0
        classifier_mean_error = 0

        for epoch in range(epochs):
            for real, labels in train_dataloader:
                cur_batch_size = len(real)
                real = real.to(device=device)

                # gen_labels
                gen_labels = make_label(cur_batch_size, self.n_classes)
                # one_hot
                one_hot_labels = make_one_hot_labels(gen_labels, n_classes=self.n_classes, n_repeats=self.n_repeats)

                #-----------
                ## Critic
                #-----------
                self.critic.zero_grad()
                z = make_noise(batch_size=cur_batch_size, z_dim=self.z_dim, device=device)
                gen_input = combine_vectors(one_hot_labels, z)
                fake = self.gen(gen_input) # G(z)

                fake_pred_class, fake_feature_map = self.classifier(fake.detach())
                real_pred_class, real_feature_map = self.classifier(real)

                fake_pred_source = self.critic(fake_feature_map.detach())
                real_pred_source = self.critic(real_feature_map.detach())

                critic_loss = self.criterion(fake_pred_source, torch.zeros_like(fake_pred_source)) + self.criterion(real_pred_source, torch.ones_like(fake_pred_source))
                critic_loss.backward(retain_graph=True)
                self.critic_opt.step()

                #-----------
                ## Classifier
                #-----------
                self.classifier.zero_grad()

                classifier_loss = self.criterion(fake_pred_class, gen_labels.float()) + self.criterion(real_pred_class, labels.float())

                classifier_loss.backward(retain_graph=True)
                self.classifier_opt.step()

                #-----------
                ## Genrator
                #-----------
                self.gen.zero_grad()
                z = make_noise(batch_size=cur_batch_size, z_dim=self.z_dim, device=device)
                gen_input = combine_vectors(one_hot_labels, z)
                fake = self.gen(gen_input) # G(z)
                fake_pred_class, fake_feature_map = self.classifier(fake) # Cl(z)
                fake_pred_source = self.critic(fake_feature_map)

                gen_loss = self.criterion(fake_pred_source, torch.ones_like(fake_pred_source)) + self.criterion(fake_pred_class, gen_labels.float())

                gen_loss.backward(retain_graph=True)
                self.gen_opt.step()

                #-------------
                # batch accuracy
                classifier_mean_error += (abs(real_pred_class - labels).sum() / cur_batch_size) * 100

                ## mean losses
                gen_loss_mean += gen_loss.item() / display_step
                classifier_loss_mean += classifier_loss.item() / display_step
                critic_loss_mean += critic_loss.item() / display_step

                cur_step += 1
                if cur_step % display_step == 0:
                    fake_images = plot_tensor(fake)
                    real_images = plot_tensor(real)

                    # streamlit dashboard
                    st.write(f'Epoch: {epoch} / {epochs}')
                    st.progress(epoch/epochs)
                    st.write(f'Classifier Mean error: {classifier_mean_error / display_step:.2f} %')
                    st.write(
                        'Generator Mean Loss:', f'{gen_loss_mean:.2f}', ' | ',
                        'Classifier Mean Loss:', f'{classifier_loss_mean:.2f}', ' | ',
                        'Critic Loss:', f'{critic_loss_mean:.2f}'
                    )

                    st.image([fake_images.numpy(), real_images.numpy()], width=300, caption=['Fake images', 'Real images'])

                    # reset the params
                    gen_loss_mean = 0
                    classifier_loss_mean = 0
                    critic_loss_mean = 0
                    classifier_mean_error = 0


    def test(self, test_dataset):
        """
        Test the model based on test_dataset.

        Args:
            test_dataset: Test dataset.

        Returns:
            Accuracy(float): Accuracy of the model on test dataset.
        """
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=None)
        with torch.no_grad():
            for real, targets in test_dataloader:
                preds, _ = self.classifier(real)
                err = abs(preds - targets).sum() / len(real)

                return (1 - err) * 100

    def cross_validate(self, dataset, k_folds=5):
        """
        K-fold cross validation of the model on dataset.

        Args:
            dataset: the dataset to test and train.
            k_folds (int, optional): number of folds. Defaults to 5.
        """
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)

        accuracy = {}
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            st.write(f'fold: {fold + 1} / {k_folds}')

            # train test dataset split
            train_subset = Subset(dataset, train_ids, )
            test_subset = Subset(dataset, test_ids)

            st.write(f"{10 * '_'}")
            st.header('Training')
            self.train(train_subset)

            st.header('Testing')
            acc = self.test(test_subset)
            accuracy[f'Fold: {fold + 1}'] = acc
            st.write(f'Fold {fold + 1} Accuracy: {acc:.2f} %')

        mean_accuracy = sum(accuracy.values()) / k_folds
        st.success('Test Completed Successfully!', icon="✅")
        st.write(f'Mean Accuracy: {mean_accuracy:.2f} %')


if __name__ == '__main__':
    # load data
    root = DATA_DIR / 'graz/stft_image_data'
    graz_dataset = load_dataset(root)

    gan_model = Model()
    gan_model.cross_validate(graz_dataset)
