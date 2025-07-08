import numpy as np

from dotmap import DotMap
import torch
import torch.nn.init as init
import torch.nn as nn

from termcolor import colored
import seaborn as sns

sns.set()


class AE_Trainer:

    def __init__(self, optimizer, scheduler, writer, device="mps"):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = device

    def train(self, model, loss_fn, train_data_loader, test_data_loader, num_epochs,
              best_model_path):

        epoch = 0
        best_test_loss = np.inf
        while epoch < num_epochs:

            train_loss = []
            test_loss = []

            train_losses = DotMap(
                {"loss_nee": [], "loss_E0": [], "loss_rb": [], "loss_alpha": [], "loss_beta": []})
            # Example of iterating over the DataLoader in the training loop
            for batch in train_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                nee = batch['NEE'].to(self.device)

                nee_pred, k_pred, z = model(x)

                z_prior = torch.randn_like(z)
                # Compute loss
                loss_nee, loss_E0,loss_rb, loss_alpha, loss_beta = self.loss_function(nee_pred,
                                                                                                      nee.view(-1, 1),
                                                                                                      z, z_prior,
                                                                                                      k_pred, k,
                                                                                                      loss_fn)
                loss = loss_nee + loss_E0 + loss_rb + loss_alpha + loss_beta

                train_losses.loss_nee.append(loss_nee)
                train_losses.loss_E0.append(loss_E0)
                train_losses.loss_rb.append(loss_rb)
                train_losses.loss_alpha.append(loss_alpha)
                train_losses.loss_beta.append(loss_beta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())

            print(colored("Training Loss: {}".format(np.mean(train_loss)), "blue"))
            self.writer.add_scalar("Train Loss", np.mean(train_loss), epoch)

            for col in train_losses.keys():
                l = [x.cpu().detach().numpy() for x in train_losses[col]]
                print(col, np.mean(l), end=" ")
                self.writer.add_scalar(f"Train Loss [{col}]", np.mean(l), epoch)
            print("\n")

            test_losses = DotMap(
                {"loss_nee": [], "loss_E0": [], "loss_rb": [], "loss_alpha": [], "loss_beta": []})

            for batch in test_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                nee = batch['NEE'].to(self.device)

                nee_pred, k_pred, z = model(x)

                z_prior = torch.randn_like(z)
                # Compute loss
                loss_nee, loss_E0, loss_rb, loss_alpha, loss_beta = self.loss_function(nee_pred,
                                                                                                      nee.view(-1, 1),
                                                                                                      z, z_prior,
                                                                                                      k_pred, k,
                                                                                                      loss_fn)
                loss = loss_nee + loss_E0 + loss_rb + loss_alpha + loss_beta

                test_loss.append(loss.cpu().detach().numpy())

                test_losses.loss_nee.append(loss_nee)
                test_losses.loss_E0.append(loss_E0)
                test_losses.loss_rb.append(loss_rb)
                test_losses.loss_alpha.append(loss_alpha)
                test_losses.loss_beta.append(loss_beta)

            print(colored("Test Loss: {}".format(np.mean(test_loss)), "red"))
            self.writer.add_scalar("Test Loss", np.mean(test_loss), epoch)

            for col in test_losses.keys():
                l = [x.cpu().detach().numpy() for x in test_losses[col]]
                print(col, np.mean(l), end=" ")
                self.writer.add_scalar(f"Test Loss [{col}]", np.mean(l), epoch)
            print("\n\n")

            # Save best model
            if epoch % 5 == 0 and np.mean(test_loss) < best_test_loss:
                best_test_loss = np.mean(test_loss)
                torch.save(model.state_dict(), best_model_path)
                print(colored(f'New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.4f}',
                              "light_grey"))

            self.scheduler.step(np.mean(test_loss))
            epoch += 1

    def loss_function(self, nee_pred, nee_true, latent, z_prior, k_pred, k_true, loss_fn):
        # MSE Loss for NEE (u)
        loss_nee = loss_fn(nee_pred, nee_true) + loss_fn(latent, z_prior)

        E0_pred, rb_pred, alpha_pred, beta_pred = k_pred[:, 0], k_pred[:, 1], k_pred[:, 2], k_pred[:, 3]
        E0_true, rb_true, alpha_true, beta_true = k_true[:, 0], k_true[:, 1], k_true[:, 2], k_true[:, 3]

        loss_E0 = loss_fn(E0_pred.view((-1, 1)), E0_true.view((-1, 1)))
        loss_rb = loss_fn(rb_pred.view((-1, 1)), rb_true.view((-1, 1)))
        loss_alpha = loss_fn(alpha_pred.view((-1, 1)), alpha_true.view((-1, 1)))
        loss_beta = loss_fn(beta_pred.view((-1, 1)), beta_true.view((-1, 1)))

        return loss_nee, loss_E0, loss_rb, loss_alpha, loss_beta

    def predict(self, model, test_data_loader):
        preds = DotMap({"nee": [], "E0": [], "rb": [], "alpha": [], "beta": []})
        gt = DotMap({"nee": [], "E0": [], "rb": [], "alpha": [], "beta": []})

        for batch in test_data_loader:
            x = batch['X'].to(self.device)
            k = batch['k'].to(self.device)
            nee = batch['NEE'].to(self.device)

            nee_pred, k_pred, z = model(x)
            E0_pred, rb_pred, alpha_pred, beta_pred = k_pred[:, 0], k_pred[:, 1], k_pred[:, 2], k_pred[:, 3]

            preds.nee.extend(nee_pred.cpu().detach().numpy().tolist())
            preds.E0.extend(E0_pred.cpu().detach().numpy().tolist())
            preds.rb.extend(rb_pred.cpu().detach().numpy().tolist())
            preds.alpha.extend(alpha_pred.cpu().detach().numpy().tolist())
            preds.beta.extend(beta_pred.cpu().detach().numpy().tolist())

            gt.nee.extend(nee.cpu().detach().numpy().tolist())
            gt.E0.extend(k[:, 0].cpu().detach().numpy().tolist())
            gt.rb.extend(k[:, 1].cpu().detach().numpy().tolist())
            gt.alpha.extend(k[:, 2].cpu().detach().numpy().tolist())
            gt.beta.extend(k[:, 3].cpu().detach().numpy().tolist())

        for col in preds:
            preds[col] = np.array(preds[col])
            if len(preds[col].shape) > 1 and preds[col].shape[1] == 1:
                preds[col] = preds[col].flatten()
        for col in gt:
            gt[col] = np.array(gt[col])
            if len(gt[col].shape) > 1 and gt[col].shape[1] == 1:
                gt[col] = gt[col].flatten()

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        # Apply He initialization
        nn.init.xavier_uniform_(layer.weight)
        # init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(layer.bias)


class AE_Model(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_dims, decoder_dims, activation=nn.Tanh, device="mps"):
        super(AE_Model, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.device = device

        # Encoder network
        modules = self.append_linear_modules(self.input_dim, self.encoder_dims)
        modules.append(nn.Linear(self.decoder_dims[-1], self.latent_dim))
        self.encoder = nn.Sequential(*modules)

        # Decoder network for NEE (u)
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims)
        modules.append(nn.Linear(self.decoder_dims[-1], 1))
        self.nee_decoder = nn.Sequential(*modules)

        # Decoder network for E0, rb, alpha and beta
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims)
        modules.append(nn.Linear(self.decoder_dims[-1], 4))
        self.k_decoder = nn.Sequential(*modules)

    def append_linear_modules(self, in_dim, dims):
        modules = []
        for i, dim in enumerate(dims):
            modules.append(nn.Linear(in_dim, dim))

            modules.append(self.activation())
            in_dim = dim
        return modules

    def forward(self, x, b, k):
        input_ = torch.cat((x, b.view(x.shape[0], 1), k), dim=1).to(self.device)
        z = self.encoder(input_)
        k = self.k_decoder(z)

        nee = self.nee_decoder(z)

        return nee, k, z
