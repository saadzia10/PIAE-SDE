import torch
from mmd_loss import MMD_loss
import numpy as np

from dotmap import DotMap
import torch
import torch.nn.init as init
import torch.nn as nn

from termcolor import colored
import seaborn as sns
sns.set()


class PIAE_SDE_Reg_Sampling_Trainer:

    def __init__(self, optimizer, scheduler, writer, device="mps"):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = device

    def train(self, model, loss_fn, train_data_loader, test_data_loader, noise_std, noise_mu, num_epochs, best_model_path, f_factor=1,
              mse_factor=1, mmd_factor=1):

        epoch = 0
        best_test_loss = np.inf
        while epoch < num_epochs:

            print(colored("Epoch: {}".format(epoch), "red"))
            train_loss = []
            test_loss = []

            train_losses = DotMap(
                {"mse_loss_nee": [], "mmd_loss_nee": [], "mse_loss_E0": [], "mse_loss_rb": [], "mse_temp_loss": [],
                 "physics_loss": [], "noise_loss": [], "mse_f_loss": []})
            # Example of iterating over the DataLoader in the training loop

            for batch in train_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                f = batch['dNEE'].to(self.device).view(-1, 1)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)
                nee = batch['NEE'].to(self.device).view(-1, 1)

                nee_pred, dT_dt_pred, k_pred, f_pred, z, residual, noise = model(x, b, k, T)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

                noise_prior = torch.randn_like(noise) * noise_std + noise_mu

                # Compute loss

                mse_loss_nee, mse_loss_E0, mse_loss_rb, mse_loss_temp, physics_loss, mse_loss_f, noise_loss = self.loss_function(
                    nee_pred, nee, noise, noise_prior, dT_dt_pred, dtemp, k_pred, k, f_pred, f, residual, loss_fn)
                mse_loss_f = mse_loss_f * f_factor
                mse_loss = mse_loss_nee + mse_loss_E0 + mse_loss_rb + mse_loss_temp + physics_loss + mse_loss_f
                mmd_loss_nee = MMD_loss()(nee, nee_pred)

                loss = mse_factor * mse_loss + mmd_factor * noise_loss + mmd_factor * mmd_loss_nee

                train_losses.mse_loss_nee.append(mse_loss_nee)
                train_losses.mmd_loss_nee.append(mmd_loss_nee)
                train_losses.noise_loss.append(noise_loss)
                train_losses.mse_loss_E0.append(mse_loss_E0)
                train_losses.mse_loss_rb.append(mse_loss_rb)
                train_losses.mse_temp_loss.append(mse_loss_temp)
                train_losses.physics_loss.append(physics_loss)
                train_losses.mse_f_loss.append(mse_loss_f)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())

            print(colored("Training Loss: {}".format(np.mean(train_loss)), "blue"))

            self.writer.add_scalar(f"Train Loss", np.mean(train_loss), epoch)

            for col in train_losses.keys():
                l = [x.cpu().detach().numpy() for x in train_losses[col]]
                print(col, np.mean(l), end=" ")
                self.writer.add_scalar(f"Train Loss [{col}]", np.mean(l), epoch)
            print("\n")

            test_losses = DotMap(
                {"mse_loss_nee": [], "mmd_loss_nee": [], "mse_loss_E0": [], "mse_loss_rb": [], "mse_temp_loss": [],
                 "physics_loss": [], "noise_loss": [], "mse_f_loss": []})

            for batch in test_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                f = batch['dNEE'].to(self.device).view(-1, 1)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)
                nee = batch['NEE'].to(self.device).view(-1, 1)

                nee_pred, dT_dt_pred, k_pred, f_pred, z, residual, noise = model(x, b, k, T)

                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

                noise_prior = torch.randn_like(noise) * noise_std + noise_mu

                # Compute loss
                mse_loss_nee, mse_loss_E0, mse_loss_rb, mse_loss_temp, physics_loss, mse_loss_f, noise_loss = self.loss_function(
                    nee_pred, nee, noise, noise_prior, dT_dt_pred, dtemp, k_pred, k, f_pred, f, residual, loss_fn)
                mse_loss_f = mse_loss_f * f_factor
                mse_loss = mse_loss_nee + mse_loss_E0 + mse_loss_rb + mse_loss_temp + physics_loss + mse_loss_f
                mmd_loss_nee = MMD_loss()(nee, nee_pred)

                loss = mse_factor * mse_loss + mmd_factor * noise_loss + mmd_factor * mmd_loss_nee

                test_losses.mse_loss_nee.append(mse_loss_nee)
                test_losses.mmd_loss_nee.append(mmd_loss_nee)
                test_losses.noise_loss.append(noise_loss)
                test_losses.mse_loss_E0.append(mse_loss_E0)
                test_losses.mse_loss_rb.append(mse_loss_rb)
                test_losses.mse_temp_loss.append(mse_loss_temp)
                test_losses.physics_loss.append(physics_loss)
                test_losses.mse_f_loss.append(mse_loss_f)

            test_loss.append(loss.cpu().detach().numpy())

            print(colored("Test Loss: {}".format(np.mean(test_loss)), "red"))
            self.writer.add_scalar(f"Test Loss", np.mean(test_loss), epoch)

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

    def loss_function(self, nee_pred, nee_true, noise, noise_prior, temp_pred, temp_true, E0_rb_pred, E0_rb_true, f_pred, f_true, physics_residual, loss_fn):
        # Loss for NEE (u)
        loss_nee = loss_fn(nee_pred, nee_true) 
    
        # Loss for dNEE (f)
        f_loss = loss_fn(f_pred, f_true)
    
        # MMD Loss on Noise
        noise_loss = MMD_loss()(noise, noise_prior)
        
        # Loss for E0 and rb (k)
        E0_pred, rb_pred = E0_rb_pred[:, 0], E0_rb_pred[:, 1]
        E0_true, rb_true = E0_rb_true[:, 0], E0_rb_true[:, 1]
    
        loss_E0 = loss_fn(E0_pred.view((-1, 1)), E0_true.view((-1, 1)))
        loss_rb = loss_fn(rb_pred.view((-1, 1)), rb_true.view((-1, 1)))
        
        # loss for temperature derivative (f)
        temp_loss = loss_fn(temp_pred.view((-1, 1)), temp_true.view((-1, 1)))
        
        # Physics-based loss (ensure the solution satisfies the physics model)
        physics_loss = torch.mean(physics_residual ** 2)
        
        # Total loss
        # total_loss = loss_nee + loss_E0 + loss_rb + temp_loss + physics_loss + f_loss
        return loss_nee, loss_E0, loss_rb , temp_loss , physics_loss , f_loss, noise_loss

    def predict(self, model, test_data_loader):
        preds = DotMap({"nee": [], "E0": [], "rb": [], "dtemp": [], "f": [], "z": [], "noise": [], "noise_mus": [],
                        "noise_stds": []})
        gt = DotMap({"nee": [], "E0": [], "rb": [], "dtemp": [], "f": []})

        for batch in test_data_loader:
            x = batch['X'].to(self.device)
            k = batch['k'].to(self.device)
            f = batch['dNEE'].to(self.device)
            b = batch['bNEE'].to(self.device)
            T = batch['T'].to(self.device)
            dtemp = batch['dT'].to(self.device)
            nee = batch['NEE'].to(self.device)

            input_ = torch.cat((x, b.view(x.shape[0], 1), k), dim=1).to(self.device)
            z = model.encoder(input_)
            noise_mu = model.fc_mu(z)
            noise_logvar = model.fc_logvar(z)
            noise = model.reparameterize(noise_mu, noise_logvar)

            nee_pred = model.nee_decoder(z) + noise
            dT_dt_pred = model.temp_derivative_decoder(z)
            k_pred = model.k_decoder(z)
            f_pred, residual = model.physics_residual(nee_pred, k_pred, T.view((-1, 1)), dT_dt_pred)

            # nee_pred, dT_dt_pred, k_pred, f_pred, z, residual, noise = model(x, b, k)
            E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

            # z = model.encoder(x, b, k)
            # noise_mu = model.fc_mu(z)
            # noise_logvar = model.fc_logvar(z)
            noise_std = torch.exp(0.5 * noise_logvar)

            preds.nee.extend(nee_pred.cpu().detach().numpy().tolist())
            preds.noise.extend(noise.cpu().detach().numpy().tolist())
            preds.E0.extend(E0_pred.cpu().detach().numpy().tolist())
            preds.rb.extend(rb_pred.cpu().detach().numpy().tolist())
            preds.dtemp.extend(dT_dt_pred.cpu().detach().numpy().flatten().tolist())
            preds.f.extend(f_pred.cpu().detach().numpy().tolist())
            preds.z.extend(z.cpu().detach().numpy().tolist())
            preds.noise_mus.extend(noise_mu.cpu().detach().numpy().tolist())
            preds.noise_stds.extend(noise_std.cpu().detach().numpy().tolist())

            gt.nee.extend(nee.cpu().detach().numpy().tolist())
            gt.E0.extend(k[:, 0].cpu().detach().numpy().tolist())
            gt.rb.extend(k[:, 1].cpu().detach().numpy().tolist())
            gt.f.extend(f.cpu().detach().numpy().tolist())
            gt.dtemp.extend(dtemp.cpu().detach().numpy().tolist())

        for col in preds:
            preds[col] = np.array(preds[col])
            if len(preds[col].shape) > 1 and preds[col].shape[1] == 1:
                preds[col] = preds[col].flatten()
        for col in gt:
            gt[col] = np.array(gt[col])
            if len(gt[col].shape) > 1 and gt[col].shape[1] == 1:
                gt[col] = gt[col].flatten()

        return gt, preds

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        # Apply He initialization
        nn.init.xavier_uniform_(layer.weight)
        #init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(layer.bias)


class PIAE_SDE_Reg_Sampling_Model(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_dims, decoder_dims, noise_dims=[4], activation=nn.ReLU, hard_z=False, device="mps"):
        super(PIAE_SDE_Reg_Sampling_Model, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.device = device
        self.Tref = torch.tensor(10).to(self.device)
        self.T0 = torch.tensor(46.02).to(self.device)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.noise_dims = noise_dims

        # Encoder network
        modules = self.append_linear_modules(self.input_dim, self.encoder_dims)
        modules.append(nn.Linear(self.encoder_dims[-1], self.latent_dim))
        print(modules)
        self.encoder = nn.Sequential(*modules)

       # Decoder network for NEE (u)
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims) 
        modules.append(nn.Linear(self.decoder_dims[-1], 1))
        self.nee_decoder = nn.Sequential(*modules)

        # Noise
        modules = self.append_linear_modules(self.latent_dim, self.noise_dims)
        mu_modules = modules + [nn.Linear(self.noise_dims[-1], 1)]
        logvar_modules = modules + [nn.Linear(self.noise_dims[-1], 1)]
        
        self.fc_mu = nn.Sequential(*mu_modules)
        self.fc_logvar = nn.Sequential(*logvar_modules)
        
        # Decoder network for dT/dt (f)
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims) 
        modules.append(nn.Linear(self.decoder_dims[-1], 1))
        self.temp_derivative_decoder = nn.Sequential(*modules)

        # Decoder network for E0 and rb
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims, nn.LeakyReLU(negative_slope=0.01)) 
        modules.append(nn.Linear(self.decoder_dims[-1], 2))
        modules.append(nn.LeakyReLU(negative_slope=0.01))
        self.k_decoder = nn.Sequential(*modules)

    def append_linear_modules(self, in_dim, dims, activation=None):
        modules = []
        for i, dim in enumerate(dims):
            modules.append(nn.Linear(in_dim, dim))
            modules.append(self.activation()) if not activation else modules.append(activation)
            in_dim = dim
        return modules

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, b, k, T):
        input_ = torch.cat((x, b.view(x.shape[0], 1), k), dim=1).to(self.device)
        z = self.encoder(input_)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        noise = self.reparameterize(mu, logvar)
        
        nee = self.nee_decoder(z) + noise
        dT_dt = self.temp_derivative_decoder(z)
        k_pred = self.k_decoder(z)
        f, residual = self.physics_residual(nee, k_pred, T.view((-1, 1)), dT_dt)
        
        return nee, dT_dt, k_pred, f, z, residual, noise

    def physics_residual(self, nee, k, T, dT_dt):
        self.E0 = k[:, 0].view((-1, 1))
        self.rb = k[:, 1].view((-1, 1))
        
        # Compute dNEE/dT using predicted E0 and rb
        self.exp_term = torch.exp(self.E0 * (1.0 / (self.Tref - self.T0) - 1.0 / (T - self.T0))).view((-1, 1))
        self.dNEE_dT = self.rb * (self.E0 / (T - self.T0)**2) * self.exp_term

        residual = torch.zeros_like(nee)
 
        f = self.dNEE_dT * dT_dt
        
        return f, residual
