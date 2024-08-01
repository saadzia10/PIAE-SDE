
import sys; sys.path.append('../')
from mlp import MLP
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        layers = []
        input_dim = config['dim_t']
        for output_dim in config['hidlayers_feat']:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Encoders(nn.Module):
    def __init__(self, config: dict):
        super(Encoders, self).__init__()

        self.dim_t = config['dim_t']
        self.dim_z_phy = config['dim_z_phy']  # 1 for E_0
        self.dim_z_aux = config['dim_z_aux']  # 1 for rb_n
        self.activation = config['activation']

        # Feature extractor
        self.func_feat = FeatureExtractor(config)

        # Encoders for z_phy (E_0)
        # Unmixing network
        self.func_unmixer = MLP(
            config['hidlayers_unmixer'] + [self.dim_t, ], self.activation)

        self.func_z_phy_mean = MLP([self.dim_t, ] + config['hidlayers_phy'] + [self.dim_z_phy, ], self.activation)
        self.func_z_phy_lnvar = MLP([self.dim_t, ] + config['hidlayers_phy'] + [self.dim_z_phy, ], self.activation)

        # Encoders for z_aux (rb_n)
        if self.dim_z_aux > 0:
            self.func_z_aux_mean = MLP([self.dim_t, ] + config['hidlayers_aux_enc'] + [self.dim_z_aux, ],
                                       self.activation)
            self.func_z_aux_lnvar = MLP([self.dim_t, ] + config['hidlayers_aux_enc'] + [self.dim_z_aux, ],
                                        self.activation)

    def forward(self, x):
        features = self.func_feat(x)
        unmixed = self.func_unmixer(features)

        z_phy_mean = self.func_z_phy_mean(unmixed)
        z_phy_lnvar = self.func_z_phy_lnvar(unmixed)

        if self.dim_z_aux > 0:
            z_aux_mean = self.func_z_aux_mean(unmixed)
            z_aux_lnvar = self.func_z_aux_lnvar(unmixed)
        else:
            z_aux_mean = torch.zeros(features.size(0), 0, device=features.device)
            z_aux_lnvar = torch.zeros(features.size(0), 0, device=features.device)

        return z_phy_mean, z_phy_lnvar, z_aux_mean, z_aux_lnvar, unmixed


class Decoders(nn.Module):
    def __init__(self, config: dict):
        super(Decoders, self).__init__()

        self.T_ref = 15.0
        self.T_0 = -46.02
        self.dim_z_phy = config['dim_z_phy']  # Assuming dim_z_phy = 1 for E_0
        self.dim_z_aux = config['dim_z_aux']  # Assuming dim_z_aux = 1 for rb_n
        self.dim_t = config['dim_t']
        self.activation = config['activation']

        # Physics function for E_0
        self.func_phy = nn.Linear(self.dim_z_phy, 1)

        # Auxiliary function for rb_n
        if self.dim_z_aux > 0:
            self.func_aux = MLP([self.dim_z_aux + 1, ] + config['hidlayers_aux_dec'] + [1, ],
                                self.activation)

        # Register buffer for log variance
        self.register_buffer('param_x_lnvar', torch.ones(1) * config['x_lnvar'])

    def forward(self, z_phy, z_aux, T_air):
        # Physics-based model output
        E_0 = z_phy
        term = (1 / (self.T_ref - self.T_0) - 1 / (T_air - self.T_0))
        NEE_phy = torch.exp(E_0 * term) * z_aux

        NEE = NEE_phy#.unsqueeze(1)

        # # Auxiliary model output
        # if self.dim_z_aux > 0:
        #     aux_input = torch.cat([z_aux, T_air.view(-1, 1)], dim=1)
        #     rb_n = self.func_aux(aux_input)
        #     NEE = rb_n * NEE_phy.unsqueeze(1)
        # else:
        #     NEE = NEE_phy.unsqueeze(1)

        return NEE, self.param_x_lnvar


class ClimateVAE(nn.Module):
    def __init__(self, config):
        super(ClimateVAE, self).__init__()
        self.enc = Encoders(config)
        self.dec = Decoders(config)
        self.range_E0 = config['range_E0']

    def generate_phyonly(self, z_phy, T_air):
        # Physics-based model output
        E_0 = z_phy[:, 0]
        term = (1 / (self.dec.T_ref - self.dec.T_0) - 1 / (T_air - self.dec.T_0))
        NEE_phy = E_0 * term
        return NEE_phy.unsqueeze(1)

    def encode(self, x):
        z_phy_mean, z_phy_lnvar, z_aux_mean, z_aux_lnvar, unmixed = self.enc(x)
        z_phy_stat = {'mean': z_phy_mean, 'lnvar': z_phy_lnvar}
        z_aux_stat = {'mean': z_aux_mean, 'lnvar': z_aux_lnvar}

        return z_phy_stat, z_aux_stat, unmixed

    def priors(self, n, device):
        # Prior for E_0 (assumed to be normally distributed)
        prior_z_phy_mean = torch.ones(n,1,device=device) * 0.5 * (self.range_E0[0] + self.range_E0[1])
        prior_z_phy_lnvar = torch.zeros(n, 1, device=device)  # ln(1) = 0 for unit variance

        # Prior for rb_n (assumed to be normally distributed)
        prior_z_aux_mean = torch.ones(n,1,device=device)
        prior_z_aux_lnvar = torch.full((n, 1), 0.0, device=device)  # ln(1) = 0 for unit variance

        prior_z_phy_stat = {'mean': prior_z_phy_mean, 'lnvar': prior_z_phy_lnvar}
        prior_z_aux_stat = {'mean': prior_z_aux_mean, 'lnvar': prior_z_aux_lnvar}

        return prior_z_phy_stat, prior_z_aux_stat

    def reparameterize(self, mean, lnvar):
        std = torch.exp(0.5 * lnvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def draw(self, z_phy_stat, z_aux_stat, hard_z=False):
        z_phy_mean = z_phy_stat['mean']
        z_phy_lnvar = z_phy_stat['lnvar']
        z_aux_mean = z_aux_stat['mean']
        z_aux_lnvar = z_aux_stat['lnvar']

        if hard_z:
            z_phy = z_phy_mean
            z_aux = z_aux_mean
        else:
            z_phy = self.reparameterize(z_phy_mean, z_phy_lnvar)
            z_aux = self.reparameterize(z_aux_mean, z_aux_lnvar)

        z_phy = torch.clamp(z_phy, 50, 500)
        z_aux = torch.abs(z_aux)

        return z_phy, z_aux

    def decode(self, z_phy, z_aux, T_air):
        return self.dec(z_phy, z_aux, T_air)

    def forward(self, x, T_air):
        z_phy_stat, z_aux_stat, unmixed = self.encode(x)
        z_phy = self.reparameterize(z_phy_stat['mean'], z_phy_stat['lnvar'])
        z_aux = self.reparameterize(z_aux_stat['mean'], z_aux_stat['lnvar'])
        z_phy = torch.clamp(z_phy, 50, 500)
        z_aux = torch.abs(z_aux)
        NEE, lnvar = self.decode(z_phy, z_aux, T_air)
        return z_phy_stat, z_aux_stat, NEE, lnvar
