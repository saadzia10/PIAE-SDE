import argparse
import os.path
import pickle

import pandas as pd
import json
import time
import numpy as np

from torchviz import make_dot
import torch
from torch import optim
import torch.utils.data

from model import ClimateVAE  # Adjusted import for the ClimateVAE model
import utils

def set_parser():
    parser = argparse.ArgumentParser(description='')

    # input/output setting
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--dataname-train', type=str, default='train')
    parser.add_argument('--dataname-valid', type=str, default='valid')

    # prior knowledge
    parser.add_argument('--range-E0', type=float, nargs=2, default=[50, 400])

    # model (general)
    parser.add_argument('--dim-z-phy', type=int, default=1)
    parser.add_argument('--dim-z-aux', type=int, default=1)
    parser.add_argument('--activation', type=str, default='relu') #choices=['relu','leakyrelu','elu','softplus','prelu'],
    parser.add_argument('--no-phy', action='store_true', default=False)

    # model (decoder)
    parser.add_argument('--x-lnvar', type=float, default=-8.0)
    parser.add_argument('--hidlayers-aux-dec', type=int, nargs='+', default=[16,])

    # model (encoder)
    parser.add_argument('--hidlayers-aux-enc', type=int, nargs='+', default=[16,])
    parser.add_argument('--hidlayers-unmixer', type=int, nargs='+', default=[16,])
    parser.add_argument('--hidlayers-phy', type=int, nargs='+', default=[16])
    parser.add_argument('--arch-feat', type=str, default='mlp')
    parser.add_argument('--num-units-feat', type=int, default=16)
    parser.add_argument('--hidlayers-feat', type=int, nargs='+', default=[16,])
    parser.add_argument('--num-rnns-feat', type=int, default=1)

    # optimization (base)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--adam-eps', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--balance-kld', type=float, default=1.0)
    parser.add_argument('--balance-unmix', type=float, default=0.0)
    parser.add_argument('--balance-dataug', type=float, default=0.0)
    parser.add_argument('--balance-lact-dec', type=float, default=0.0)
    parser.add_argument('--balance-lact-enc', type=float, default=0.0)

    # others
    parser.add_argument('--train-size', type=int, default=-1)
    parser.add_argument('--save-interval', type=int, default=999999999)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234567890)

    return parser


def loss_function(args, data, z_phy_stat, z_aux_stat, x_mean):
    n = data.shape[0]
    device = data.device

    # Reconstruction error (Mean Squared Error)
    recerr_sq = torch.sum((x_mean - data).pow(2), dim=1).mean()

    # Priors for the latent variables
    prior_z_phy_stat, prior_z_aux_stat = model.priors(n, device)

    # KL divergence for auxiliary latent variables (rb_n)
    KL_z_aux = utils.kldiv_normal_normal(z_aux_stat['mean'], z_aux_stat['lnvar'],
                                         prior_z_aux_stat['mean'],
                                         prior_z_aux_stat['lnvar']) if args.dim_z_aux > 0 else torch.zeros(1,
                                                                                                           device=device)

    # KL divergence for physics latent variables (E_0)
    KL_z_phy = utils.kldiv_normal_normal(z_phy_stat['mean'], z_phy_stat['lnvar'],
                                         prior_z_phy_stat['mean'],
                                         prior_z_phy_stat['lnvar']) if not args.no_phy else torch.zeros(1,
                                                                                                        device=device)

    # Total KL divergence
    kldiv = (KL_z_aux + KL_z_phy).mean()

    return recerr_sq, kldiv


def train(epoch, args, device, loader, model, optimizer):
    model.train()
    logs = {'recerr_sq':.0, 'kldiv':.0, 'unmix':.0, 'dataug':.0, 'lact_dec':.0}

    rb0 = {'mean': [], 'lnvar': []}
    e0 = {'mean': [], 'lnvar': []}

    for batch_idx, (data, T_air, NEE) in enumerate(loader):
        data = data.to(device)
        T_air = T_air.to(device).reshape((-1, 1))
        batch_size = len(data)
        optimizer.zero_grad()

        NEE = NEE.reshape((-1, 1))

        # yhat = model(data, T_air)
        # make_dot(yhat[2], params=dict(model.named_parameters())).render("vae.png", format="png")

        # inference & reconstruction on original data
        z_phy_stat, z_aux_stat, unmixed = model.encode(data)
        rb0['mean'].append(z_aux_stat['mean'].detach().mean())
        rb0['lnvar'].append(z_aux_stat['lnvar'].detach().mean())

        e0['mean'].append(z_phy_stat['mean'].detach().mean())
        e0['lnvar'].append(z_phy_stat['lnvar'].detach().mean())

        z_phy, z_aux = model.draw(z_phy_stat, z_aux_stat, hard_z=False)
        NEE_pred, lnvar = model.decode(z_phy, z_aux, T_air)
        # ELBO
        recerr_sq, kldiv = loss_function(args, NEE, z_phy_stat, z_aux_stat, NEE_pred)

        # unmixing regularization (R_{DA,1})
        reg_unmix = torch.sum((unmixed - z_phy.detach()).pow(2), dim=1).mean()

        # least action principle (R_ppc)
        # least action principle (R_ppc)
        # Calculate differences between the reconstructions
        NEE_PA, _ = model.decode(z_phy, torch.zeros_like(z_aux), T_air)
        NEE_PB, _ = model.decode(torch.zeros_like(z_phy), z_aux, T_air)

        dif_PA_P = torch.sum((NEE_PA - NEE).pow(2), dim=1).mean()
        dif_PB_P = torch.sum((NEE_PB - NEE).pow(2), dim=1).mean()
        dif_PAB_PA = torch.sum((NEE - NEE_PA).pow(2), dim=1).mean()
        dif_PAB_PB = torch.sum((NEE - NEE_PB).pow(2), dim=1).mean()
        reg_lact_dec = 0.25 * dif_PA_P + 0.25 * dif_PB_P + 0.25 * dif_PAB_PA + 0.25 * dif_PAB_PB

        # loss function
        kldiv_balanced = (args.balance_kld + args.balance_lact_enc) * kldiv
        loss = recerr_sq + kldiv_balanced \
            + args.balance_unmix * reg_unmix + args.balance_lact_dec * reg_lact_dec

        # update model parameters
        loss.backward()
        # if args.grad_clip > 0.0:
        #     torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)

        optimizer.step()

        # log
        logs['recerr_sq'] += recerr_sq.detach() * batch_size
        logs['kldiv'] += kldiv.detach() * batch_size
        logs['unmix'] += reg_unmix.detach() * batch_size
        # logs['dataug'] += reg_dataug.detach() * batch_size
        logs['lact_dec'] += reg_lact_dec.detach() * batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Training (rec. err.)^2: {:.6f}  kldiv: {:.6f}  unmix: {:.4f}  dataug: {:.4f}  lact_dec: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv'], logs['unmix'], logs['dataug'], logs['lact_dec']))
    return logs, e0, rb0


def valid(epoch, args, device, loader, model):
    model.eval()
    logs = {'recerr_sq': .0, 'kldiv': .0}
    with torch.no_grad():
        for i, (data, T_air, NEE) in enumerate(loader):
            data = data.to(device)
            T_air = T_air.to(device).reshape((-1, 1))
            batch_size = len(data)

            NEE = NEE.reshape((-1, 1))

            # inference & reconstruction on original data
            z_phy_stat, z_aux_stat, unmixed = model.encode(data)
            z_phy, z_aux = model.draw(z_phy_stat, z_aux_stat, hard_z=False)
            NEE_pred, lnvar = model.decode(z_phy, z_aux, T_air)

            # ELBO
            recerr_sq, kldiv = loss_function(args, NEE, z_phy_stat, z_aux_stat, NEE_pred)

            # log
            logs['recerr_sq'] += recerr_sq.detach() * batch_size
            logs['kldiv'] += kldiv.detach() * batch_size

    for key in logs:
        logs[key] /= len(loader.dataset)
    print('====> Epoch: {}  Validation (rec. err.)^2: {:.4f}  kldiv: {:.4f}'.format(
        epoch, logs['recerr_sq'], logs['kldiv']))
    return logs


def create_data_loader(file_path, batch_size=32, shuffle=True, get_dim=False, kwargs=None):
    # Load the CSV file
    df = pd.read_csv(file_path)

    NEE = df['NEE']
    Tair = df["Ta"]
    del df['NEE']
    del df["Ta"]
    # Convert data to tensors
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    ta_col = torch.tensor(Tair.values, dtype=torch.float32)
    nee_col = torch.tensor(NEE.values, dtype=torch.float32)

    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(data_tensor, ta_col, nee_col)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              **kwargs) if kwargs else torch.utils.data.DataLoader(dataset,
                                                                                                   batch_size=batch_size,
                                                                                                   shuffle=shuffle)
    if not get_dim:
        return data_loader
    return data_loader, df.shape[1]


if __name__ == '__main__':

    parser = set_parser()
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")


    # set random seed
    torch.manual_seed(args.seed)

    # load training/validation data
    train_file_path = args.datadir + '/train_night_subset_train.csv'
    val_file_path = args.datadir + '/train_night_subset_train.csv'


    # set data loaders
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

    # Create DataLoaders
    loader_train, dims = create_data_loader(train_file_path, batch_size=args.batch_size, shuffle=True, get_dim=True, kwargs= kwargs)
    loader_valid = create_data_loader(val_file_path, batch_size=args.batch_size, shuffle=False, kwargs= kwargs)

    args.dim_t = dims
    # set model
    model = ClimateVAE(vars(args)).to(device)

    # set optimizer
    kwargs = {'lr': args.learning_rate, 'weight_decay': args.weight_decay, 'eps': args.adam_eps}
    optimizer = optim.Adam(model.parameters(), **kwargs)

    print('start training with device', device)
    print(vars(args))
    print()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # save args
    with open('{}/args.json'.format(args.outdir), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # create log files
    with open('{}/log.txt'.format(args.outdir), 'w') as f:
        print('# epoch recerr_sq kldiv unmix dataug lact_dec valid_recerr_sq valid_kldiv duration', file=f)


    # main iteration
    info = {'bestvalid_epoch':0, 'bestvalid_recerr':1e10}
    dur_total = .0

    rb0s = {'mean': [], 'lnvar': []}
    e0s = {'mean': [], 'lnvar': []}

    for epoch in range(1, args.epochs + 1):
        # training
        start_time = time.time()
        logs_train, e0, rb0 = train(epoch, args, device, loader_train, model, optimizer)

        rb0s['mean'].append(np.mean(rb0['mean']))
        rb0s['lnvar'].append(np.mean(rb0['lnvar']))

        e0s['mean'].append(np.mean(e0['mean']))
        e0s['lnvar'].append(np.mean(e0['lnvar']))

        dur_total += time.time() - start_time

        # validation
        logs_valid = valid(epoch, args, device, loader_valid, model)

        # save loss information
        with open('{}/log.txt'.format(args.outdir), 'a') as f:
            print('{} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e} {:.7e}'.format(epoch,
                logs_train['recerr_sq'], logs_train['kldiv'], logs_train['unmix'], logs_train['dataug'], logs_train['lact_dec'],
                logs_valid['recerr_sq'], logs_valid['kldiv'], dur_total), file=f)

        # save model if best validation loss is achieved
        if logs_valid['recerr_sq'] < info['bestvalid_recerr']:
            info['bestvalid_epoch'] = epoch
            info['bestvalid_recerr'] = logs_valid['recerr_sq']
            torch.save(model.state_dict(), '{}/model.pt'.format(args.outdir))
            print('best model saved')

        # save model at interval
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), '{}/model_e{}.pt'.format(args.outdir, epoch))

        print()

    print()
    print('end training')

    with open("z_phy.pickle", "wb") as fp:
        pickle.dump(e0s, fp)
    with open("z_aux.pickle", "wb") as fp:
        pickle.dump(rb0s, fp)
