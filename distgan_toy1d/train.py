import torch
import torch.nn as nn
import numpy as np
import os
import itertools
import pathlib
from functions import train_wgangp, train_distgan, Visualization
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import argparse

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1, 4)
        self.l2 = nn.Linear(4, 1)

    def forward(self, z):
        return self.l2(nn.ReLU()(self.l1(z)))


def main(args):
 
    gen_net = Model()
    dis_net = Model()
    enc_net = Model()
    
    wgangp_filename = '{}/wgangp.gif'.format(args.save_path)
    distgan_filename = '{}/distgan.gif'.format(args.save_path)
    
    vis_wgangp = Visualization(file_name=wgangp_filename, model_name='WGAN-GP')
    vis_distgan = Visualization(file_name=distgan_filename, model_name='Dist-GAN')
    
    ae_recon_optimizer = torch.optim.Adam(itertools.chain(enc_net.parameters(), gen_net.parameters()),
                                     args.ae_recon_lr, (args.beta1, args.beta2))
    ae_reg_optimizer = torch.optim.Adam(itertools.chain(enc_net.parameters(), gen_net.parameters()),
                                     args.ae_reg_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(dis_net.parameters(),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_optimizer = torch.optim.Adam(gen_net.parameters(),
                                     args.g_lr, (args.beta1, args.beta2))                                 
    
    train_distgan(args, gen_net, dis_net, enc_net, gen_optimizer, dis_optimizer, \
                                         ae_recon_optimizer, ae_reg_optimizer, vis_distgan)  
    train_wgangp(args, gen_net, dis_net, gen_optimizer, dis_optimizer, vis_wgangp)  

    
             
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-train_bs',
        '--train_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.5,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of second order momentum of gradient')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.001,
        help='adam: generator learning rate')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.001,
        help='adam: discriminator learning rate')
    parser.add_argument(
        '--ae_recon_lr',
        type=float,
        default=0.001,
        help='adam: autoencoder (reconstruction loss) learning rate')
    parser.add_argument(
        '--ae_reg_lr',
        type=float,
        default=0.001,
        help='adam: autoencoder (regularization loss) learning rate')  
    parser.add_argument(
        '--print_freq',
        type=int,
        default=100,
        help='interval between each verbose')
    parser.add_argument(
        '--inter_gif',
        type=int,
        default=50,
        help='interval iterations between each frame in gif')
    parser.add_argument(
        '--save_path',
        type=str,
        default='{}'.format(str(pathlib.Path(__file__).parent)))        
    args = parser.parse_args()
    main(args)