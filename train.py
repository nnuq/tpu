# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import torch
import torch.nn as nn
import numpy as np
import os
import itertools
import pathlib
import cfg
import datasets
from models import Encoder, Discriminator, Generator
from functions import train, validate, LinearLrDecay, load_params, copy_params
from utils import set_log_dir, save_checkpoint, create_logger
from download import download_stat_cifar10_test
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

def main(index, args):
    device = xm.xla_device()

    gen_net = Generator(args).to(device)
    dis_net = Discriminator(args).to(device)
    enc_net = Encoder(args).to(device)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    enc_net.apply(weights_init)
        
    ae_recon_optimizer = torch.optim.Adam(itertools.chain(enc_net.parameters(), gen_net.parameters()),
                                     args.ae_recon_lr, (args.beta1, args.beta2))
    ae_reg_optimizer = torch.optim.Adam(itertools.chain(enc_net.parameters(), gen_net.parameters()),
                                     args.ae_reg_lr, (args.beta1, args.beta2))                                 
    dis_optimizer = torch.optim.Adam(dis_net.parameters(),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_optimizer = torch.optim.Adam(gen_net.parameters(),
                                     args.g_lr, (args.beta1, args.beta2))                                 
    
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    valid_loader = dataset.valid
    para_loader = pl.ParallelLoader(train_loader, [device])

    fid_stat = str(pathlib.Path(__file__).parent.absolute()) + '/fid_stat/fid_stat_cifar10_test.npz'
    if not os.path.exists(fid_stat):
         download_stat_cifar10_test()

    is_best = True
    args.num_epochs = np.ceil(args.num_iter / len(train_loader))
    
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0, args.num_iter/2, args.num_iter)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0, args.num_iter/2, args.num_iter)
    ae_recon_scheduler = LinearLrDecay(ae_recon_optimizer, args.ae_recon_lr, 0, args.num_iter/2, args.num_iter)
    ae_reg_scheduler = LinearLrDecay(ae_reg_optimizer, args.ae_reg_lr, 0, args.num_iter/2, args.num_iter)

    # initial
    start_epoch = 0
    best_fid = 1e4
    
    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        enc_net.load_state_dict(checkpoint['enc_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        ae_recon_optimizer.load_state_dict(checkpoint['ae_recon_optimizer'])
        ae_reg_optimizer.load_state_dict(checkpoint['ae_reg_optimizer'])
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        logs_dir = str(pathlib.Path(__file__).parent.parent) + '/logs'
        args.path_helper = set_log_dir(logs_dir, args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
    
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    
    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.num_epochs)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler, ae_recon_scheduler, ae_reg_scheduler) 
        train(device, args, gen_net, dis_net, enc_net, gen_optimizer, dis_optimizer, ae_recon_optimizer, ae_reg_optimizer, 
                                                    para_loader, epoch, writer_dict, lr_schedulers)             
        if epoch and epoch % args.val_freq == 0 or epoch == args.num_epochs - 1:
            fid_score = validate(args, fid_stat, gen_net, writer_dict, valid_loader)
            logger.info(f'FID score: {fid_score} || @ epoch {epoch}.')
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False
        
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'enc_state_dict': enc_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'ae_recon_optimizer': ae_recon_optimizer.state_dict(),
            'ae_reg_optimizer': ae_reg_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper
        }, is_best, args.path_helper['ckpt_path'])
        
    
if __name__ == '__main__':
    arg = cfg.parse_args()

    xmp.spawn(main, args=(arg,), nprocs=8)