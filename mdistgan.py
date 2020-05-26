import os
from argparse import ArgumentParser
from collections import OrderedDict
from torch.autograd import Variable
from torch import autograd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models import Encoder, Discriminator, Generator
import itertools
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision.utils import save_image


def calculate_gradient_penalty(dis_net: nn.Module, real_samples, fake_samples):
        
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = dis_net(interpolates, 0)
    fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return grad_penalty 


class GAN(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gen_net = Generator(self.hparams).cuda()
        self.dis_net = Discriminator(self.hparams).cuda()
        self.enc_net = Encoder(self.hparams).cuda()
        
        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None
    
    def forward(self, z):
        return self.gen_net(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (imgs.shape[0], self.hparams.latent_dim)))
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # train generator
        if optimizer_idx == 0:
            # sample noise
            
            z_enc = self.enc_net(real_imgs)
            
            with torch.no_grad():
                feature_real = self.dis_net(real_imgs ,'feat')
            
            reconstructed_imgs = self.gen_net(z_enc)
            feature_recon = self.dis_net(reconstructed_imgs,'feat')
            ae_loss    = torch.mean((feature_real - feature_recon) ** 2)
        
            tqdm_dict = {'ae_loss': ae_loss}
            output = OrderedDict({
                'loss': ae_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            z_enc = self.enc_net(real_imgs)
            
            with torch.no_grad():
                feature_real = self.dis_net(real_imgs ,'feat')
            
            reconstructed_imgs = self.gen_net(z_enc)
            feature_recon = self.dis_net(reconstructed_imgs,'feat')
            generate_imgs = self.gen_net(z)
            feature_fake = self.dis_net(generate_imgs,'feat')
            
            lambda_w  = np.sqrt(self.hparams.latent_dim * 1.0/feature_recon.size()[1])
            md_x       = torch.mean(feature_recon - feature_fake)
            md_z       = torch.mean(z_enc - z) * lambda_w
            ae_reg     = (md_x - md_z) ** 2

            tqdm_dict = {'ae_reg': ae_reg}
            output = OrderedDict({
                'loss': ae_reg,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        
        # train discriminator
        if optimizer_idx == 2:
            # Measure discriminator's ability to classify real from generated samples
            
            with torch.no_grad():
                z_enc = self.enc_net(real_imgs)
                generate_imgs = self.gen_net(z)
                reconstructed_imgs = self.gen_net(z_enc)
            
            #images_mix, larg_mix, _ = argument_image_rotation_and_fake_mix(real_imgs, generate_imgs)    
            
            disc_real_logit = self.dis_net(real_imgs, 'out')
            disc_fake_logit = self.dis_net(generate_imgs, 'out')
            disc_recon_logit = self.dis_net(reconstructed_imgs, 'out')
            #mixe_cls = self.dis_net(images_mix, 'cls')
            
            #d_acc = torch.sum(binary_cross_entropy_with_logits(mixe_cls, larg_mix))
            
            grad_penalty = calculate_gradient_penalty(self.dis_net, real_imgs, generate_imgs)
            
            l_1 = torch.mean(nn.ReLU()(1 - disc_real_logit))
            l_2 = torch.mean(nn.ReLU()(1 - disc_recon_logit))
            l_3 = torch.mean(nn.ReLU()(1 + disc_fake_logit))    
            
            t = float(self.global_step)/self.hparams.max_iter
            
            mu = max(min((t*0.1-0.05)*2, 0.05),0.0)
            w_real  = 0.95 + mu
            w_recon = 0.05 - mu
            
            d_loss = w_real * l_1 + w_recon * l_2 + l_3 + grad_penalty# + 0.5 * d_acc

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        
        # train discriminator
        if optimizer_idx == 3:
            # Measure discriminator's ability to classify real from generated samples
            generate_imgs = self.gen_net(z)

            #Xarg, larg, ridx = argument_image_rotation_and_fake(real_imgs)
            #Xarg_f, larg_f, _ = argument_image_rotation_and_fake(generate_imgs, ridx=ridx)

            with torch.no_grad():
                disc_real_logit = self.dis_net(real_imgs, 'out')
                #real_cls = self.dis_net(Xarg, 2)
                #g_real_acc = torch.sum(binary_cross_entropy_with_logits(real_cls, larg))
            
            disc_fake_logit = self.dis_net(generate_imgs, 'out')
            #fake_cls = self.dis_net(Xarg_f, 2)
            #g_fake_acc = torch.sum(binary_cross_entropy_with_logits(fake_cls, larg_f))       
            
            g_loss  = torch.abs(torch.mean(nn.Sigmoid()(disc_real_logit)) - torch.mean(nn.Sigmoid()(disc_fake_logit)))
            # + 0.1 * torch.abs(g_fake_acc - g_real_acc)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output        

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        ae_recon_opt = torch.optim.Adam(itertools.chain(self.enc_net.parameters(), self.gen_net.parameters()),
                                        lr, (b1, b2))
        ae_reg_opt = torch.optim.Adam(itertools.chain(self.enc_net.parameters(), self.gen_net.parameters()),
                                        lr, (b1, b2))                                 
        dis_opt = torch.optim.Adam(self.dis_net.parameters(),
                                        lr, (b1, b2))
        gen_opt = torch.optim.Adam(self.gen_net.parameters(),
                                        lr, (b1, b2)) 
        return [ae_recon_opt, ae_reg_opt, dis_opt, gen_opt], []

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (real_imgs.shape[0], self.hparams.latent_dim)))
        gen_imgs = self.gen_net(z)
        z_enc = self.enc_net(real_imgs.cuda())
        recon_imgs = self.gen_net(z_enc)
        
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(os.getcwd() + '/fid_buffer_gen', f'iter{batch_idx}_b{img_idx}.png')
            save_image(img, file_name, normalize=True)
        
        for img_idx, img in enumerate(recon_imgs):
            file_name = os.path.join(os.getcwd() + '/fid_buffer_recon', f'iter{batch_idx}_b{img_idx}.png')
            save_image(img, file_name, normalize=True)
         
        for img_idx, img in enumerate(real_imgs):
            file_name = os.path.join(os.getcwd() + '/fid_buffer_real', f'iter{batch_idx}_b{img_idx}.png')
            save_image(img, file_name, normalize=True) 
        
    def validation_epoch_end(self, outputs):     
        fid_score_gen = calculate_fid_given_paths([os.getcwd() + '/fid_buffer_gen', os.getcwd() + '/fid_buffer_real'], 200) 
        fid_score_ae = calculate_fid_given_paths([os.getcwd() + '/fid_buffer_recon', os.getcwd() + '/fid_buffer_real'], 200)
        
        tqdm_dict = {'fid_score_gen': fid_score_gen.item(),
                     'fid_score_ae': fid_score_ae.item()}
        
        results =  {
                    'progress_bar': tqdm_dict,
                    'log': {'fid_score_gen': fid_score_gen.item(),
                    'fid_score_ae': fid_score_ae.item(), 'step': self.current_epoch}
                    }

        os.system('rm -r {}'.format(os.getcwd() + '/fid_buffer_gen'))
        os.system('rm -r {}'.format(os.getcwd() + '/fid_buffer_recon'))
        os.system('rm -r {}'.format(os.getcwd() + '/fid_buffer_real'))

        return results
        
    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.train_batch_size)

         
    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.eval_batch_size)    


import pytorch_lightning as pl

class MyPrintingCallback(pl.Callback):
    
    def on_init_start(self, trainer):
        fid_buffer_dir_gen = os.path.join(os.getcwd() + '/fid_buffer_gen')
        os.makedirs(fid_buffer_dir_gen, exist_ok=True)
        
        fid_buffer_dir_recon = os.path.join(os.getcwd() + '/fid_buffer_recon')
        os.makedirs(fid_buffer_dir_recon, exist_ok=True)
        
        fid_buffer_dir_real = os.path.join(os.getcwd() + '/fid_buffer_real')
        os.makedirs(fid_buffer_dir_real, exist_ok=True)

    def on_validation_batch_start(self, trainer, pl_module):
        fid_buffer_dir_gen = os.path.join(os.getcwd() + '/fid_buffer_gen')
        os.makedirs(fid_buffer_dir_gen, exist_ok=True)
        
        fid_buffer_dir_recon = os.path.join(os.getcwd() + '/fid_buffer_recon')
        os.makedirs(fid_buffer_dir_recon, exist_ok=True)
        
        fid_buffer_dir_real = os.path.join(os.getcwd() + '/fid_buffer_real')
        os.makedirs(fid_buffer_dir_real, exist_ok=True)    

   

def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(hparams)
    
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer(callbacks=[MyPrintingCallback()], check_val_every_n_epoch=1)
    
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--max_iter',
        type=int,
        default=300000,
        help='number of epochs of training')
    parser.add_argument(
        '-train_bs',
        '--train_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='adam: learning rate')    
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN")    
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=10,
        help='interval between each validation')    
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf_dim', type=int, default=128,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=128,
                        help='The base channel num of disc')
    parser.add_argument('--ef_dim', type=int, default=128,
                        help='The base channel num of enc')                    
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=12345)
    hparams = parser.parse_args()

    main(hparams)