import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import Variable
from torch import autograd
from tqdm import tqdm
from matplotlib.animation import ImageMagickWriter


class Dataset(object):
    
    def __init__(self, mu=4, sigma=1.0, seed=0):
        """Initialize the 1D Gaussian to generate.
        
        :param mu: Mean of the Gaussian.
        :param sigma: Standard deviation of the Gaussian.
        :param seed: Random seed to create a reproducible sequence.
        """
        self.seed = seed
        self.mu = mu
        self.sigma = sigma
        np.random.seed(seed)
        
    def next_batch(self, batch_size):
        """Generate the next batch of toy input data.
         
        :param batch_size: Sample size.
        :return: Batch of toy data, generated from a 1D Gaussian.
        """
        return torch.FloatTensor(np.random.normal(self.mu, self.sigma, (batch_size, 1)))


class Visualization(object):
    """Helper class to visualize the progress of the GAN training procedure.
    """
    
    def __init__(self, file_name, model_name, fps=15):
        """Initialize the helper class.
        
        :param fps: The number of frames per second when saving the gif animation.
        """
        self.fps = fps
        self.figure, (self.ax2) = plt.subplots(1, 1, figsize=(5, 5))
        self.figure.suptitle("{}".format(model_name))
        sns.set(color_codes=True, style='white', palette='colorblind')
        sns.despine(self.figure)
        plt.show(block=False)
        self.real_data = Dataset()
        self.step = 0
        self.writer = ImageMagickWriter(fps=self.fps)
        self.writer.setup(self.figure, file_name, dpi=100)
       
    def plot_progress(self, gen_net):
        """Plot the progress of the training procedure. This can be called back from the GAN fit method.
        
        :param gan: The GAN we are fitting.
        :param session: The current session of the GAN.
        :param data: The data object from which we are sampling the input data.
        """
                
        real = self.real_data.next_batch(batch_size=10000)
        r1,r2 = self.ax2.get_xlim()
        x = np.linspace(r1, r2, 10000)[:, np.newaxis]
        g = gen_net(torch.FloatTensor(np.random.randn(10000, 1)))
        
        self.ax2.clear()
        self.ax2.set_ylim([0, 1])
        self.ax2.set_xlim([0, 8])
        sns.kdeplot(real.numpy().flatten(), shade=True, ax=self.ax2, label='Real data')
        sns.kdeplot(g.detach().numpy().flatten(), shade=True, ax=self.ax2, label='Generated data')
        self.ax2.set_title('Distributions')
        self.ax2.set_title('{} iterations'.format(self.step * 50))
        self.ax2.set_xlabel('Input domain')
        self.ax2.set_ylabel('Probability density')
        self.ax2.legend(loc='upper left', frameon=True)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.writer.grab_frame()        
        self.step += 1 
        


def calculate_gradient_penalty(dis_net: nn.Module, real_samples, fake_samples):
        
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = dis_net(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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


def train_distgan(args, gen_net: nn.Module, dis_net: nn.Module, enc_net: nn.Module, gen_optimizer, dis_optimizer, 
                                                        ae_recon_optimizer, ae_reg_optimizer, vis, schedulers=None):
    
    for iter_idx in tqdm(range(10000)):
        batch_size = args.train_batch_size
        # -----------------------------------------
        #  Train Autoencoder (reconstruction loss)
        # -----------------------------------------
        real_imgs = torch.FloatTensor(np.random.normal(4, 1, (batch_size, 1)))
        
        gen_net.zero_grad()
        enc_net.zero_grad()
        
        z_enc = enc_net(real_imgs)
        reconstructed_imgs = gen_net(z_enc)
        
        ae_recon_loss = torch.mean((real_imgs - reconstructed_imgs) ** 2)
        ae_recon_loss.backward(retain_graph=True)
        ae_recon_optimizer.step()
        
        # -----------------------------------------
        #  Train Autoencoder (regularization loss)
        # -----------------------------------------
        real_imgs = torch.FloatTensor(np.random.normal(4, 1, (batch_size, 1)))
        gen_net.zero_grad()
        enc_net.zero_grad()
        
        z_enc = enc_net(real_imgs)
        z = torch.FloatTensor(np.random.randn(batch_size, 1))                           
        generate_imgs = gen_net(z)
                
        md_x        = torch.mean(real_imgs - generate_imgs)
        md_z        = torch.mean(z_enc - z)
        ae_reg_loss = (md_x - md_z) ** 2
        ae_reg_loss.backward(retain_graph=True)
        ae_reg_optimizer.step()
        
        # -----------------------------------------
        #  Train Discriminator
        # -----------------------------------------
        dis_net.zero_grad()
        real_imgs = torch.FloatTensor(np.random.normal(4, 1, (batch_size, 1)))
        z = torch.FloatTensor(np.random.randn(batch_size, 1))                           
        
        with torch.no_grad():
            z_enc = enc_net(real_imgs)
            generate_imgs = gen_net(z)
            reconstructed_imgs = gen_net(z_enc)
        
        disc_real_logit = dis_net(real_imgs)
        disc_fake_logit = dis_net(generate_imgs)
        disc_recon_logit = dis_net(reconstructed_imgs)
                
        grad_penalty = calculate_gradient_penalty(dis_net, real_imgs, generate_imgs)
        
        l_1 = torch.mean(nn.ReLU()(1 - disc_real_logit))
        l_2 = torch.mean(nn.ReLU()(1 - disc_recon_logit))
        l_3 = torch.mean(nn.ReLU()(1 + disc_fake_logit))    
        
        w_real  = 0.95 
        w_recon = 0.05
        
        d_loss = w_real * l_1 + w_recon * l_2 + l_3 + 0.1 * grad_penalty
        d_loss.backward(retain_graph=True)
        dis_optimizer.step()
        
        # -----------------------------------------
        #  Train Generator
        # -----------------------------------------
        gen_net.zero_grad()
        
        z = torch.FloatTensor(np.random.randn(batch_size, 1))                                
        generate_imgs = gen_net(z)
        
        with torch.no_grad():
            disc_real_logit = dis_net(real_imgs)
            
        disc_fake_logit = dis_net(generate_imgs)
        
        g_loss = torch.abs(torch.mean(nn.Sigmoid()(disc_real_logit)) - torch.mean(nn.Sigmoid()(disc_fake_logit)))
        g_loss.backward()
        gen_optimizer.step()
        
        # verbose
        if iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Recon loss: %f] [Reg loss: %f] [D loss: %f] [G loss: %f]" %
                (ae_recon_loss.item(), ae_reg_loss.item(), d_loss.item(), g_loss.item()))
        if iter_idx % args.inter_gif == 0:
            vis.plot_progress(gen_net)


def train_wgangp(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, vis, schedulers=None):
    
    for iter_idx in tqdm(range(10000)):
        batch_size = args.train_batch_size
        # -----------------------------------------
        #  Train Discriminator
        # -----------------------------------------
        
        dis_net.zero_grad()
        real_imgs = torch.FloatTensor(np.random.normal(4, 1, (batch_size, 1)))
        z = torch.FloatTensor(np.random.randn(batch_size, 1))                           
        
        with torch.no_grad():
            generate_imgs = gen_net(z)
        
        disc_real_logit = dis_net(real_imgs)
        disc_fake_logit = dis_net(generate_imgs)
        
        grad_penalty = calculate_gradient_penalty(dis_net, real_imgs, generate_imgs)

        d_loss = torch.mean(nn.ReLU()(1 - disc_real_logit)) + torch.mean(nn.ReLU()(1 + disc_fake_logit)) + 0.1 * grad_penalty    
        d_loss.backward()
        dis_optimizer.step()
        
        # -----------------------------------------
        #  Train Generator
        # -----------------------------------------
        gen_net.zero_grad()
        
        z = torch.FloatTensor(np.random.randn(batch_size, 1))                                
        generate_imgs = gen_net(z)

        disc_fake_logit = dis_net(generate_imgs)
        
        g_loss = -torch.mean(disc_fake_logit)
        g_loss.backward()
        gen_optimizer.step()
        
        # verbose
        if iter_idx % args.print_freq == 0:
            tqdm.write(
                "[D loss: %f] [G loss: %f]" %
                (d_loss.item(), g_loss.item()))
        if iter_idx % args.inter_gif == 0:
            vis.plot_progress(gen_net)

