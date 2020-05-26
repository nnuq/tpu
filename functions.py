import torch
import torch.nn as nn
import logging
import os
import numpy as np
from copy import deepcopy
from torch.autograd import Variable
from torch import autograd
from imageio import imsave
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.transforms.functional import rotate
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)

def calculate_gradient_penalty(dis_net: nn.Module, real_samples, fake_samples):
        
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = dis_net(interpolates, 'out')
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


def argument_image_rotation_and_fake(X, ridx = None):
    
    n = X.size()[0]
    
    l_0  = torch.cuda.LongTensor([0]).repeat(n)

    X_90  = torch.rot90(X, 1, [2, 3])
    l_90  = torch.cuda.LongTensor([1]).repeat(n)

    X_180  = torch.rot90(X, 2, [2, 3])
    l_180  = torch.cuda.LongTensor([2]).repeat(n)

    X_270  = torch.rot90(X, 3, [2, 3])
    l_270  = torch.cuda.LongTensor([3]).repeat(n)

    Xarg = torch.cat([X, X_90, X_180, X_270])
    larg = torch.cat([l_0, l_90, l_180, l_270])
        
    if ridx is None:
        ridx = np.arange(4 * n)
        np.random.shuffle(ridx)
        ridx = ridx[0 : n]
    
    X_out = []
    l_out = []
    
    for index in ridx:
        X_out.append(Xarg[index])
        l_out.append(larg[index])

    rot_labels = one_hot(torch.stack(l_out), 5)    
    
    return torch.stack(X_out), rot_labels.double(), ridx    


def argument_image_rotation_and_fake_mix(X, X_f, ridx = None):
    
    n = X.size()[0]
    
    l_0  = torch.cuda.LongTensor([0]).repeat(n)

    X_90  = torch.rot90(X, 1, [2, 3])
    l_90  = torch.cuda.LongTensor([1]).repeat(n)

    X_180  = torch.rot90(X, 2, [2, 3])
    l_180  = torch.cuda.LongTensor([2]).repeat(n)

    X_270  = torch.rot90(X, 3, [2, 3])
    l_270  = torch.cuda.LongTensor([3]).repeat(n)

    l_fake  = torch.cuda.LongTensor([4]).repeat(n)

    Xarg = torch.cat([X, X_90, X_180, X_270, X_f])
    larg = torch.cat([l_0, l_90, l_180, l_270, l_fake])
        
    if ridx is None:
        ridx = np.arange(5 * n)
        np.random.shuffle(ridx)
        ridx = ridx[0 : n]
    
    X_out = []
    l_out = []
    
    for index in ridx:
        X_out.append(Xarg[index])
        l_out.append(larg[index])
    
    rot_labels = one_hot(torch.stack(l_out), 5)
    
    return torch.stack(X_out), rot_labels.double(), ridx    


def train(device, args, gen_net: nn.Module, dis_net: nn.Module, enc_net: nn.Module, gen_optimizer, dis_optimizer, 
             ae_recon_optimizer, ae_reg_optimizer, train_loader, epoch, writer_dict, schedulers=None):
    writer = writer_dict['writer']
    
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader.per_device_loader(device))):
        global_steps = writer_dict['train_global_steps']
        batch_size = imgs.shape[0]
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # -----------------------------------------
        #  Train Autoencoder (reconstruction loss)
        # -----------------------------------------
        
        gen_net.zero_grad()
        enc_net.zero_grad()

        with torch.no_grad():
            feature_real = dis_net(real_imgs, 'feat')
        
        z_enc = enc_net(real_imgs)
        reconstructed_imgs = gen_net(z_enc)
        feature_recon = dis_net(reconstructed_imgs, 'feat')

        ae_recon_loss = torch.mean((feature_real - feature_recon) ** 2)
        ae_recon_loss.backward(retain_graph=True)
        xm.optimizer_step(ae_recon_optimizer)
        writer.add_scalar('ae_recon_loss', ae_recon_loss.item(), global_steps)
        
        # -----------------------------------------
        #  Train Autoencoder (regularization loss)
        # -----------------------------------------
        
        gen_net.zero_grad()
        enc_net.zero_grad()

        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (batch_size, args.latent_dim)))                            
        generate_imgs = gen_net(z)
        feature_fake = dis_net(generate_imgs, 'feat')
        
        lambda_w    = np.sqrt(args.latent_dim * 1.0/feature_recon.size()[1])
        md_x        = torch.mean(feature_real - feature_fake)
        md_z        = torch.mean(z_enc - z) * lambda_w
        ae_reg_loss = (md_x - md_z) ** 2
        ae_reg_loss.backward()
        xm.optimizer_step(ae_reg_optimizer)
        writer.add_scalar('ae_reg_loss', ae_reg_loss.item(), global_steps)

        # -----------------------------------------
        #  Train Discriminator
        # -----------------------------------------
        dis_net.zero_grad()
        
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (batch_size, args.latent_dim)))                            
        with torch.no_grad():
            z_enc = enc_net(real_imgs)
            generate_imgs = gen_net(z)
            reconstructed_imgs = gen_net(z_enc)

        images_mix, larg_mix, _ = argument_image_rotation_and_fake_mix(real_imgs, generate_imgs)    
        
        disc_real_logit = dis_net(real_imgs, 'out')
        disc_fake_logit = dis_net(generate_imgs, 'out')
        disc_recon_logit = dis_net(reconstructed_imgs, 'out')
        mixe_cls = dis_net(images_mix, 'cls')
        
        d_acc = torch.sum(binary_cross_entropy_with_logits(mixe_cls, larg_mix))
        
        grad_penalty = calculate_gradient_penalty(dis_net, real_imgs, generate_imgs)
        
        l_1 = torch.mean(nn.ReLU()(1 - disc_real_logit))
        l_2 = torch.mean(nn.ReLU()(1 - disc_recon_logit))
        l_3 = torch.mean(nn.ReLU()(1 + disc_fake_logit))    
        
        t = float(global_steps)/args.num_iter
        
        mu = max(min((t*0.1-0.05)*2, 0.05),0.0)
        w_real  = 0.95 + mu
        w_recon = 0.05 - mu
        
        d_loss = w_real * l_1 + w_recon * l_2 + l_3 + grad_penalty
        d_loss_cls = args.lambda_d * d_acc
        d_loss += d_loss_cls
        d_loss.backward()
        xm.optimizer_step(dis_optimizer)
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        
        # -----------------------------------------
        #  Train Generator
        # -----------------------------------------
        gen_net.zero_grad()
        
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (batch_size, args.latent_dim)))                            
        generate_imgs = gen_net(z)
        
        Xarg, larg, ridx = argument_image_rotation_and_fake(real_imgs)
        Xarg_f, larg_f, _ = argument_image_rotation_and_fake(generate_imgs, ridx=ridx)
        
        with torch.no_grad():
            disc_real_logit = dis_net(real_imgs, 'out')
            real_cls = dis_net(Xarg, 'cls')
            g_real_acc = torch.sum(binary_cross_entropy_with_logits(real_cls, larg))
        
        disc_fake_logit = dis_net(generate_imgs, 'out')
        fake_cls = dis_net(Xarg_f, 'cls')
        g_fake_acc = torch.sum(binary_cross_entropy_with_logits(fake_cls, larg_f))       
        
        g_loss  = torch.abs(torch.mean(nn.Sigmoid()(disc_real_logit)) - torch.mean(nn.Sigmoid()(disc_fake_logit)))
        g_loss_cls =  args.lambda_g * torch.abs(g_fake_acc - g_real_acc)
        g_loss += g_loss_cls
        g_loss.backward()
        xm.optimizer_step(gen_optimizer)
        writer.add_scalar('g_loss', g_loss.item(), global_steps)
        
        # adjust learning rate
        if schedulers:
            gen_scheduler, dis_scheduler, ae_recon_scheduler, ae_reg_scheduler = schedulers
            g_lr = gen_scheduler.step(global_steps)
            d_lr = dis_scheduler.step(global_steps)
            e_lr = ae_recon_scheduler.step(global_steps)
            e_lr = ae_reg_scheduler.step(global_steps)
        
        # verbose
        if iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [Recon loss: %f] [Reg loss: %f] [D loss: %f] [G loss: %f]" %
                (epoch, args.num_epochs, iter_idx % len(train_loader), len(train_loader), ae_recon_loss.item(), 
                                                            ae_reg_loss.item(), d_loss.item(), g_loss.item()))
        
        writer_dict['train_global_steps'] = global_steps + 1
    
     
def validate(args, fid_stat, gen_net: nn.Module, writer_dict, valid_loader):
    
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
        
    fid_buffer_dir_gen = os.path.join(args.path_helper['sample_path'], 'fid_buffer_gen')
    os.makedirs(fid_buffer_dir_gen, exist_ok=True)
    
    for iter_idx, (real_imgs, _) in enumerate(tqdm(valid_loader)):
        z = torch.cuda.FloatTensor(np.random.uniform(-1, 1, (args.eval_batch_size, args.latent_dim)))
        
        gen_imgs = gen_net(z)
        
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir_gen, f'iter{iter_idx}_b{img_idx}.png')
            save_image(img, file_name, normalize=True)
        
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths(paths=[fid_stat, fid_buffer_dir_gen], batch_size=args.eval_batch_size)
    print(f"FID score: {fid_score}")

    os.system('rm -r {}'.format(fid_buffer_dir_gen))
    
    writer.add_scalar('FID_score', fid_score, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    return fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):
        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten