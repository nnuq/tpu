import torch.nn as nn

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super(GenBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), n_classes=0):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Sigmoid()(self.c5(h))
        return h

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation
        
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        self.cls6 = nn.Linear(self.ch * 8 * 8, 5, bias=False)
        
    def forward(self, x, type_):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        if type_ == 'feat':
            return h.view(-1, self.ch * 8 * 8)
        if type_ == 'out':
            return self.l5(h.sum(2).sum(2))
        if type_ == 'cls':
            return self.cls6(h.view(-1, self.ch * 8 * 8))    

class EncBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            hidden_channels,
            ksize=3,
            pad=1,
            activation=nn.PReLU(init=0.2),
            downsample=False):
        super(EncBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample     
        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            
    def residual(self, x):
        h = x
        h = self.activation(self.b1(h))
        h = self.c1(h)
        h = self.activation(self.b2(h))
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h
    
    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)        

class Encoder(nn.Module):
    def __init__(self, args, activation=nn.PReLU(init=0.2)):
        super(Encoder, self).__init__()
        self.ch = args.ef_dim
        self.activation = activation
        self.block1 = nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding=1)
        self.block2 = EncBlock(
            args,
            self.ch,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = EncBlock(
            args,
            self.ch,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block4 = EncBlock(
            args,
            self.ch,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)    
        self.b1 = nn.BatchNorm2d(self.ch)    
        self.dense = nn.Linear(self.ch * 4 * 4, 128)
         
    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3, self.block4]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.activation(self.b1(h))
        h = h.view(-1, self.ch * 4 * 4)
        return self.dense(h)