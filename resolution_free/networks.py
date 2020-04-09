import torch
import torch.nn as nn
from torchvision import models


class MultiscaleVGG(nn.Module):
    def __init__(self, feature_layers, use_bn=True,
                 use_input_norm=True, use_avg_pool=True):
        super(MultiscaleVGG, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = models.vgg19_bn(pretrained=True)
        else:
            model = models.vgg19(pretrained=True)
        if use_avg_pool:
            for idx, module in model.features._modules.items():
                if module.__class__.__name__ == 'MaxPool2d':
                    model.features._modules[idx] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            # [0.485, 0.456, 0.406] if input in range [0, 1]
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            # [0.229, 0.224, 0.225] if input in range [0, 1]
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        model.eval()
        self.features = nn.ModuleList()
        for i in range(len(feature_layers) - 1):
            feats = nn.Sequential(*list(model.features.children())[feature_layers[i]:feature_layers[i + 1]])
            for k, v in feats.named_parameters():
                v.requires_grad = False
            self.features.append(feats)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        outs = [x]
        for f in self.features:
            out = f(outs[-1])
            outs.append(out)
        return outs[1:]


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers, use_bn=True,
                 use_input_norm=True, use_avg_pool=True):
        super().__init__()
        self.net_perc = MultiscaleVGG(feature_layers=feature_layers,
                                      use_bn=use_bn,
                                      use_input_norm=use_input_norm,
                                      use_avg_pool=use_avg_pool)
        self.loss_features = nn.L1Loss()

    def forward(self, real_image, fake_image, feature_weights=None):
        real_fea = self.net_perc(real_image)
        fake_fea = self.net_perc(fake_image)
        loss_perc = 0
        if feature_weights is None:
            feature_weights = [1] * len(fake_fea)

        for fake_layer, real_layer, weight in zip(fake_fea, real_fea, feature_weights):
            loss_perc_curr = self.loss_features(fake_layer, real_layer).mean()
            loss_perc += loss_perc_curr * weight
        return loss_perc


class RGBNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        internal_dim = config['internal_dim']
        self.p1 = nn.Sequential(nn.Conv2d(2, internal_dim, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(internal_dim, internal_dim * 2, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(internal_dim * 2, internal_dim * 3, 1))
        self.p2 = nn.Sequential(nn.Conv2d(internal_dim * 3 + 2, internal_dim * 2, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(internal_dim * 2, internal_dim * 2, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(internal_dim * 2, internal_dim * 2, 1))
        self.final = nn.Conv2d(internal_dim * 2 + internal_dim * 3, 3, 1)
        self.interm = nn.Conv2d(internal_dim * 3, 3, 1)

    def forward(self, x):
        s1 = self.p1(x)
        s2 = self.p2(torch.cat([x, torch.relu(s1)], dim=1))
        finegrained = self.final(torch.cat([s1, s2], dim=1))
        coarse = self.interm(torch.relu(s1))
        return coarse, finegrained
