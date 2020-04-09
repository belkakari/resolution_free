import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from argparse import Namespace

from resolution_free.networks import RGBNet, PerceptualLoss
from resolution_free.utils import blur, get_grid
from resolution_free.data import OneImageSet


class SingleImageModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = Namespace(**config)
        self.config = config
        self.net = RGBNet(self.config['model'])
        self.perceptual_loss = PerceptualLoss(config['layers'],
                                              use_bn=False,
                                              use_input_norm=True,
                                              use_avg_pool=True)

    def forward(self, x):
        coarse, finegrained = self.net(x)
        return coarse, finegrained

    def training_step(self, batch, batch_nb):
        img_crop, grid_crop = batch
        coarse, finegrained = self(grid_crop)
        mse_loss = nn.MSELoss()(torch.sigmoid(coarse),
                                blur(img_crop, sigma=3, kernel_size=15))
        perc_loss = self.perceptual_loss(torch.sigmoid(coarse).detach() + torch.tanh(finegrained),
                                         img_crop,
                                         self.config['vgg_weights'])
        loss = mse_loss + 0.1 * perc_loss
        tensorboard_logs = {'train_loss': loss,
                            'mse_loss': mse_loss,
                            'perc_loss': perc_loss}
        return {'loss': loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])

    def train_dataloader(self):
        return DataLoader(OneImageSet(self.config['img_path'],
                                      self.config['max_crop'],
                                      self.config['len_dloader']),
                          batch_size=self.config['data']['batch_size'],
                          num_workers=self.config['data']['num_workers'])

    def on_epoch_end(self):
        grid = get_grid(1024, 1024, b=1, norm=True).permute(0, 3, 1, 2) * 2 - 1
        if self.on_gpu:
            grid = grid.cuda()
        coarse, finegrained = self(grid)
        final_img = torch.sigmoid(coarse) + torch.tanh(finegrained)
        for_grid = torch.cat([torch.sigmoid(coarse),
                              ((torch.tanh(finegrained) + 1.) / 2.),
                              final_img.clamp(0, 1)], dim=0)
        grid = torchvision.utils.make_grid(for_grid,
                                           nrow=1)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)
