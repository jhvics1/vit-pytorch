import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import argparse
from vit_pytorch import ViT
import torch.nn.functional as F

from torchvision import transforms as T
import framework.optimizers.scheduler as scheduler
from vit_simclr import ViTSimCLR
from pytorch_lightning.callbacks import LearningRateMonitor
from einops import rearrange, repeat
from torch import nn
import random
import math


def default(val, def_val):
    return def_val if val is None else val


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class PLLearner(pl.LightningModule):
    def __init__(self, model, args, length, image_size):
        super().__init__()

        self.model = model
        self.batch_size = args.batch_size
        self.depth = args.depth

        self.fp16_precision = False

        self.optim = torch.optim.Adam(self.model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay, amsgrad=False)
        # self.optim = LARS(self.optim, eps=0.0)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=length, eta_min=0, last_epoch=-1)
        w = scheduler.LinearWarmup(self.optim, warmup_steps=args.warmup, last_epoch=-1)
        sched = scheduler.Scheduler(sched, w)
        sched.optimizer = self.optim
        self.scheduler = sched

        # self.scaler = GradScaler(enabled=self.fp16_precision)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.label = None
        self.mask = None
        self.labels = None
        self.label_layer = None
        self.mask_layer = None
        self.labels_layer = None

        DEFAULT_AUG = torch.nn.Sequential(
            T.RandomResizedCrop((image_size, image_size)),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * image_size), sigma=(0.1, 2.0)),
            # GaussianBlur(kernel_size=int(0.1 * image_size))
        )

        self.augment1 = default(None, DEFAULT_AUG)
        self.augment2 = default(None, self.augment1)

        self.save_hyperparameters()

    def forward(self, x):
        # with autocast(enabled=self.fp16_precision):
        features = self.model(x)
        logits, labels = self.info_nce_loss(features)
        loss = self.criterion(logits, labels)
        return loss

    def get_layer_loss(self):
        # with autocast(enabled=self.fp16_precision):
        layer_features = self.model.layer_repr_mlp()
        predictions = self.model.layer_repr_pred()
        logits1, logits2, labels = self.info_nce_loss_layers(layer_features, predictions)
        loss1 = self.criterion(logits1, labels)
        loss2 = self.criterion(logits2, labels)
        loss = loss1 + loss2
        loss /= 2
        return loss

    def training_step(self, batch, _):
        aug1 = self.augment1(batch[0])
        aug2 = self.augment2(batch[0])
        output_loss = self.forward(torch.cat((aug1, aug2), dim=0))
        # layer_loss = self.get_layer_loss() / (self.depth - 1)
        # ratio = 0.5 * (math.exp(self.current_epoch/100)-1)
        # loss = output_loss + ratio * layer_loss
        loss = output_loss
        # self.logger.experiment.add_scalar('layer_loss', layer_loss.detach().item(), self.global_step)
        # self.logger.experiment.add_scalar('ratio', ratio, self.global_step)

        self.logger.experiment.add_scalar('output_loss', output_loss.detach().item(), self.global_step)
        self.logger.experiment.add_scalar('train_loss', loss.detach().item(), self.global_step)
        return {'loss': loss}

    def configure_optimizers(self):
        return [self.optim], [{
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
        }]

    # def on_after_backward(self):
    #     opt = self.optimizers()
    #     self.scaler.step(opt)
    #     self.scaler.update()

    def info_nce_loss(self, features):
        if self.labels is None:
            device = features.device
            b, _ = features.shape
            self.labels = torch.cat([torch.arange(b / 2) for i in range(2)], dim=0)
            self.labels = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1)).float()
            self.labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     2 * self.batch_size, 2 * self.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        if self.mask is None:
            self.mask = torch.eye(self.labels.shape[0], dtype=torch.bool).to(device)
        labels = self.labels[~self.mask].view(self.labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~self.mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        if self.label is None:
            self.label = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / 0.1  # temperature
        return logits, self.label

    def info_nce_loss_layers(self, layer_features, prediction):
        device = layer_features.device
        low_layer = layer_features[:-1]
        high_layer = layer_features[1:].detach()
        _, b, _ = high_layer.shape
        b = int(b / 2)
        low_layer1, low_layer2 = torch.split(low_layer, [b, b], dim=1)
        high_layer1, high_layer2 = torch.split(high_layer, [b, b], dim=1)

        if self.labels_layer is None:
            #
            # self.labels_layer = torch.cat([torch.arange(b / 2) for i in range(2)], dim=0)
            # self.labels_layer = (self.labels_layer.unsqueeze(0) == self.labels_layer.unsqueeze(1)).float()
            # self.labels_layer = torch.unsqueeze(self.labels_layer, 0)
            # self.labels_layer = repeat(self.labels_layer, '() b c -> d b c', d=self.depth - 1)
            # self.labels_layer.to(device)
            self.labels_layer = torch.eye(high_layer1.shape[1], dtype=torch.bool)
            self.labels_layer = torch.unsqueeze(self.labels_layer, 0)
            self.labels_layer = repeat(self.labels_layer, '() b c -> d b c', d=self.depth - 1)
            self.labels_layer.to(device)

        similarity_matrix1 = torch.matmul(low_layer1, torch.transpose(high_layer1, 1, 2))
        similarity_matrix2 = torch.matmul(low_layer2, torch.transpose(high_layer2, 1, 2))
        #
        # if self.mask_layer is None:
        #     self.mask_layer = torch.eye(self.labels_layer.shape[1], dtype=torch.bool)
        #     self.mask_layer = torch.unsqueeze(self.mask_layer, 0)
        #     self.mask_layer = repeat(self.mask_layer, '() b c -> d b c', d=self.depth - 1)
        #     self.mask_layer.to(device)
        # labels = self.labels_layer[~self.mask_layer].view(self.labels_layer.shape[0], self.labels_layer.shape[1], -1)
        # similarity_matrix = similarity_matrix[~self.mask_layer].view(similarity_matrix.shape[0],
        #                                                              similarity_matrix.shape[1], -1)

        positives1 = similarity_matrix1[self.labels_layer.bool()].view(self.labels_layer.shape[0],
                                                                       self.labels_layer.shape[1], -1)

        negatives1 = similarity_matrix1[~self.labels_layer.bool()].view(similarity_matrix1.shape[0],
                                                                        similarity_matrix1.shape[1], -1)

        logits1 = torch.cat([positives1, negatives1], dim=2)
        # logits = torch.reshape(logits, )
        logits1 = rearrange(logits1, 'd b c -> (d b) c')

        positives2 = similarity_matrix2[self.labels_layer.bool()].view(self.labels_layer.shape[0],
                                                                       self.labels_layer.shape[1], -1)

        negatives2 = similarity_matrix2[~self.labels_layer.bool()].view(similarity_matrix2.shape[0],
                                                                        similarity_matrix2.shape[1], -1)

        logits2 = torch.cat([positives2, negatives2], dim=2)
        # logits = torch.reshape(logits, )
        logits2 = rearrange(logits2, 'd b c -> (d b) c')

        if self.label_layer is None:
            self.label_layer = torch.zeros((logits1.shape[0]), dtype=torch.long).to(device)

        temperature = 0.07

        logits1 /= temperature
        logits2 /= temperature

        return logits1, logits2, self.label_layer


# class Monitor(pl.Callback):
#     def on_train_batch_start(self, pl_trainer, pl_module, batch, batch_idx, dataloader_idx):
#         if batch_idx % 100 == 0:
#             pl_logger = pl_trainer.logger
#             pl_logger.experiment.add_histogram("input", batch, global_step=pl_trainer.global_step)
#
#

to_tensor_transform = T.Compose([T.ToTensor()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vit-simclr')

    parser.add_argument('--lr', '-l', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=100, help="epoch")
    parser.add_argument('--batch-size', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--warmup', '-w', type=int, default=500, help='warmup iteration')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--accumulate', '-ac', type=int, default=1, help='gradient accumulation step')
    parser.add_argument('--num-workers', '-n', type=int, default=16, help='number of workers')
    parser.add_argument('--board-path', '--bp', default='./log', type=str, help='tensorboardx path')

    parser.add_argument('--data', '-d', metavar='DIR', default='../data',
                        help='path to dataset')
    parser.add_argument('--dataset_name', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('--name', required=True, help='name for tensorboard')
    parser.add_argument('--depth', default=6, type=int, help='transformer depth')

    args = parser.parse_args()
    args.lr *= (args.batch_size / 256)

    print("start")
    dataset = datasets.STL10(args.data, split='unlabeled', download=True, transform=to_tensor_transform)
    print("make dataset")
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
    print("loaded dataset!")

    image_size = 96 if args.dataset_name == "stl10" else 32

    model = ViT(image_size=image_size,
                patch_size=32,
                dim=512,
                depth=args.depth,
                heads=12,
                mlp_dim=1024,
                pool="cls",
                dropout=0.1,
                emb_dropout=0.1)
    model = ViTSimCLR(model, 256)
    # model = models.resnet18(pretrained=False, num_classes=128)
    # dim_mlp = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

    learner = PLLearner(model, args, len(train_loader), image_size)

    logger = pl.loggers.TensorBoardLogger(args.board_path, name='vit-simclr/'+args.name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epoch,
        accumulate_grad_batches=args.accumulate,
        default_root_dir="output/vit.model",
        accelerator='ddp',
        logger=logger,
        callbacks=[lr_monitor]
    )

    trainer.fit(learner, train_loader)
