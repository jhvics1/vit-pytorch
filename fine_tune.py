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
import torchvision.models as models
import numpy as np
from pl_train import PLLearner
from torchmetrics import Accuracy


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


class InputMonitor(pl.Callback):
    def on_train_batch_start(self, pl_trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx % 10 == 0:
            x, y, z = batch
            sample_input = x
            sample_output = pl_module.model(sample_input.to(pl_module.device), z.to(pl_module.device)).pooler_output
            pl_logger = pl_trainer.logger
            # pl_logger.experiment.add_histogram("input", x, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("label", y, global_step=pl_trainer.global_step)
            pl_logger.experiment.add_histogram("repr_cls", sample_output, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("repr_first", sample_output[1, :], global_step=pl_trainer.global_step)


class Tuner(pl.LightningModule):
    def __init__(self, model, args, length):
        super().__init__()

        self.model = model
        # dim_mlp = self.model.fc[0].in_features
        # self.model.fc = nn.Identity()
        # self.model.train()
        self.fc = nn.Linear(model.dim, 10)

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.optim = torch.optim.Adam(self.fc.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay, amsgrad=False)
        # self.optim = LARS(self.optim, eps=0.0)

        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=length, eta_min=0, last_epoch=-1)
        # w = scheduler.LinearWarmup(self.optim, warmup_steps=args.warmup, last_epoch=-1)
        # sched = scheduler.Scheduler(sched, w)
        # sched.optimizer = self.optim
        # self.scheduler = sched

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        # with torch.no_grad():
        x = self.model(x, 1).detach()
        # x = self.model(x)
        logits = self.fc(x)
        loss = self.criterion(logits.view(-1, 10), labels.view(-1))

        return loss, logits

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        # print(pred_flat)
        # print(labels_flat)
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def configure_optimizers(self):
        return [self.optim]#, [{
        #     'scheduler': self.scheduler,
        #     'interval': 'step',
        #     'frequency': 1,
        #     'reduce_on_plateau': False,
        # }]

    def training_step(self, batch, _):
        x, label = batch

        loss, logits = self.forward(x, label)

        self.log('train_loss', loss.detach().item(), on_step=True, prog_bar=True)
        self.log('train_acc', self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy()), on_step=True,
                 prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, _):
        x, label = batch

        loss, logits = self.forward(x, label)

        self.log('val_loss', loss.detach().item(), prog_bar=True)
        self.log('val_acc', self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy()),
                 prog_bar=True)


to_tensor_transform = T.Compose([T.ToTensor()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vit-simclr-fine-tune')

    parser.add_argument('--checkpoint', '-ch', required=True, type=str, help='checkpoint path')
    parser.add_argument('--lr', '-l', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=100, help="epoch")
    parser.add_argument('--batch-size', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--warmup', '-w', type=int, default=50, help='warmup iteration')
    parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--accumulate', '-ac', type=int, default=1, help='gradient accumulation step')
    parser.add_argument('--num-workers', '-n', type=int, default=16, help='number of workers')
    parser.add_argument('--board-path', '-bp', default='./log', type=str, help='tensorboardx path')

    parser.add_argument('--data', '-d', metavar='DIR', default='../data',
                        help='path to dataset')
    parser.add_argument('--dataset_name', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('--depth', default=6, type=int, help='transformer depth')
    parser.add_argument('--name', required=True, help='name for tensorboard')
    # parser.add_argument('--checkpoint', '-ch', required=True, type=str, help='checkpoint path')

    args = parser.parse_args()
    args.lr *= (args.batch_size / 256)

    print("start")
    train_dataset = datasets.STL10(args.data, split='train', download=True, transform=to_tensor_transform)
    test_dataset = datasets.STL10(args.data, split='test', download=True, transform=to_tensor_transform)
    print("make dataset")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)
    test_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)
    print("loaded dataset!")

    image_size = 96 if args.dataset_name == "stl10" else 32

    logger = pl.loggers.TensorBoardLogger(args.board_path, name='vit-simclr/'+args.name)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # model = models.resnet18(pretrained=False, num_classes=128)
    # model.fc = nn.Sequential(model.fc, nn.ReLU(), nn.Linear(128, 128))
    model = ViT(image_size=image_size,
                patch_size=32,
                dim=512,
                depth=args.depth,
                heads=12,
                mlp_dim=1024,
                pool="mean",
                dropout=0.1,
                emb_dropout=0.1)
    model = ViTSimCLR(model, 256)
    learner = PLLearner.load_from_checkpoint(args.checkpoint, model=model, args=args, length=len(train_loader),
                                             image_size=image_size)

    tuner = Tuner(learner.model.backbone, args, len(train_loader))
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epoch,
        accumulate_grad_batches=args.accumulate,
        default_root_dir="output/vit.model",
        accelerator='ddp',
        logger=logger,
        val_check_interval=5,
        callbacks=[lr_monitor]
    )

    trainer.fit(tuner, train_loader, test_loader)
