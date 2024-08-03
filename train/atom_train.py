"""
author: nabin 
timestamp: Tue Jan 02 2024 02:00 PM
"""

import os
import numpy as np

import torch
import torch.nn as nn
from einops import rearrange

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from self_attention_cv.UnetTr.modules import TranspConv3DBlock, BlueBlock, Conv3DBlock
from self_attention_cv.UnetTr.volume_embedding import Embeddings3D
from self_attention_cv.transformer_vanilla import TransformerBlock

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from argparse import ArgumentParser
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, FBetaScore

AVAIL_GPUS = 4
NUM_NODES = 1
BATCH_SIZE = 4 * 4 * 1 # batch size * available GPU * number of nodes
DATALOADERS = 6
STRATEGY = "ddp_find_unused_parameters_false"
ACCELERATOR = "gpu"
GPU_PLUGIN = "ddp_sharded"
EPOCHS = 100
CHECKPOINT_PATH = "atom_checkpoint"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

DATASET_DIR = "input/"
TRAIN_SUB_GRIDS = "train_sub_grids"
VALID_SUB_GRIDS = "valid_sub_grids"

file = open(os.path.join(DATASET_DIR, 'train_splits.txt'))
train = file.readlines()
print("Training Data file found and the number of protein graph splits are:", len(train))

file = open(os.path.join(DATASET_DIR, 'valid_splits.txt'))
valid = file.readlines()
print("Valid Data file found and the number of protein graph splits are:", len(valid))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class CryoData(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.trarget_transform = target_transform

    def __len__(self):
        return len(train)

    def __getitem__(self, idx):
        cryodata = train[idx]
        cryodata = cryodata.strip("\n")
        # loaded_data = np.load(f"{DATASET_DIR}/{TRAIN_SUB_GRIDS}/{cryodata}")
        file_path = os.path.join(self.root, TRAIN_SUB_GRIDS, cryodata)
        # print(f"Loading file: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        loaded_data = np.load(file_path)
        
        protein_manifest = loaded_data['protein_grid']
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        atom_manifest = loaded_data['atom_grid']
        atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)
        # esm_embeds = loaded_data['embeds']
        # esm_embeds_torch = torch.from_numpy(esm_embeds).type(torch.FloatTensor)

        return [protein_torch, atom_torch]


class CryoData_valid(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.trarget_transform = target_transform

    def __len__(self):
        return len(valid)

    def __getitem__(self, idx):
        cryodata = valid[idx]
        cryodata = cryodata.strip("\n")
        # loaded_data = np.load(f"{DATASET_DIR}/{VALID_SUB_GRIDS}/{cryodata}")
        file_path = os.path.join(DATASET_DIR, VALID_SUB_GRIDS, cryodata)
        # print(f"Loading file: {file_path}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
        loaded_data = np.load(file_path)
        
        protein_manifest = loaded_data['protein_grid']
        protein_torch = torch.from_numpy(protein_manifest).type(torch.FloatTensor)
        atom_manifest = loaded_data['atom_grid']
        atom_torch = torch.from_numpy(atom_manifest).type(torch.FloatTensor)
        return [protein_torch, atom_torch]


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers

        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=True))

    def forward(self, x):
        extract_layers = []
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x)
            if (depth + 1) in self.extract_layers:
                extract_layers.append(x)
        return extract_layers


class Transformer_UNET(nn.Module):
    def __init__(self, img_shape=(32, 32, 32), input_dim=1, output_dim=4, embed_dim=768, patch_size=16,
                 num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                 dim_linear_block=3072):
        super().__init__()
        self.num_layers = 12
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.patch_dim = [int(x / patch_size) for x in
                          img_shape]

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d

        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim, cube_size=img_shape,
                                  patch_size=patch_size, dropout=dropout)

        self.transformer = TransformerEncoder(embed_dim, num_heads, self.num_layers, dropout, ext_layers,
                                              dim_linear_block=dim_linear_block)

        self.init_conv = Conv3DBlock(input_dim, base_filters, double=True, norm=self.norm)

        # blue block

        self.z3_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 2, layers=3)
        self.z6_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 4, layers=2)
        self.z9_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 8, layers=1)

        # Green block

        self.z12_deconv = TranspConv3DBlock(embed_dim, base_filters * 8)
        self.z9_deconv = TranspConv3DBlock(base_filters * 8, base_filters * 4)
        self.z6_deconv = TranspConv3DBlock(base_filters * 4, base_filters * 2)
        self.z3_deconv = TranspConv3DBlock(base_filters * 2, base_filters)

        # yellow block

        self.z9_conv = Conv3DBlock(base_filters * 8 * 2, base_filters * 8, double=True, norm=self.norm)
        self.z6_conv = Conv3DBlock(base_filters * 4 * 2, base_filters * 4, double=True, norm=self.norm)
        self.z3_conv = Conv3DBlock(base_filters * 2 * 2, base_filters * 2, double=True, norm=self.norm)

        # out convolutions

        self.out_conv = nn.Sequential(
            # last yellow conv block
            Conv3DBlock(base_filters * 2, base_filters, double=True, norm=self.norm),

            # brown block, final classification
            nn.Conv3d(base_filters, output_dim, kernel_size=1, stride=1))

    def forward(self, x):
        transformer_input = self.embed(x)
        z3, z6, z9, z12 = map(
            lambda t: rearrange(t, 'b (x y z) d -> b d x y z', x=self.patch_dim[0], y=self.patch_dim[1],
                                z=self.patch_dim[2]), self.transformer(transformer_input))

        # Blue convs
        z0 = self.init_conv(x)
        z3 = self.z3_blue_conv(z3)
        z6 = self.z6_blue_conv(z6)
        z9 = self.z9_blue_conv(z9)

        # Green blocks for z12
        z12 = self.z12_deconv(z12)

        # concat + yellow conv

        y = torch.cat([z12, z9], dim=1)
        y = self.z9_conv(y)

        # Green blocks for z6
        y = self.z9_deconv(y)

        # concat + yellow conv
        y = torch.cat([y, z6], dim=1)

        # y = torch.cat([attention_values, z6], dim=1)
        y = self.z6_conv(y)

        # Green block for z3
        y = self.z6_deconv(y)

        # concat + yellow conv

        y = torch.cat([y, z3], dim=1)

        y = self.z3_conv(y)

        y = self.z3_deconv(y)

        y = torch.cat([y, z0], dim=1)
        return self.out_conv(y)


def calc_ce_weights(batch):
    y_zeros = (batch == 0.).sum()
    y_ones = (batch == 1.).sum()
    y_two = (batch == 2.).sum()
    y_three = (batch == 3.).sum()
    nSamples = [y_zeros, y_ones, y_two, y_three]
    normedWeights_1 = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = [x + 1e-5 for x in normedWeights_1]
    balance_weights = torch.FloatTensor(normedWeights).to("cuda")
    return balance_weights


class VoxelClassify(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer_UNET(**model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

        self.metrics_macro = MetricCollection([Accuracy(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               Precision(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               Recall(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               F1Score(task='multiclass', num_classes=4, average='macro', mdmc_average="global"),
                                               FBetaScore(task='multiclass', num_classes=4, average='macro', mdmc_average="global")])
        
        self.metrics_weighted = MetricCollection([Accuracy(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  Precision(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  Recall(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  F1Score(task='multiclass', num_classes=4, average='weighted', mdmc_average="global"),
                                                  FBetaScore(task='multiclass', num_classes=4, average='weighted', mdmc_average="global")])
        """
        self.metrics_micro = MetricCollection([Accuracy(num_classes=4, average='micro', mdmc_average="global"),
                                               Precision(num_classes=4, average='micro', mdmc_average="global"),
                                               Recall(num_classes=4, average='micro', mdmc_average="global"),
                                               F1Score(num_classes=4, average='micro', mdmc_average="global"),
                                               FBetaScore(num_classes=4, average='micro', mdmc_average="global")])
        """
        self.train_metrics_macro = self.metrics_macro.clone(prefix="train_macro_")
        self.valid_metrics_macro = self.metrics_macro.clone(prefix="valid_macro_")
        self.test_metrics_macro = self.metrics_macro.clone(prefix="test_macro_")

        self.train_metrics_weighted = self.metrics_weighted.clone(prefix="train_weighted_")
        self.valid_metrics_weighted = self.metrics_weighted.clone(prefix="valid_weighted_")
        self.test_metrics_weighted = self.metrics_weighted.clone(prefix="test_weighted_")
        """
        self.train_metrics_micro = self.metrics_micro.clone(prefix="train_micro_")
        self.valid_metrics_micro = self.metrics_micro.clone(prefix="valid_micro_")
        self.test_metrics_micro = self.metrics_micro.clone(prefix="test_micro_")
        
        """

    def forward(self, data):
        x = self.model(data)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1,
                                                                  patience=10, eps=1e-10, verbose=True)
        metric_to_track = 'train_loss'
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': metric_to_track
        }

    def training_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1]
        protein_data = torch.unsqueeze(protein_data, 1)
        y_hat = self.forward(protein_data)
        balance_weights = calc_ce_weights(atom_data)
        loss_fn_train = nn.CrossEntropyLoss(weight=balance_weights)
        loss = loss_fn_train(y_hat, atom_data.long())
        
        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        # Compute and log metrics
        metric_log_macro = self.train_metrics_macro(y_hat, atom_data.int())
        metric_log_weighted = self.train_metrics_weighted(y_hat, atom_data.int())
        self.log_dict(metric_log_macro, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1]
        protein_data = torch.unsqueeze(protein_data, 1)
        y_hat = self.forward(protein_data)
        loss = self.loss_fn(y_hat, atom_data.long())
        
        # Log the loss
        self.log('valid_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        # Compute and log metrics
        metric_log_macro = self.valid_metrics_macro(y_hat, atom_data.int())
        metric_log_weighted = self.valid_metrics_weighted(y_hat, atom_data.int())
        self.log_dict(metric_log_macro, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        protein_data, atom_data = batch[0], batch[1]
        protein_data = torch.unsqueeze(protein_data, 1)
        y_hat = self.forward(protein_data)
        loss = self.loss_fn(y_hat, atom_data.long())
        # loss = loss_fn_train(y_hat, batch.y.long())
        metric_log_macro = self.test_metrics_macro(y_hat, atom_data.int())
        # metric_log_micro = self.test_metrics_micro(y_hat, amino_data.int())
        metric_log_weighted = self.test_metrics_weighted(y_hat, atom_data.int())
        self.log_dict(metric_log_macro,  on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(metric_log_micro)
        self.log_dict(metric_log_weighted, on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser

@rank_zero_only
def get_logger():
    return TensorBoardLogger("tb_logs", name="cryo2struct")

@rank_zero_only
def make_logger_dir():
    os.makedirs("tb_logs/cryo2struct", exist_ok=True)
    
@rank_zero_only
def print_train_info(train_splits, valid_splits):
    print("Training Data file found and the number of protein graph splits are:", len(train_splits))
    print("Valid Data file found and the number of protein graph splits are:", len(valid_splits))

def train_node_classifier():
    pl.seed_everything(42)
    parser = ArgumentParser()
    
    # 添加 PyTorch Lightning 的参数
    parser = pl.Trainer.add_argparse_args(parser)
    
    # 添加自定义的模型相关参数
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--multi_gpu_backend', type=str, default='dp', help="Backend to use for multi-GPU training")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--num_epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS, help="Number of workers for data loading")
    
    args = parser.parse_args()

    # 获取日志记录器
    make_logger_dir()
    logger = get_logger()

    # 设置训练器参数
    trainer_args = {
        'logger': logger,
        'accelerator': 'gpu',
        'devices': args.num_gpus,
        'max_epochs': args.num_epochs,
        'precision': 16,
        'strategy': args.multi_gpu_backend,
        'log_every_n_steps': 10,
        'callbacks': [
            ModelCheckpoint(dirpath=CHECKPOINT_PATH, save_top_k=3, monitor='valid_loss'),
            LearningRateMonitor(logging_interval='epoch')
        ]
    }

    trainer = pl.Trainer(**trainer_args)

    # 加载数据集
    dataset = CryoData(DATASET_DIR)
    dataset_valid = CryoData_valid(DATASET_DIR)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_dataloader_workers)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_dataloader_workers)

    # 创建模型实例
    model = VoxelClassify(learning_rate=args.learning_rate, img_shape=(32, 32, 32), input_dim=1, output_dim=4, embed_dim=768,
                          patch_size=16, num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance", base_filters=16,
                          dim_linear_block=3072)

    # 开始训练
    trainer.fit(model, train_loader, valid_loader)

if __name__ == "__main__":
    train_node_classifier()
