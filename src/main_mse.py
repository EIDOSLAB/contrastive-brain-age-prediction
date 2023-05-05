import datetime
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import argparse
import models
import losses
import time
import wandb
import torch.utils.tensorboard

from torchvision import transforms
from util import AverageMeter, MAE, ensure_dir, set_seed, arg2bool, save_model
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, compute_site_ba
from data import FeatureExtractor, OpenBHB, bin_age
from data.transforms import Crop, Pad, Cutout

def parse_arguments():
    parser = argparse.ArgumentParser(description="Augmentation for multiview",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=50)

    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)

    parser.add_argument('--epochs', type=int, help='number of epochs', default=200)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='cosine')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=None)

    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"], default="sgd")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)

    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')

    parser.add_argument('--method', type=str, help='loss function', choices=['mae', 'mse'], default='mae')
    
    
    parser.add_argument('--train_all', type=arg2bool, help='train on all dataset including validation (int+ext)', default=False)
    parser.add_argument('--tf', type=str, help='data augmentation', choices=['none', 'crop', 'cutout', 'all'], default='none')

    parser.add_argument('--amp', action='store_true', help='use amp')

    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True
    
    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    opts.fairkl_kernel = opts.kernel != 'none'
    return opts

def get_transforms(opts):
    selector = FeatureExtractor("vbm")
    
    if opts.tf == 'none':
        aug = transforms.Lambda(lambda x: x)

    elif opts.tf == 'crop':
        aug = transforms.Compose([
            Crop((1, 121, 128, 121), type="random"),
            Pad((1, 128, 128, 128))
        ])  

    elif opts.tf == 'cutout':
        aug = Cutout(patch_size=[1, 32, 32, 32], probability=0.5)

    elif opts.tf == 'all':
        aug = transforms.Compose([
            Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
            Crop((1, 121, 128, 121), type="random"),
            Pad((1, 128, 128, 128))
        ])
    
    T_pre = transforms.Lambda(lambda x: selector.transform(x))
    T_train = transforms.Compose([
        T_pre,
        aug,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    T_test = transforms.Compose([
        T_pre,
        transforms.Lambda(lambda x: torch.from_numpy(x).float()),
        transforms.Normalize(mean=0.0, std=1.0)
    ])

    return T_train, T_test


def load_data(opts):
    T_train, T_test = get_transforms(opts)
    
    train_dataset = OpenBHB(opts.data_dir, train=True, internal=True, transform=T_train,
                            load_feats=opts.biased_features)

    if opts.train_all:
        valint_feats, valext_feats = None, None
        if opts.biased_features is not None:
            valint_feats = opts.biased_features.replace('.pth', '_valint.pth')
            valext_feats = opts.biased_features.replace('.pth', '_valext.pth')

        valint = OpenBHB(opts.data_dir, train=False, internal=True, transform=T_train,
                         load_feats=valint_feats)
        valext = OpenBHB(opts.data_dir, train=False, internal=False, transform=T_train,
                         load_feats=valext_feats)      
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, valint, valext])
        print("Total dataset lenght:", len(train_dataset))


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, 
                                               num_workers=8, persistent_workers=True)
   
    test_internal = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=True, transform=T_test), 
                                                batch_size=opts.batch_size, shuffle=False, num_workers=8,
                                                persistent_workers=True)
    test_external = torch.utils.data.DataLoader(OpenBHB(opts.data_dir, train=False, internal=False, transform=T_test), 
                                                batch_size=opts.batch_size, shuffle=False, num_workers=8,
                                                persistent_workers=True)

    return train_loader, test_internal, test_external

def load_model(opts):
    if 'resnet' in opts.model:
        model = models.SupRegResNet(opts.model)
    
    elif 'alexnet' in opts.model:
        model = models.SupRegAlexNet()
    
    elif 'densenet121' in opts.model:
        model = models.SupRegDenseNet()
    
    else:
        raise ValueError("Unknown model", opts.model)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)
    model = model.to(opts.device)
    
    methods = {
        'mae': F.l1_loss,
        'mse': F.mse_loss
    }
    regression_loss = methods[opts.method]

    return model, regression_loss

def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, 
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    return optimizer

def train(train_loader, model, criterion, optimizer, opts, epoch):
    loss = AverageMeter()
    mae = MAE()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    model.train()

    t1 = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - t1)

        images, labels = images.to(opts.device), labels.to(opts.device)
        bsz = labels.shape[0]

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):
            output, features = model(images)
            output = output.view(-1)
            running_loss = criterion(output, features, labels.float())
        
        optimizer.zero_grad()
        if scaler is None:
            running_loss.backward() 
            optimizer.step()
        else:
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        loss.update(running_loss.item(), bsz)
        mae.update(output, labels)

        batch_time.update(time.time() - t1)
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t"
                  f"MAE {mae.avg:.3f}")

        t1 = time.time()

    return loss.avg, mae.avg, batch_time.avg, data_time.avg

@torch.no_grad()
def test(test_loader, model, criterion, opts, epoch):
    loss = AverageMeter()
    mae = MAE()
    batch_time = AverageMeter()

    model.eval()
    t1 = time.time()
    for idx, (images, labels, _) in enumerate(test_loader):
        images, labels = images.to(opts.device), labels.to(opts.device)
        bsz = labels.shape[0]

        output, features = model(images)
        output = output.view(-1)
        running_loss = criterion(output, features, labels.float())
        
        loss.update(running_loss.item(), bsz)
        mae.update(output, labels)

        batch_time.update(time.time() - t1)
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Test: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t"
                  f"MAE {mae.avg:.3f}")
    
        t1 = time.time()

    return loss.avg, mae.avg

if __name__ == '__main__':
    opts = parse_arguments()
    
    set_seed(opts.trial)

    train_loader, test_loader_int, test_loader_ext = load_data(opts)
    model, criterion = load_model(opts)
    optimizer = load_optimizer(model, opts)

    model_name = opts.model
    if opts.warm:
        model_name = f"{model_name}_warm"
    

    run_name = (f"{model_name}_{opts.method}_"
                f"{opts.optimizer}_"
                f"tf_{opts.tf}_"
                f"lr{opts.lr}_{opts.lr_decay}_step{opts.lr_decay_step}_rate{opts.lr_decay_rate}_"
                f"wd{opts.weight_decay}_"
                f"trainall_{opts.train_all}_"
                f"bsz{opts.batch_size}_"
                f"trial{opts.trial}")
    tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)
    save_dir = os.path.join(opts.save_dir, f"openbhb_models", run_name)
    ensure_dir(tb_dir)
    ensure_dir(save_dir)

    opts.model_class = model.__class__.__name__
    opts.criterion = opts.method
    opts.optimizer_class = optimizer.__class__.__name__

    wandb.init(project="brain-age-prediction", config=opts, name=run_name, sync_tensorboard=True, tags=['to test'])
    print('Config:', opts)
    print('Model:', model.__class__.__name__)
    print('Criterion:', opts.criterion)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)

    writer = torch.utils.tensorboard.writer.SummaryWriter(tb_dir)
    if opts.amp:
        print("Using AMP")
    
    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss_train, mae_train, batch_time, data_time = train(train_loader, model, criterion, optimizer, opts, epoch)
        t2 = time.time()
        writer.add_scalar("train/loss", loss_train, epoch)
        # writer.add_scalar("train/mae", mae_train, epoch)

        loss_test, mae_int = test(test_loader_int, model, criterion, opts, epoch)
        writer.add_scalar("test/loss_int", loss_test, epoch)
        # writer.add_scalar("test/mae_int", mae_int, epoch)

        loss_test, mae_ext = test(test_loader_ext, model, criterion, opts, epoch)
        writer.add_scalar("test/loss_ext", loss_test, epoch)
        # writer.add_scalar("test/mae_ext", mae_ext, epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        print(f"epoch {epoch}, total time {t2-start_time:.2f}, epoch time {t2-t1:.3f} loss {loss_test:.4f} "
              f"mae_int {mae_int:.3f} mae_ext {mae_ext:.3f}")

        if epoch % opts.save_freq == 0:
            # save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
            # save_model(model, optimizer, opts, epoch, save_file)
            mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader, test_loader_int, test_loader_ext, opts)
            
            writer.add_scalar("train/mae", mae_train, epoch)
            writer.add_scalar("test/mae_int", mae_int, epoch)
            writer.add_scalar("test/mae_ext", mae_ext, epoch)
            print("Age MAE:", mae_train, mae_int, mae_ext)

            ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader, test_loader_int, test_loader_ext, opts)
            writer.add_scalar("train/site_ba", ba_train, epoch)
            writer.add_scalar("test/ba_int", ba_int, epoch)
            writer.add_scalar("test/ba_ext", ba_ext, epoch)
            print("Site BA:", ba_train, ba_int, ba_ext)

            challenge_metric = ba_int**0.3 * mae_ext
            writer.add_scalar("test/score", challenge_metric, epoch)
            print("Challenge score", challenge_metric)

        save_file = os.path.join(save_dir, f"weights.pth")
        save_model(model, optimizer, opts, epoch, save_file)
    
    mae_train, mae_int, mae_ext = compute_age_mae(model, train_loader, test_loader_int, test_loader_ext, opts)
    writer.add_scalar("train/mae", mae_train, epoch)
    writer.add_scalar("test/mae_int", mae_int, epoch)
    writer.add_scalar("test/mae_ext", mae_ext, epoch)
    print("Age MAE:", mae_train, mae_int, mae_ext)

    ba_train, ba_int, ba_ext = compute_site_ba(model, train_loader, test_loader_int, test_loader_ext, opts)
    writer.add_scalar("train/site_ba", ba_train, epoch)
    writer.add_scalar("test/ba_int", ba_int, epoch)
    writer.add_scalar("test/ba_ext", ba_ext, epoch)
    print("Site BA:", ba_train, ba_int, ba_ext)

    challenge_metric = ba_int**0.3 * mae_ext
    writer.add_scalar("test/score", challenge_metric, epoch)
    print("Challenge score", challenge_metric)