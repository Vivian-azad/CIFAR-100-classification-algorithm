"""
CIFAR-100 分类器训练脚本  v1

date:2026-10-14
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# CIFAR-100 统计量（训练集上算得，不要用 ImageNet 的）
# ---------------------------------------------------------------------------
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
NUM_CLASSES = 100


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.6
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# 数据：分层切分 + 防泄露校验
# ---------------------------------------------------------------------------
def stratified_split(targets, n_train_per_class: int, n_val_per_class: int, seed: int):
    """按类别分层抽取。返回 train_idx, val_idx, pool_idx（三者互不相交）。"""
    targets = np.asarray(targets)
    classes = np.unique(targets)
    rng = np.random.RandomState(seed)  # 专用 RNG，与训练随机性解耦，保证切分恒定
    train_idx, val_idx, pool_idx = [], [], []
    for c in classes:
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        need = n_train_per_class + n_val_per_class
        if len(idx) < need:
            raise ValueError(
                f"类别 {c} 只有 {len(idx)} 张图，不足以切出 {need} 张，请调小 --train-per-class / --val-per-class"
            )
        val_idx.extend(idx[:n_val_per_class].tolist())
        train_idx.extend(idx[n_val_per_class:n_val_per_class + n_train_per_class].tolist())
        pool_idx.extend(idx[need:].tolist())
    assert len(set(train_idx) & set(val_idx)) == 0, "切分出错：train 与 val 有交集！"
    assert len(set(train_idx) & set(pool_idx)) == 0, "切分出错：train 与 pool 有交集！"
    assert len(set(val_idx) & set(pool_idx)) == 0, "切分出错：val 与 pool 有交集！"
    return train_idx, val_idx, pool_idx


class IndexedCIFAR(Dataset):
    """在 CIFAR100 基础数据集上按索引取子集，并可独立指定 transform。"""

    def __init__(self, base, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = Image.fromarray(self.base.data[j])
        y = int(self.base.targets[j])
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_train_transform(aug: str, erase: float, input_size: int = 32):
    ops = []
    if input_size != 32:
        ops.append(transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC))
    ops += [
        transforms.RandomCrop(input_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if aug in ("trivial", "rand"):
        try:
            if aug == "trivial":
                ops.append(transforms.TrivialAugmentWide())
            else:
                ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        except Exception as e:  # torchvision 版本过旧
            print(f"[WARN] {aug} 不可用（{e}），回退到 basic 增强")
            aug = "basic"
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    if erase > 0:
        ops.append(
            transforms.RandomErasing(p=erase, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
        )
    return transforms.Compose(ops), aug


def build_eval_transform(input_size: int = 32):
    ops = []
    if input_size != 32:
        ops.append(transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC))
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(name: str, pretrained: str, num_classes: int = NUM_CLASSES,
                stem: str = "auto", input_size: int = 32):
    table = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "wideresnet50": models.wide_resnet50_2,
    }
    if name not in table:
        raise ValueError(f"未知模型 {name}，可选：{list(table)}")
    weights = None
    if pretrained == "imagenet":
        try:
            weights = "DEFAULT"
            model = table[name](weights=weights)
        except Exception:  # 老版本 torchvision
            model = table[name](pretrained=True)
    else:
        model = table[name](weights=None)

    # --- stem 决策 ---
    # cifar : 3x3 stride1 + 去 maxpool，适配 32x32（从零训练必选）
    # imagenet: 保留 7x7 stride2 + maxpool，只在输入足够大（>=96）时才有意义
    # auto  : 预训练且输入>=96 时保留 ImageNet stem，否则一律用 CIFAR stem
    if stem == "auto":
        stem = "imagenet" if (pretrained == "imagenet" and input_size >= 96) else "cifar"

    if stem == "cifar":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        conv1_reinit = True
    else:
        conv1_reinit = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if pretrained == "imagenet":
        # conv1 形状若被改动，ImageNet 权重无法复用，重新初始化；fc 一律重新初始化
        if conv1_reinit:
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(model.fc.weight, 0, 0.01)
        nn.init.zeros_(model.fc.bias)
    return model


# ---------------------------------------------------------------------------
# MixUp / CutMix
# ---------------------------------------------------------------------------
def rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = int(torch.empty(1).random_(0, W).item())
    cy = int(torch.empty(1).random_(0, H).item())
    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, y1, x2, y2


def apply_mix(images, targets, mixup_alpha, cutmix_alpha, num_classes):
    """返回 (images, y_a, y_b, lam)。未启用时 lam=1 且 y_b==y_a。"""
    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if not (use_mixup or use_cutmix):
        return images, targets, targets, 1.0

    r = random.random()
    if use_mixup and use_cutmix:
        mode = "mixup" if r < 0.5 else "cutmix"
    elif use_mixup:
        mode = "mixup"
    else:
        mode = "cutmix"

    alpha = mixup_alpha if mode == "mixup" else cutmix_alpha
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)  # 保证主样本占多数，稳定统计
    perm = torch.randperm(images.size(0), device=images.device)

    if mode == "mixup":
        images = lam * images + (1.0 - lam) * images[perm]
        return images, targets, targets[perm], lam

    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1) / float(images.size(2) * images.size(3))
    lam_adj = 1.0 - area
    return images, targets, targets[perm], lam_adj


def mix_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, collect=False, num_classes=NUM_CLASSES):
    model.eval()
    total_loss, total_correct1, total_correct5, total_n = 0.0, 0.0, 0.0, 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_count = np.zeros(num_classes, dtype=np.int64)
    all_true, all_pred, all_prob = [], [], []
    all_probs = []

    autocast_ok = device.type == "cuda"
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if autocast_ok:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        n = targets.size(0)
        total_loss += loss.item() * n
        total_n += n
        top1, top5 = accuracy_topk(logits.float(), targets, topk=(1, 5))
        total_correct1 += top1 * n / 100.0
        total_correct5 += top5 * n / 100.0

        if collect:
            probs = F.softmax(logits.float(), dim=1)
            conf, pred = probs.max(dim=1)
            yt = targets.cpu().numpy()
            yp = pred.cpu().numpy()
            np.add.at(per_class_correct, yt, (yt == yp).astype(np.int64))
            np.add.at(per_class_count, yt, 1)
            all_true.append(yt)
            all_pred.append(yp)
            all_prob.append(conf.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    res = {
        "loss": total_loss / max(1, total_n),
        "top1": 100.0 * total_correct1 / max(1, total_n),
        "top5": 100.0 * total_correct5 / max(1, total_n),
        "n": total_n,
    }
    if collect:
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        conf = np.concatenate(all_prob)
        probs = np.concatenate(all_probs)
        cls_acc = np.divide(
            per_class_correct, per_class_count, out=np.zeros(num_classes), where=per_class_count > 0
        )
        correct_mask = y_true == y_pred
        # 预测分布诊断
        counts = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
        share = counts / counts.sum()
        order = np.argsort(-share)
        p_safe = np.clip(probs, 1e-12, 1.0)
        res["diag"] = {
            "cls_acc_mean": float(cls_acc.mean()),
            "cls_acc_std": float(cls_acc.std()),
            "cls_acc_min": float(cls_acc.min()),
            "cls_acc_max": float(cls_acc.max()),
            "cls_acc_worst5": [
                [int(i), float(cls_acc[i])] for i in np.argsort(cls_acc)[:5]
            ],
            "cls_acc_best5": [
                [int(i), float(cls_acc[i])] for i in np.argsort(-cls_acc)[:5]
            ],
            "mean_entropy": float(-(p_safe * np.log(p_safe)).sum(1).mean()),
            "pred_max_class_share": float(share.max()),          # 均匀=1%，>10% 视为坍塌
            "pred_top10_coverage": float(share[order[:10]].sum()),
            "never_predicted_classes": int((counts == 0).sum()),
            "conf_correct": float(conf[correct_mask].mean()) if correct_mask.any() else float("nan"),
            "conf_wrong": float(conf[~correct_mask].mean()) if (~correct_mask).any() else float("nan"),
            "per_class_acc": cls_acc.tolist(),
        }
    return res


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)

    def apply(self, model):
        msd = model.state_dict()
        self.backup = {k: msd[k].detach().clone() for k in self.shadow}
        msd.update({k: v.clone() for k, v in self.shadow.items()})
        model.load_state_dict(msd)

    def restore(self, model):
        msd = model.state_dict()
        msd.update(self.backup)
        model.load_state_dict(msd)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("CIFAR-100 训练器 v1")
    # 数据
    p.add_argument("--data", default="./data", help="CIFAR-100 存放/下载目录")
    p.add_argument("--out", default="./runs", help="输出根目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-per-class", type=int, default=200, help="每类训练图数（默认 200*100=20000）")
    p.add_argument("--val-per-class", type=int, default=50, help="每类内部验证图数（默认 50*100=5000）")
    p.add_argument("--eval-train-clean", type=int, default=2000, help="无增强训练集上评估多少张，用于算真实泛化间隙")
    # 模型
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "wideresnet50"])
    p.add_argument("--pretrained", default="none", choices=["none", "imagenet"])
    p.add_argument("--stem", default="auto", choices=["auto", "cifar", "imagenet"])
    p.add_argument("--input-size", type=int, default=32, help="输入边长；>32 会先做 Resize（更慢，预训练撞瓶颈时的备用杠杆）")
    # 一键配方（显式传入的 flag 优先于 preset）
    p.add_argument("--preset", default="none", choices=["none", "scratch_v1", "finetune_v1"])
    p.add_argument("--smoke", action="store_true", help="冒烟测试：极小数据跑 2 个 epoch，验证链路")
    # 优化
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=5.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=0.0, help=">0 时开启梯度裁剪")
    # 正则 / 增强
    p.add_argument("--aug", default="basic", choices=["basic", "trivial", "rand"])
    p.add_argument("--erase", type=float, default=0.0, help="RandomErasing 概率")
    p.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha，0=关闭")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha，0=关闭")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay，0=关闭")
    p.add_argument("--dropout", type=float, default=0.0)
    # 系统
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tag", default="", help="附加到 run 名的备注")
    p.add_argument("--print-every", type=int, default=20, help="终端每隔多少 epoch 打印一行")
    return p


# 一键配方：避免手工配错 lr/epochs 导致整轮白跑
PRESETS = {
    # 从零训练：大 lr + 长周期 + 强权重衰减
    "scratch_v1": dict(model="resnet18", pretrained="none", epochs=200, bs=128, lr=0.1,
                       wd=5e-4, warmup_epochs=5.0, aug="basic", label_smoothing=0.1),
    # 预训练微调：小 lr（大了会把 ImageNet 特征冲掉）+ 短周期 + 弱权重衰减
    "finetune_v1": dict(model="resnet50", pretrained="imagenet", epochs=100, bs=128, lr=0.01,
                        wd=1e-4, warmup_epochs=2.0, aug="trivial", label_smoothing=0.1),
}


def apply_preset(p, args):
    """preset 只覆盖仍停留在 argparse 默认值的参数，用户显式传入的 flag 优先。"""
    if args.preset == "none":
        return args
    base = p.parse_args([])
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) == getattr(base, k):
            setattr(args, k, v)
    return args


def apply_smoke(args):
    if not args.smoke:
        return args
    args.train_per_class = 20
    args.val_per_class = 5
    args.eval_train_clean = 100
    args.epochs = 2
    args.workers = 0
    args.print_every = 1
    args.tag = (args.tag + "_smoke").strip("_")
    return args


def main():
    p = get_args()
    args = apply_smoke(apply_preset(p, p.parse_args()))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"

    # ---------------- 数据 ----------------
    # 注意：只加载 train=True，官方测试集在本脚本中从头到尾不存在。
    base = datasets.CIFAR100(root=args.data, train=True, download=True)
    class_names = list(base.classes)

    train_idx, val_idx, pool_idx = stratified_split(
        base.targets, args.train_per_class, args.val_per_class, seed=args.seed
    )

    train_tf, aug_used = build_train_transform(args.aug, args.erase, args.input_size)
    eval_tf = build_eval_transform(args.input_size)

    ds_train = IndexedCIFAR(base, train_idx, train_tf)
    ds_val = IndexedCIFAR(base, val_idx, eval_tf)
    ds_pool = IndexedCIFAR(base, pool_idx, eval_tf)

    # 无增强的训练子集 —— 用来算真实泛化间隙（关键诊断量）
    rng = np.random.RandomState(args.seed + 1)
    clean_idx = list(np.asarray(train_idx)[rng.permutation(len(train_idx))[: args.eval_train_clean]])
    ds_clean = IndexedCIFAR(base, clean_idx, eval_tf)

    pin = device.type == "cuda"
    ld_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    ld_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
    ld_clean = DataLoader(ds_clean, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
    ld_pool = DataLoader(ds_pool, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)

    # ---------------- 模型 ----------------
    model = build_model(args.model, args.pretrained, stem=args.stem, input_size=args.input_size).to(device)
    if args.dropout > 0 and hasattr(model, "fc"):
        model.fc = nn.Sequential(nn.Dropout(args.dropout), model.fc)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.wd, nesterov=True)

    iters_per_epoch = len(ld_train)
    total_steps = args.epochs * iters_per_epoch
    warmup_steps = int(args.warmup_epochs * iters_per_epoch)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=False)

    ema = EMA(model, args.ema) if args.ema > 0 else None

    # ---------------- 输出目录 ----------------
    run_name = f"{args.model}_{'pt' if args.pretrained == 'imagenet' else 'scratch'}_{aug_used}" \
               f"_e{args.epochs}_bs{args.bs}_lr{args.lr}_s{args.seed}"
    if args.mixup > 0:
        run_name += f"_mu{args.mixup}"
    if args.cutmix > 0:
        run_name += f"_cm{args.cutmix}"
    if args.ema > 0:
        run_name += f"_ema{args.ema}"
    if args.tag:
        run_name += f"_{args.tag}"
    run_dir = Path(args.out) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(vars(args))
    cfg.update({
        "aug_used": aug_used,
        "stem_used": ("imagenet" if (args.pretrained == "imagenet" and args.input_size >= 96 and args.stem == "auto")
                      else ("cifar" if args.stem == "auto" else args.stem)),
        "n_train": len(ds_train), "n_val": len(ds_val), "n_pool": len(ds_pool),
        "n_clean_eval": len(ds_clean),
        "params": param_count(model),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "python": platform.python_version(), "iters_per_epoch": iters_per_epoch,
        "official_test_used": False,
    })
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"RUN: {run_name}   [preset={args.preset}]")
    print(f"数据：train={len(ds_train)}  val={len(ds_val)}  pool(未用)={len(ds_pool)}  clean-eval={len(ds_clean)}")
    print(f"切分校验：train∩val={len(set(train_idx) & set(val_idx))}  官方测试集：未加载")
    print(f"模型：{args.model} pretrained={args.pretrained} stem={cfg['stem_used']} "
          f"input={args.input_size}  参数量={param_count(model):,}")
    print(f"配方：epochs={args.epochs} bs={args.bs} lr={args.lr} wd={args.wd} aug={aug_used} "
          f"ls={args.label_smoothing} mixup={args.mixup} cutmix={args.cutmix} ema={args.ema}")
    if device.type == "cuda":
        print(f"设备：{torch.cuda.get_device_name(device)}  CUDA={torch.version.cuda}  "
              f"AMP={use_amp}  workers={args.workers}")
    else:
        print(f"设备：CPU（警告：会很慢）  workers={args.workers}")
    print("=" * 78)

    csv_path = run_dir / "metrics.csv"
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)
    header = ["epoch", "lr", "train_loss", "train_acc", "train_clean_loss", "train_clean_acc",
              "val_loss", "val_acc", "val_top5", "gap_clean_val", "grad_norm", "grad_norm_max", "sec"]
    writer.writerow(header)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    t_start = time.time()
    peak_mem = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        run_loss, run_correct, run_n = 0.0, 0.0, 0
        gnorm_sum, gnorm_max, gsteps = 0.0, 0.0, 0
        cur_lr = optimizer.param_groups[0]["lr"]

        for images, targets in ld_train:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            images, y_a, y_b, lam = apply_mix(images, targets, args.mixup, args.cutmix, NUM_CLASSES)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = mix_criterion(criterion, logits, y_a, y_b, lam)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            gv = float(gn.item()) if torch.isfinite(gn) else float("nan")
            gnorm_sum += gv
            gnorm_max = max(gnorm_max, gv)
            gsteps += 1

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            bs = targets.size(0)
            run_loss += loss.item() * bs
            with torch.no_grad():
                pred = logits.float().argmax(1)
                run_correct += (pred == y_a).float().sum().item()
            run_n += bs

        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024 ** 3)

        ep_sec = time.time() - ep_t0
        train_loss = run_loss / max(1, run_n)
        train_acc = 100.0 * run_correct / max(1, run_n)

        # 无增强训练集评估（真实泛化间隙的分子）
        c = evaluate(model, ld_clean, criterion, device)
        v = evaluate(model, ld_val, criterion, device)

        gap = c["top1"] - v["top1"]
        gnorm_mean = gnorm_sum / max(1, gsteps)

        writer.writerow([epoch, f"{cur_lr:.6g}", f"{train_loss:.4f}", f"{train_acc:.2f}",
                         f"{c['loss']:.4f}", f"{c['top1']:.2f}", f"{v['loss']:.4f}", f"{v['top1']:.2f}",
                         f"{v['top5']:.2f}", f"{gap:.2f}", f"{gnorm_mean:.4f}", f"{gnorm_max:.4f}",
                         f"{ep_sec:.1f}"])
        fcsv.flush()

        if v["top1"] > best_val:
            best_val = v["top1"]
            best_epoch = epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"ep{epoch:4d} lr={cur_lr:.5f} | tr_loss={train_loss:.3f} tr_acc={train_acc:5.2f} "
                  f"tr_clean={c['top1']:5.2f} | val_loss={v['loss']:.3f} val={v['top1']:5.2f} "
                  f"top5={v['top5']:5.2f} | gap={gap:6.2f} | gn={gnorm_mean:.3f} | {ep_sec:.1f}s")

    fcsv.close()
    total_time = time.time() - t_start

    # ---------------- 用最优权重做最终诊断 ----------------
    if best_state is not None:
        model.load_state_dict(best_state)
    if ema is not None:
        ema.apply(model)

    v_best = evaluate(model, ld_val, criterion, device, collect=True)
    c_best = evaluate(model, ld_clean, criterion, device)
    p_best = evaluate(model, ld_pool, criterion, device)  # pool 只做参考，不参与任何决策
    d = v_best["diag"]

    summary = {
        "run_name": run_name,
        "best_val_acc": best_val, "best_epoch": best_epoch,
        "final_epoch_val_acc": v_best["top1"],
        "val_top5_at_best": v_best["top5"],
        "val_loss_at_best": v_best["loss"],
        "train_clean_acc_at_best": c_best["top1"],
        "gap_at_best": c_best["top1"] - v_best["top1"],
        "pool_acc_reference_only": p_best["top1"],
        "diag": d,
        "params": param_count(model),
        "peak_gpu_mem_gb": round(peak_mem, 2),
        "total_time_sec": round(total_time, 1),
        "img_per_sec": round((len(ds_train) * args.epochs) / total_time, 1),
        "config": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if best_state is not None:
        torch.save(best_state, run_dir / "best.pt")
    torch.save(model.state_dict(), run_dir / "last.pt")

    # ---------------- 报告块（原样复制粘贴给算法侧） ----------------
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))[1:]
    pick = [r for i, r in enumerate(rows)
            if (i + 1) % args.print_every == 0 or (i + 1) == 1 or (i + 1) >= len(rows) - 5]

    L = []
    L.append("=" * 78)
    L.append("REPORT BLOCK —— 请把本块原样复制给算法侧（小算）")
    L.append("=" * 78)
    L.append(f"run_name        : {run_name}")
    L.append(f"model/pretrained: {args.model} / {args.pretrained}  stem={cfg['stem_used']} "
             f"input={args.input_size}  params={param_count(model):,}")
    L.append(f"data            : train={len(ds_train)} val={len(ds_val)} pool={len(ds_pool)} "
             f"(官方test未使用)")
    L.append(f"recipe          : preset={args.preset} epochs={args.epochs} bs={args.bs} lr={args.lr} "
             f"wd={args.wd} aug={aug_used} ls={args.label_smoothing} mixup={args.mixup} "
             f"cutmix={args.cutmix} ema={args.ema} erase={args.erase} dropout={args.dropout}")
    L.append(f"env             : torch={torch.__version__} tv={torchvision.__version__} "
             f"device={device} amp={use_amp} peak_mem={peak_mem:.2f}GB")
    L.append(f"speed           : total={total_time/60:.1f}min  {summary['img_per_sec']} img/s")
    L.append("-" * 78)
    L.append("epoch  lr       tr_loss tr_acc tr_clean val_loss val_acc val_top5   gap   gradnorm   sec")
    for r in pick:
        L.append(f"{r[0]:>5} {r[1]:>8} {r[2]:>7} {r[3]:>6} {r[5]:>8} {r[6]:>8} {r[7]:>7} {r[8]:>8} "
                 f"{r[9]:>6} {r[10]:>9} {r[12]:>5}")
    L.append("-" * 78)
    L.append(f"best_val_acc    : {best_val:.2f}  (epoch {best_epoch} / {args.epochs})")
    L.append(f"val_top5@best   : {v_best['top5']:.2f}")
    L.append(f"train_clean@best: {c_best['top1']:.2f}   gap@best = {c_best['top1'] - v_best['top1']:.2f} pt")
    L.append(f"pool_acc(参考)  : {p_best['top1']:.2f}")
    L.append("-" * 78)
    L.append(f"类准确率        : mean={d['cls_acc_mean']*100:.2f}% std={d['cls_acc_std']*100:.2f} "
             f"min={d['cls_acc_min']*100:.2f}% max={d['cls_acc_max']*100:.2f}%")
    wc = ", ".join(f"{class_names[i]}={a*100:.0f}%" for i, a in d["cls_acc_worst5"])
    bc = ", ".join(f"{class_names[i]}={a*100:.0f}%" for i, a in d["cls_acc_best5"])
    L.append(f"最差5类         : {wc}")
    L.append(f"最好5类         : {bc}")
    L.append(f"预测分布        : 熵={d['mean_entropy']:.3f} 最大类占比={d['pred_max_class_share']*100:.2f}% "
             f"top10覆盖={d['pred_top10_coverage']*100:.1f}% 从未预测的类数={d['never_predicted_classes']}")
    L.append(f"置信度          : 正确样本={d['conf_correct']:.3f}  错误样本={d['conf_wrong']:.3f}")
    L.append("=" * 78)
    L.append(f"完整逐 epoch 日志: {csv_path}")
    L.append(f"完整诊断 json    : {run_dir / 'summary.json'}")
    L.append("=" * 78)

    report = "\n".join(L)
    print(report)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
