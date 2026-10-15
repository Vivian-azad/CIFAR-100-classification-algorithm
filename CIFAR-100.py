"""
CIFAR-100 分类器训练脚本  v3
date：2026-10-15

相比于上一版，优化的地方：
1.微调用小 lr（0.01）+ 弱 wd（1e-4）+ 短周期（100 epoch）+ 2 epoch warmup，避免大 lr 冲掉 ImageNet 特征。
2.EMA（指数移动平均）​
已实现且独立追踪：单独评估、单独选最优 epoch、单独存 best_ema.pt，绝不混进 best_val_acc 的口径——保证跨轮可比。
3.RandomErasing / RandAugment 强增强
"""


from __future__ import annotations

import argparse
import csv
import json
import math
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
# [§0] 基础工具
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def get_grad_norm(parameters):
    """只读的全局梯度 L2 范数，**不修改梯度**。

    与 clip_grad_norm_ 的职责分开：那个函数是用来"裁剪"的，
    拿它（配合 inf 上限）当"测量工具"用，在非有限梯度时会把
    裁剪系数算成 nan 再乘回梯度，诊断污染处理，很脆。
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.zeros(())
    norms = torch.stack([torch.norm(p.grad.detach(), 2.0) for p in params])
    return torch.norm(norms, 2.0)


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """topk 准确率。correct 形状 [maxk, N]，correct[:k] 取"前 k 个预测"的行。

    注意：reshape(-1).sum() 统计的是"真实类别是否出现在前 k 个预测里"
    （只要命中一次就算对），这正是 top-k 的定义 —— 不是把 k 行分别求和。
    """
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


# ---------------------------------------------------------------------------
# [§1] 数据：分层切分 + 防泄露校验
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
# [§2] 模型
# ---------------------------------------------------------------------------
def build_model(name: str, pretrained: str, num_classes: int = NUM_CLASSES,
                stem: str = "auto", input_size: int = 32):
    """返回 (model, stem_used)。stem 决策只在这里发生一次，调用方不要重算。"""
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
        except TypeError:
            # 只有"老版本 torchvision 不认识 weights 参数"才回退旧 API；
            # 别的异常（如下载失败、网络错误）必须原样抛出，不能吞掉
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
    return model, stem


# ---------------------------------------------------------------------------
# [§3] MixUp / CutMix
# ---------------------------------------------------------------------------
def rand_bbox(size, lam):
    """标准 CutMix 裁剪框。纯 python int 运算：这步跑在 CPU 主进程，
    别为了取随机数去分配 tensor（会打断 CUDA 异步流水线）。"""
    _, _, H, W = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)
    x1 = max(0, min(cx - cut_w // 2, W))
    y1 = max(0, min(cy - cut_h // 2, H))
    x2 = max(0, min(cx + cut_w // 2, W))
    y2 = max(0, min(cy + cut_h // 2, H))
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
    # 注意：不要对 lam 做 max(lam, 1-lam) 之类的钳制。
    # 那会把 CutMix 的裁剪面积限制在 0~50%（标准应为 0~100%，均值 50%），
    # 相当于把 CutMix 强度砍半。精度统计用 y_a 只是诊断近似，不影响训练信号。
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
# [§4] 评估与诊断
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, collect=False, num_classes=NUM_CLASSES):
    """collect=True 时额外给出类级/分布级诊断。

    存储口径：只在 collect=True 时缓存，且只缓存一份完整 softmax 概率
    （5000x100 float32 = 2MB）。置信度 = probs.max(1)、预测 = probs.argmax(1)，
    都由这一份推出来，不重复存。非 collect 调用全程零额外内存。
    """
    model.eval()
    total_loss, total_correct1, total_correct5, total_n = 0.0, 0.0, 0.0, 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_count = np.zeros(num_classes, dtype=np.int64)
    all_true, all_probs = [], []

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
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            yt = targets.cpu().numpy()
            yp = probs.argmax(axis=1)
            np.add.at(per_class_correct, yt, (yt == yp).astype(np.int64))
            np.add.at(per_class_count, yt, 1)
            all_true.append(yt)
            all_probs.append(probs)

    res = {
        "loss": total_loss / max(1, total_n),
        "top1": 100.0 * total_correct1 / max(1, total_n),
        "top5": 100.0 * total_correct5 / max(1, total_n),
        "n": total_n,
    }
    if collect:
        y_true = np.concatenate(all_true)
        probs = np.concatenate(all_probs)
        y_pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
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
# [§5] EMA
# ---------------------------------------------------------------------------
class EMA:
    """权重滑动平均。

    口径：只跟踪浮点张量，因此包含 BN 的 running_mean / running_var
    （与 timm 的 ModelEma 一致），不含 num_batches_tracked 这类整型 buffer。
    若日后发现 EMA 权重下的 BN 统计漂移，再改成排除 buffer + 评估前重算 BN。
    apply/restore 自己管好 training flag，调用方不必操心。
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}
        self.was_training = True

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)

    def apply(self, model):
        self.was_training = model.training
        msd = model.state_dict()
        self.backup = {k: msd[k].detach().clone() for k in self.shadow}
        msd.update({k: v.clone() for k, v in self.shadow.items()})
        model.load_state_dict(msd)
        model.eval()

    def restore(self, model):
        msd = model.state_dict()
        msd.update(self.backup)
        model.load_state_dict(msd)
        self.backup = {}
        if self.was_training:
            model.train()


# ---------------------------------------------------------------------------
# [§6] 参数解析与配方
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
    # 差异化学习率：预训练骨干 vs 新初始化层（分类头 / 被替换的 conv1）
    p.add_argument("--backbone-lr-mult", type=float, default=1.0, help="预训练骨干的学习率倍率")
    p.add_argument("--head-lr-mult", type=float, default=1.0, help="分类头 + 新 conv1 的学习率倍率")
    # 系统
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--workers", type=int, default=2, help="数据加载进程数；吃 CPU/内存不吃显存，跑不动就调小")
    p.add_argument("--tag", default="", help="附加到 run 名的备注")
    p.add_argument("--print-every", type=int, default=20, help="终端每隔多少 epoch 打印一行")
    p.add_argument("--deterministic", action="store_true",
                   help="严格可复现：关 cudnn.benchmark、开确定性算法。会明显变慢，只在最终验收轮用")
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


def _explicit_dests(argv):
    """返回用户在命令行上"真的写过"的 dest 集合。

    做法：复制一份 parser，把每个参数的 default 换成 argparse.SUPPRESS，
    再解析同一份 argv —— 得到的 namespace 只含显式传入的键。
    比"拿默认值反推"可靠：--aug basic 这种"显式传入且恰好等于默认值"
    的情况，旧写法会漏判、被 preset 静默覆盖。
    """
    shadow = get_args()
    for a in shadow._actions:
        a.default = argparse.SUPPRESS
    ns = shadow.parse_args(argv)  # 主 parse 已成功过，这里严格解析即可（argv 必合法）
    return set(vars(ns))


def resolve_args(argv=None):
    """一次性把 args 定下来：解析 -> 记录显式项 -> 套 preset -> 套 smoke。

    返回 (args, explicit)。explicit 会写进 config.json，便于复盘某轮到底
    哪些参数是我给的、哪些是同事手改的。
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    p = get_args()
    args = p.parse_args(argv)
    explicit = _explicit_dests(argv)

    if args.preset != "none":
        for k, v in PRESETS[args.preset].items():
            if k not in explicit:      # 只填空，不抢显式传入的 flag
                setattr(args, k, v)

    if args.smoke:                      # smoke 是调试模式，无条件覆盖 preset
        args.train_per_class = 20
        args.val_per_class = 5
        args.eval_train_clean = 100
        args.epochs = 2
        args.workers = 0
        args.print_every = 1
        args.tag = (args.tag + "_smoke").strip("_")

    _validate_args(args)
    return args, explicit


def _validate_args(args):
    """启动前把明显非法的参数拦下来，别等跑到一半才崩。"""
    checks = [
        (args.epochs >= 1, "epochs 必须 >= 1"),
        (args.bs >= 1, "bs 必须 >= 1"),
        (args.lr > 0, "lr 必须 > 0"),
        (args.wd >= 0, "wd 必须 >= 0"),
        (0 <= args.label_smoothing < 1, "label_smoothing 必须在 [0, 1)"),
        (0 <= args.momentum < 1, "momentum 必须在 [0, 1)"),
        (args.train_per_class >= 1, "train-per-class 必须 >= 1"),
        (args.val_per_class >= 1, "val-per-class 必须 >= 1"),
        (args.train_per_class + args.val_per_class <= 500, "train+val 每类不能超过 500 张"),
        (0 <= args.ema < 1, "ema decay 必须在 [0, 1)，0 表示关闭"),
        (0 <= args.dropout < 1, "dropout 必须在 [0, 1)"),
        (args.input_size >= 16, "input-size 不能小于 16"),
        (args.warmup_epochs >= 0, "warmup-epochs 必须 >= 0"),
        (args.backbone_lr_mult > 0 and args.head_lr_mult > 0, "lr 倍率必须 > 0"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"非法参数：{msg}（当前配置见 --help 或 config.json）")


def main():
    args, explicit = resolve_args()
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"
    mixing_on = (args.mixup > 0) or (args.cutmix > 0)

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
    # pool 只是"官方训练集里我们没用上的那部分"，全程不参与训练与任何决策。
    # 只有在它非空时才建 DataLoader（放量到 450/类 后它会是空的）。
    ds_pool = IndexedCIFAR(base, pool_idx, eval_tf) if len(pool_idx) > 0 else None

    # 无增强的训练子集 —— 用来算真实泛化间隙（关键诊断量）
    rng = np.random.RandomState(args.seed + 1)
    clean_idx = list(np.asarray(train_idx)[rng.permutation(len(train_idx))[: args.eval_train_clean]])
    ds_clean = IndexedCIFAR(base, clean_idx, eval_tf)

    pin = device.type == "cuda"
    ld_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                          pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    ld_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
    ld_clean = DataLoader(ds_clean, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
    ld_pool = (DataLoader(ds_pool, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=pin)
               if ds_pool is not None else None)

    # ---------------- 模型 ----------------
    model, stem_used = build_model(args.model, args.pretrained, stem=args.stem, input_size=args.input_size)
    model = model.to(device)
    if args.dropout > 0 and hasattr(model, "fc"):
        model.fc = nn.Sequential(nn.Dropout(args.dropout), model.fc)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 差异化学习率：新初始化的层（分类头；stem 被改成 cifar 时还包括 conv1）单独一组
    head_ids = set(id(p) for p in model.fc.parameters())
    if stem_used == "cifar":
        head_ids |= set(id(p) for p in model.conv1.parameters())
    p_head, p_back = [], []
    for _, prm in model.named_parameters():
        (p_head if id(prm) in head_ids else p_back).append(prm)
    groups = []
    if p_back:
        groups.append({"params": p_back, "lr": args.lr * args.backbone_lr_mult})
    if p_head:
        groups.append({"params": p_head, "lr": args.lr * args.head_lr_mult})
    optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum,
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

    # 梯度诊断/裁剪只作用于 optimizer 管理的参数。
    # 注意：scaler.unscale_ 只还原 optimizer 里的 param；若将来加入不在
    # optimizer 中的 frozen 参数，其 .grad 仍是放大过的，不能直接进范数统计。
    opt_params = [p for g in optimizer.param_groups for p in g["params"]]

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
        "stem_used": stem_used,
        "lr_backbone": args.lr * args.backbone_lr_mult,
        "lr_head": args.lr * args.head_lr_mult,
        "n_train": len(ds_train), "n_val": len(ds_val),
        "n_pool": (len(ds_pool) if ds_pool is not None else 0),
        "n_clean_eval": len(ds_clean),
        "explicit_args": sorted(explicit),
        "params": param_count(model),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "python": platform.python_version(), "iters_per_epoch": iters_per_epoch,
        "cudnn_benchmark": (not args.deterministic),
        "official_test_used": False,
    })
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"RUN: {run_name}   [preset={args.preset}]")
    print(f"数据：train={len(ds_train)}  val={len(ds_val)}  "
          f"pool(未用)={len(ds_pool) if ds_pool is not None else 0}  clean-eval={len(ds_clean)}")
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
    header = ["epoch", "lr_backbone", "lr_head", "train_loss", "train_acc_mixed",
              "train_clean_loss", "train_clean_acc", "val_loss", "val_acc", "val_top5",
              "gap_clean_val", "grad_norm", "grad_norm_max", "grad_nonfinite_steps",
              "opt_skipped_steps", "sec", "val_acc_ema"]
    writer.writerow(header)
    col = {name: i for i, name in enumerate(header)}  # 报告块按列名取，别按死索引

    best_val = -1.0
    best_epoch = -1
    best_state = None
    best_val_ema = -1.0
    best_epoch_ema = -1
    best_state_ema = None
    t_start = time.time()
    peak_mem = 0.0
    total_bad_grad = 0
    total_skip = 0
    last_val_acc = None  # 显式记录"最后一轮的 val"，不依赖循环残留变量

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        run_loss, run_correct, run_n = 0.0, 0.0, 0
        gnorm_sum, gnorm_max, gsteps, gsteps_bad, gsteps_skip = 0.0, 0.0, 0, 0, 0
        # 分组 lr 都记下来：开了差异化 lr 之后只报 group0 会误导
        lr_back = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[-1]["lr"]

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
                gn = torch.nn.utils.clip_grad_norm_(opt_params, args.clip_grad)
            else:
                gn = get_grad_norm(opt_params)  # 只读，诊断与梯度处理彻底分离
            gv = float(gn.item())
            if math.isfinite(gv):
                gnorm_sum += gv
                gnorm_max = max(gnorm_max, gv)
                gsteps += 1
            else:
                gsteps_bad += 1

            # 用公开 API 判定本步是否被 AMP 跳过。GradScaler 的行为是：
            # 发现非有限梯度 → update() 里 scale 减半（且增长计数清零）；
            # 连续正常步满 growth_interval 才 ×2。**同一步不可能既减半又翻倍**，
            # 所以 "scale 变小 ⟺ 本步被跳过" 是精确等价，不是近似。
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            else:
                gsteps_skip += 1

            if ema is not None:
                ema.update(model)

            bs = targets.size(0)
            run_loss += loss.item() * bs
            with torch.no_grad():
                pred = logits.float().argmax(1)
                # 混合样本按 lam 加权命中（mixup/cutmix 论文的标准口径）：
                # 衡量的是"对混合训练目标的拟合程度"，不是真实分类准确率。
                # 未开混合时 lam=1 且 y_a==y_b，退化为普通 train_acc。
                run_correct += (lam * (pred == y_a).float()
                                + (1.0 - lam) * (pred == y_b).float()).sum().item()
            run_n += bs

        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024 ** 3)

        ep_sec = time.time() - ep_t0
        train_loss = run_loss / max(1, run_n)
        train_acc = 100.0 * run_correct / max(1, run_n)

        # 无增强训练集评估（真实泛化间隙的分子）
        c = evaluate(model, ld_clean, criterion, device)
        v = evaluate(model, ld_val, criterion, device)

        # EMA 权重单独评估：EMA 是与原始权重并列的一条候选模型，
        # 必须独立追踪它自己的最优 epoch，否则报告里的 best_epoch 和最终评估的权重不一致
        v_ema = None
        if ema is not None:
            ema.apply(model)
            v_ema = evaluate(model, ld_val, criterion, device)
            # 只在刷新最优时才 clone 权重（~100MB），不是每 epoch 都拷一份
            if v_ema["top1"] > best_val_ema:
                best_val_ema = v_ema["top1"]
                best_epoch_ema = epoch
                best_state_ema = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            ema.restore(model)

        gap = c["top1"] - v["top1"]
        gnorm_mean = gnorm_sum / max(1, gsteps)
        total_bad_grad += gsteps_bad
        total_skip += gsteps_skip
        last_val_acc = v["top1"]  # 每轮覆盖，出循环即为最后一轮（原始权重口径）

        writer.writerow([epoch, f"{lr_back:.6g}", f"{lr_head:.6g}", f"{train_loss:.4f}",
                         f"{train_acc:.2f}", f"{c['loss']:.4f}", f"{c['top1']:.2f}",
                         f"{v['loss']:.4f}", f"{v['top1']:.2f}", f"{v['top5']:.2f}", f"{gap:.2f}",
                         f"{gnorm_mean:.4f}", f"{gnorm_max:.4f}", gsteps_bad, gsteps_skip,
                         f"{ep_sec:.1f}",
                         ("" if v_ema is None else f"{v_ema['top1']:.2f}")])
        fcsv.flush()

        if v["top1"] > best_val:
            best_val = v["top1"]
            best_epoch = epoch
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            tag_tr = "tr_acc*" if mixing_on else "tr_acc "
            print(f"ep{epoch:4d} lr={lr_head:.5f} | tr_loss={train_loss:.3f} {tag_tr}={train_acc:5.2f} "
                  f"tr_clean={c['top1']:5.2f} | val_loss={v['loss']:.3f} val={v['top1']:5.2f} "
                  f"top5={v['top5']:5.2f} | gap={gap:6.2f} | gn={gnorm_mean:.3f} | {ep_sec:.1f}s")

    fcsv.close()
    total_time = time.time() - t_start

    # ---------------- 用最优权重做最终诊断 ----------------
    def final_diag(state):
        model.load_state_dict(state)
        vv = evaluate(model, ld_val, criterion, device, collect=True)
        cc = evaluate(model, ld_clean, criterion, device)
        pp = None
        if ld_pool is not None:
            pp = evaluate(model, ld_pool, criterion, device)  # pool 只做参考，不参与任何决策
        return vv, cc, pp

    v_best, c_best, p_best = final_diag(best_state)
    d = v_best["diag"]

    # EMA 权重独立成一份诊断：绝不把 EMA 结果混进 best_val_acc，
    # 否则跨轮次（有的开 EMA 有的不开）就没法横向比较
    ema_block = None
    if ema is not None and best_state_ema is not None:
        ve, ce, _ = final_diag(best_state_ema)
        ema_block = {
            "best_val_acc_ema": best_val_ema, "best_epoch_ema": best_epoch_ema,
            "val_top5_ema": ve["top5"], "val_loss_ema": ve["loss"],
            "train_clean_acc_ema": ce["top1"], "gap_ema": ce["top1"] - ve["top1"],
            "diag_ema": ve["diag"],
        }

    summary = {
        "run_name": run_name,
        "best_val_acc": best_val, "best_epoch": best_epoch,
        # 两个值别混：一个是"最优权重复查"，一个是"训练最后一轮"（均为原始权重口径；
        # EMA 口径的最优/最终见 ema_block，绝不混入）
        "val_acc_at_best_recheck": v_best["top1"],   # 重载 best_state 后的复核值
        "final_epoch_val_acc": last_val_acc,         # 训练最后一轮的 val
        "val_top5_at_best": v_best["top5"],
        "val_loss_at_best": v_best["loss"],
        "train_clean_acc_at_best": c_best["top1"],
        "gap_at_best": c_best["top1"] - v_best["top1"],
        "pool_acc_reference_only": (p_best["top1"] if p_best is not None else None),
        "ema": ema_block,
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
    if best_state_ema is not None:
        torch.save(best_state_ema, run_dir / "best_ema.pt")

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
    L.append(f"data            : train={len(ds_train)} val={len(ds_val)} "
             f"pool={len(ds_pool) if ds_pool is not None else 0} (官方test未使用)")
    L.append(f"recipe          : preset={args.preset} epochs={args.epochs} bs={args.bs} "
             f"lr_backbone={cfg['lr_backbone']:g} lr_head={cfg['lr_head']:g} wd={args.wd} "
             f"aug={aug_used} ls={args.label_smoothing} mixup={args.mixup} "
             f"cutmix={args.cutmix} ema={args.ema} erase={args.erase} dropout={args.dropout}")
    L.append(f"explicit_args   : {','.join(sorted(explicit)) or '(无，全部走 preset/默认)'}")
    L.append(f"env             : torch={torch.__version__} tv={torchvision.__version__} "
             f"device={device} amp={use_amp} peak_mem={peak_mem:.2f}GB")
    L.append(f"speed           : total={total_time/60:.1f}min  {summary['img_per_sec']} img/s")
    L.append("-" * 78)
    L.append("epoch  lr_head  tr_loss tr_acc* tr_clean val_loss val_acc val_top5   gap   gradnorm   sec")
    if mixing_on:
        L.append("(*) 开启混合增强后 tr_acc 是混合目标的 lam 加权命中率"
                 "（衡量对训练目标的拟合，≠ 真实分类准确率）；泛化间隙看 train_clean - val_acc")
    for r in pick:
        g = lambda k: r[col[k]]
        L.append(f"{g('epoch'):>5} {g('lr_head'):>8} {g('train_loss'):>7} {g('train_acc_mixed'):>7} "
                 f"{g('train_clean_acc'):>8} {g('val_loss'):>8} {g('val_acc'):>7} {g('val_top5'):>8} "
                 f"{g('gap_clean_val'):>6} {g('grad_norm'):>9} {g('sec'):>5}")
    L.append("-" * 78)
    L.append(f"best_val_acc    : {best_val:.2f}  (epoch {best_epoch} / {args.epochs})")
    if ema_block is not None:
        L.append(f"best_val_acc_ema: {ema_block['best_val_acc_ema']:.2f}  "
                 f"(epoch {ema_block['best_epoch_ema']} / {args.epochs})   [EMA 独立追踪]")
        L.append(f"val_top5_ema   : {ema_block['val_top5_ema']:.2f}   "
                 f"gap_ema = {ema_block['gap_ema']:.2f} pt")
    L.append(f"val_top5@best   : {v_best['top5']:.2f}")
    L.append(f"train_clean@best: {c_best['top1']:.2f}   gap@best = {c_best['top1'] - v_best['top1']:.2f} pt")
    L.append(f"pool_acc(参考)  : {p_best['top1']:.2f}" if p_best is not None
             else "pool_acc(参考)  : 无（官方训练集已全量用作 train+val）")
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
    L.append(f"AMP 诊断        : 非有限梯度 {total_bad_grad} 步 | optimizer 实际跳过 {total_skip} 步"
             + ("（占比高需降 lr 或关 AMP）" if total_skip > 0 else ""))
    L.append("=" * 78)
    L.append(f"完整逐 epoch 日志: {csv_path}")
    L.append(f"完整诊断 json    : {run_dir / 'summary.json'}")
    L.append("=" * 78)

    report = "\n".join(L)
    print(report)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

