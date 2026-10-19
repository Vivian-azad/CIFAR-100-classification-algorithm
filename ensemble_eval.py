#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100 集成评估脚本  v1   (2026-10-18)
配合训练脚本 v6（新增 --init-seed）使用。

为什么需要单独一个文件：训练脚本一次只评一个模型，做不了集成；
而且训练脚本从头到尾不加载官方测试集，最终验收也需要一个独立入口。

两件事：
  ① --split val   在内部验证集上评估 N 个 checkpoint + 它们的集成（可反复跑）
  ② --split test  评估**官方测试集**（纪律：只跑一次，仅最终验收）

用法（Kaggle notebook 单元格）：
  # 集成效果验证
  !python ensemble_eval.py --data ./data --split val --seed 42 \
      --ckpts runs/*_i42*/best.pt runs/*_i43*/best.pt runs/*_i44*/best.pt

  # 加 TTA（原图 + 水平翻转）
  !python ensemble_eval.py --data ./data --split val --tta 1 --ckpts ...

  # 最终验收：官方 test，只跑一次
  !python ensemble_eval.py --data ./data --split test --tta 1 --ckpts ...
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

# 复用训练脚本的切分/建模/transform，保证"评估用的 val"与"训练用的 val"逐张一致。
# （train_cifar100.py 有 __main__ 保护，import 不会触发训练）
from train_cifar100 import (stratified_split, build_model, build_eval_transform,
                            NUM_CLASSES)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="checkpoint 路径列表（best.pt 或 best_ema.pt）")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42,
                   help="切分种子，必须与训练时一致（集成三兄弟都用同一个）")
    p.add_argument("--train-per-class", type=int, default=450)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--model", default="resnet50", help="与训练时一致")
    p.add_argument("--input-size", type=int, default=32)
    p.add_argument("--bs", type=int, default=256, help="推理批大小，可开大（不吃优化器显存）")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tta", type=int, default=0, choices=[0, 1],
                   help="0=仅原图；1=原图+水平翻转（各视角分别 softmax 后平均）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save", default="", help="可选：把结果 JSON 写到该路径")
    return p.parse_args()


def build_dataset(args):
    tf = build_eval_transform(args.input_size)
    if args.split == "test":
        return datasets.CIFAR100(root=args.data, train=False, download=True, transform=tf)
    base = datasets.CIFAR100(root=args.data, train=True, download=True)
    _, val_idx, _ = stratified_split(base.targets, args.train_per_class,
                                     args.val_per_class, args.seed)
    full = datasets.CIFAR100(root=args.data, train=True, download=True, transform=tf)
    return Subset(full, val_idx)


@torch.no_grad()
def predict(model, loader, device, tta):
    """返回 (概率 [N,C], 标签 [N])。TTA 在 tensor 上做 flip，零插值伪影。"""
    model.eval()
    probs, ys = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        views = [x]
        if tta:
            views.append(torch.flip(x, dims=[-1]))
        ps = []
        for v in views:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(v)
            ps.append(F.softmax(logits.float(), dim=1))
        probs.append(torch.stack(ps).mean(0).cpu())
        ys.append(y)
    return torch.cat(probs), torch.cat(ys)


def topk(probs, ys, ks=(1, 5)):
    n = probs.size(0)
    out = []
    for k in ks:
        _, pred = probs.topk(k, dim=1)                    # [N, k]
        hit = (pred == ys.unsqueeze(1)).any(1).sum().item()
        out.append(100.0 * hit / n)
    return out


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = build_dataset(args)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print("=" * 74)
    print(f"集成评估  split={args.split}  n={len(ds)}  input={args.input_size}  "
          f"tta={args.tta}  seed={args.seed}  成员数={len(args.ckpts)}")
    print("-" * 74)

    all_probs, rows = [], []
    for i, ck in enumerate(args.ckpts, 1):
        # pretrained="none"：架构相同但不下载 97MB 权重，反正马上被 load 覆盖；
        # 且 input_size<96 时 stem 判定恒为 cifar，与训练时一致。
        model, stem = build_model(args.model, "none", NUM_CLASSES, "auto", args.input_size)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd, strict=True)            # 严格加载：键不匹配立刻报错
        model.to(device)
        probs, ys = predict(model, loader, device, args.tta)
        t1, t5 = topk(probs, ys)
        rows.append({"ckpt": ck, "top1": round(t1, 2), "top5": round(t5, 2)})
        print(f"  [{i}/{len(args.ckpts)}] top1={t1:6.2f}  top5={t5:6.2f}   {ck}")
        all_probs.append(probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 集成：各成员概率取平均（比 logit 平均对"成员置信度尺度不一"更鲁棒）
    ens = torch.stack(all_probs).mean(0)
    e1, e5 = topk(ens, ys)
    best_single = max(r["top1"] for r in rows)
    mean_single = sum(r["top1"] for r in rows) / len(rows)

    print("-" * 74)
    print(f"  单模型最好      top1={best_single:6.2f}")
    print(f"  单模型平均      top1={mean_single:6.2f}")
    print(f"  >>> 集成        top1={e1:6.2f}  top5={e5:6.2f}")
    print(f"  >>> 集成增益    {e1 - best_single:+.2f} pt (对最好单模型)   "
          f"{e1 - mean_single:+.2f} pt (对单模型均值)")
    print("=" * 74)

    if args.save:
        Path(args.save).write_text(json.dumps(
            {"split": args.split, "n": len(ds), "tta": args.tta, "seed": args.seed,
             "members": rows, "ensemble": {"top1": round(e1, 2), "top5": round(e5, 2)},
             "gain_vs_best": round(e1 - best_single, 2)},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已保存：{args.save}")


if __name__ == "__main__":
    main()
