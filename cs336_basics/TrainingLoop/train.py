#!/usr/bin/env python3
"""
CS336 Assignment 1 — Training Loop (§5.3)

把所有已实现的组件整合在一起：
  - TrainableTransformerLM  (可训练的 Transformer LM)
  - AdamW
  - cosine learning rate schedule (带 warmup)
  - gradient clipping
  - cross entropy loss
  - np.memmap 高效加载数据
  - checkpoint 保存/恢复
  - 可选 Weights & Biases 日志

用法示例
--------
python -m cs336_basics.TrainingLoop.train \
    --train_data data/train.bin \
    --val_data   data/val.bin \
    --vocab_size 10000 \
    --batch_size 8 \
    --context_length 64 \
    --d_model 128 \
    --num_layers 2 \
    --num_heads 4 \
    --d_ff 256 \
    --max_iters 500 \
    --eval_interval 50 \
    --checkpoint_dir checkpoints
"""

import argparse
import math
import os
import time

import numpy as np
import torch

# ── 项目内部导入 ─────────────────────────────────────────────────────────────
from cs336_basics.Transformer.TrainableTransformerLM import TrainableTransformerLM
from cs336_basics.TrainningATransformerLM.AdamW import AdamW
from cs336_basics.TrainningATransformerLM.CrossEntropy import cross_entropy
from cs336_basics.TrainningATransformerLM.GradientClipping import gradient_clipping
from cs336_basics.TrainningATransformerLM.Learning_rate_scheduling import scheduler
from cs336_basics.TrainingLoop.data_loading import get_batch
from cs336_basics.TrainingLoop.Checkpoint import save_checkpoint, load_checkpoint


# ═══════════════════════════════════════════════════════════════════════════
#  命令行参数
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="CS336 Transformer LM 训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------- 数据 ----------
    p.add_argument("--train_data", type=str, required=True,
                   help="训练数据路径 (.bin, np.uint16 memmap)")
    p.add_argument("--val_data", type=str, default=None,
                   help="验证数据路径 (.bin, np.uint16 memmap)")

    # ---------- 模型超参数 ----------
    p.add_argument("--vocab_size",      type=int,   required=True)
    p.add_argument("--context_length",  type=int,   default=256)
    p.add_argument("--d_model",         type=int,   default=512)
    p.add_argument("--num_layers",      type=int,   default=4)
    p.add_argument("--num_heads",       type=int,   default=8)
    p.add_argument("--d_ff",            type=int,   default=1024)
    p.add_argument("--rope_theta",      type=float, default=10000.0)

    # ---------- 优化器超参数 ----------
    p.add_argument("--max_lr",       type=float, default=1e-3,
                   help="最大学习率 (cosine schedule 峰值)")
    p.add_argument("--min_lr",       type=float, default=1e-4,
                   help="最小学习率 (cosine schedule 底部)")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.999)
    p.add_argument("--eps",          type=float, default=1e-8)
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="梯度裁剪的最大 L2 范数 (0 = 不裁剪)")

    # ---------- 学习率调度 ----------
    p.add_argument("--warmup_iters",       type=int, default=100)
    p.add_argument("--cosine_cycle_iters", type=int, default=0,
                   help="cosine 周期长度 (默认 = max_iters)")

    # ---------- 训练控制 ----------
    p.add_argument("--batch_size",          type=int, default=32)
    p.add_argument("--max_iters",           type=int, default=1000)
    p.add_argument("--eval_interval",       type=int, default=100,
                   help="每 N 步做一次 eval")
    p.add_argument("--eval_batches",        type=int, default=10,
                   help="每次 eval 平均多少个 batch")
    p.add_argument("--log_interval",        type=int, default=10,
                   help="每 N 步打印一次训练 loss")
    p.add_argument("--checkpoint_interval", type=int, default=500,
                   help="每 N 步保存一次 checkpoint")
    p.add_argument("--checkpoint_dir",      type=str, default="checkpoints",
                   help="checkpoint 保存目录")
    p.add_argument("--resume",              type=str, default=None,
                   help="从某个 checkpoint 恢复继续训练")

    # ---------- 设备 ----------
    p.add_argument("--device", type=str, default="auto",
                   help="训练设备 (auto / cpu / cuda / mps)")

    # ---------- Weights & Biases ----------
    p.add_argument("--wandb",            action="store_true",
                   help="启用 wandb 日志")
    p.add_argument("--wandb_project",    type=str, default="cs336")
    p.add_argument("--wandb_run_name",   type=str, default=None)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  数据加载 (np.memmap，内存高效)
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> np.ndarray:
    """
    用 np.memmap 读取 .bin 文件。
    数据格式: 连续的 uint16 token ID。
    np.memmap 不会把整个文件读入内存，而是按需映射，
    因此可以处理比物理内存更大的数据集。
    """
    return np.memmap(path, dtype=np.uint16, mode="r")


# ═══════════════════════════════════════════════════════════════════════════
#  评估函数
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss(model, dataset, batch_size, context_length, device, num_batches):
    """在 num_batches 个随机 batch 上计算平均 cross-entropy loss。"""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)                          # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy(logits.view(B * S, V), y.view(B * S))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


# ═══════════════════════════════════════════════════════════════════════════
#  主训练循环
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── 1. 选择设备 ──────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"✦ Device: {device}")

    # ── 2. cosine cycle 默认值 ───────────────────────────────────────────────
    if args.cosine_cycle_iters == 0:
        args.cosine_cycle_iters = args.max_iters

    # ── 3. 加载数据 (memmap) ─────────────────────────────────────────────────
    print(f"✦ Loading training data: {args.train_data}")
    train_data = load_dataset(args.train_data)
    print(f"  → {len(train_data):,} tokens")

    val_data = None
    if args.val_data is not None:
        print(f"✦ Loading validation data: {args.val_data}")
        val_data = load_dataset(args.val_data)
        print(f"  → {len(val_data):,} tokens")

    # ── 4. 构建模型 ──────────────────────────────────────────────────────────
    model = TrainableTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✦ Model parameters: {num_params:,}")

    # ── 5. 构建优化器 ────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ── 6. 恢复 checkpoint（如果有） ─────────────────────────────────────────
    start_iter = 0
    if args.resume is not None:
        print(f"✦ Resuming from: {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"  → Resumed at iteration {start_iter}")

    # ── 7. 初始化 W&B（可选） ────────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wandb.watch(model, log="all", log_freq=args.log_interval)

    # ── 8. 创建 checkpoint 目录 ──────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── 9. 训练循环 ──────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    for it in range(start_iter, args.max_iters):

        # ---- 9a. 更新学习率 (cosine schedule with warmup) ----
        lr = scheduler(
            it, args.max_lr, args.min_lr,
            args.warmup_iters, args.cosine_cycle_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- 9b. 采样 batch ----
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # ---- 9c. 前向传播 ----
        logits = model(x)                          # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy(logits.view(B * S, V), y.view(B * S))

        # ---- 9d. 反向传播 ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ---- 9e. 梯度裁剪 ----
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        # ---- 9f. 参数更新 ----
        optimizer.step()

        # ---- 9g. 打印训练 loss ----
        if it % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"  iter {it:6d} | loss {loss.item():.4f} | "
                  f"lr {lr:.6f} | time {elapsed:.1f}s")
            if args.wandb:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/iter": it,
                }, step=it)

        # ---- 9h. 定期评估 ----
        if it % args.eval_interval == 0 and it > 0:
            train_loss = estimate_loss(
                model, train_data, args.batch_size, args.context_length,
                device, args.eval_batches,
            )
            msg = f"  [eval] iter {it:6d} | train_loss {train_loss:.4f}"
            log_dict = {"eval/train_loss": train_loss, "eval/iter": it}

            if val_data is not None:
                val_loss = estimate_loss(
                    model, val_data, args.batch_size, args.context_length,
                    device, args.eval_batches,
                )
                msg += f" | val_loss {val_loss:.4f}"
                log_dict["eval/val_loss"] = val_loss

            print(msg)
            if args.wandb:
                import wandb
                wandb.log(log_dict, step=it)

        # ---- 9i. 定期保存 checkpoint ----
        if it % args.checkpoint_interval == 0 and it > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"  [ckpt] saved → {ckpt_path}")

    # ── 10. 最终 checkpoint ──────────────────────────────────────────────────
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"\n✦ Training complete!  Final checkpoint → {final_path}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
