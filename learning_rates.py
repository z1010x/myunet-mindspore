import numpy as np


def cosine_lr(base_lr, decay_steps, total_steps):
    # 总共步长total_steps eg15530 单个ep步长：decay_steps eg1553
    # cosine_lr(BASE_LR, iters_per_epoch,total_train_steps)
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * (
            (1.0 - step_ / decay_steps)**power) + end_lr


def exponential_lr(base_lr,
                   decay_steps,
                   decay_rate,
                   total_steps,
                   staircase=False):
    for i in range(total_steps):
        if staircase:
            power_ = i // decay_steps
        else:
            power_ = float(i) / decay_steps
        yield base_lr * (decay_rate**power_)
