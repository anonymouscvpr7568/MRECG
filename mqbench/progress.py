import math
from functools import partial

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


def reset_scheduler(scheduler):
    """Reset the learning rate scheduler.
    INQ requires resetting the learning rate every iteration of the procedure.
    Example:
        >>> optimizer = inq.SGD(...)
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)
        >>> inq_scheduler = INQScheduler(optimizer, [0.5, 0.75, 0.82, 1.0], strategy="pruning")
        >>> for inq_step in range(3):
        >>>     reset_lr_scheduler(scheduler)
        >>>     inq_scheduler.step()
        >>>     for epoch in range(5):
        >>>         scheduler.step()
        >>>         train(...)
        >>> inq_scheduler.step()
        >>> validate(...)
    """
    scheduler.base_lrs = list(map(lambda group: group['initial_lr'], scheduler.optimizer.param_groups))
    last_epoch = 0
    scheduler.last_epoch = last_epoch
    scheduler.step(last_epoch)