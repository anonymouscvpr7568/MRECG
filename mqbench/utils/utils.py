import copy, sys, pdb
import yaml
from easydict import EasyDict
import os

import torch
import torch.nn as nn
import torch.fx
from torch.fx import GraphModule
from torch.nn import Module
from torch.nn import functional as F

USE_LINK = False
USE_DDP = False

try:
    import spring.linklink as link
    assert link.is_initialized()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def sync_tensor(tensor):
    global USE_LINK
    global USE_DDP
    if USE_LINK:
        if tensor.is_cuda is True:
            tensor.data = tensor.data / link.get_world_size()
            link.allreduce(tensor.data)
    elif USE_DDP:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


def pot_quantization(tensor: torch.Tensor, mode='round'):
    log2t = torch.log2(tensor)
    if mode == 'round':
        log2t = (torch.round(log2t) - log2t).detach() + log2t
    else:
        assert mode == 'floor' 
        log2t = (torch.floor(log2t) - log2t).detach() + log2t
    return 2 ** log2t



def is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def is_tracing_state():
    return torch._C._get_tracing_state()

# noinspection PyUnresolvedReferences
def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax(dim=1)
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target.long(), 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def deepcopy_graphmodule(gm: GraphModule):
    """Rewrite the deepcopy of GraphModule. (Copy its 'graph'.)

    Args:
        gm (GraphModule): 

    Returns:
        GraphModule: A deepcopied gm.
    """
    copied_gm = copy.deepcopy(gm)
    copied_gm.graph = copy.deepcopy(gm.graph)
    return copied_gm


def deepcopy_mixedmodule(mm: Module, module_list: list):
    """Support for `module_list` which splits modules' nn part and post precess.

    Args:
        mm (nn.Module)
        module_list (list): the children of the mm who are a GraphModule.

    Returns:
        nn.Module
    """
    copied_mm = copy.deepcopy(mm)
    for mname in module_list:
        mod = getattr(mm, mname)
        child_graph = copy.deepcopy(mod.graph)
        copied_child = getattr(copied_mm, mname)
        setattr(copied_child, 'graph', child_graph)
    return copied_mm


def getitem2node(model: GraphModule) -> dict:
    def _update_getitem_path(getitem_args_dict):
        for node in getitem_args_dict:
            args_list = getitem_args_dict[node]
            while args_list[0] in getitem_args_dict:
                args_list = getitem_args_dict[args_list[0]] + args_list[1:]
            getitem_args_dict[node] = args_list
        return getitem_args_dict

    def _getitem_from_args(args, original_args_dict):
        ret = original_args_dict
        for a in args:
            try:
                ret = ret[a]
            except (IndexError, KeyError):
                return {}
        return ret 
    import operator
    nodes = list(model.graph.nodes)
    # the getitem's call graph
    getitem_args_dict = {}
    # the dict used in the model 
    original_key_dict = {}
    getitem2node = {}
    for node in nodes:
        # update the getitems
        if node.target == operator.getitem:
            getitem_args_dict[node] = list(node.args)
            getitem_args_dict = _update_getitem_path(getitem_args_dict)
            for _node in getitem_args_dict:
                if _node in getitem2node:
                    continue
                val = _getitem_from_args(getitem_args_dict[_node], original_key_dict)
                if isinstance(val, torch.fx.node.Node):
                    getitem2node[_node] = val
        elif node.target == 'update':
            if node.args[0] not in original_key_dict:
                original_key_dict[node.args[0]] = {}
            if isinstance(node.args[1], dict):
                original_key_dict[node.args[0]].update(node.args[1])
            elif isinstance(node.args[1], torch.fx.node.Node):
                original_key_dict[node.args[0]].update(original_key_dict[node.args[1]])
            else:
                raise ValueError('Wrong type for update')


    return getitem2node


def _fix_succ_recursivly(args, target_node, inserted_node):
    # List / Tuple
    if isinstance(args, (list, tuple)):
        _tmp = list(args)
        for _i, _arg in enumerate(args):
            if _arg == target_node:
                _tmp[_i] = inserted_node
            elif isinstance(_arg, tuple):
                _tmp[_i] = _fix_succ_recursivly(_arg, target_node, inserted_node)
            elif isinstance(_arg, list):
                _tmp[_i] = list(_fix_succ_recursivly(_arg, target_node, inserted_node))
            elif isinstance(_arg, dict):
                _tmp[_i] = _fix_succ_recursivly(_arg, target_node, inserted_node)
        return tuple(_tmp)
    # Dict
    elif isinstance(args, dict):
        _tmp = {}
        for k, v in args.items():
            if v == target_node:
                _tmp[k] = inserted_node
            elif not isinstance(v, torch.fx.node.Node):
                _tmp[k] = _fix_succ_recursivly(v, target_node, inserted_node)
            else:
                _tmp[k] = v
        return _tmp
    else:
        raise NotImplementedError('{} can not be handled now.'.format(type(args)))


def topology_order(model):
    node2idx = {}
    for idx, node in enumerate(model.graph.nodes):
        node2idx[node] = idx 
    return node2idx


def KL_divergence(model_output, real_output):

    size_average = True

    # Target is ignored at training time. Loss is defined as KL divergence
    # between the model output and the refined labels.
    if real_output.requires_grad:
        raise ValueError("real network output should not require gradients.")

    model_output_log_prob = F.log_softmax(model_output, dim=1)
    real_output_soft = F.softmax(real_output, dim=1)
    real_output_log_prob = F.log_softmax(real_output, dim=1)
    del model_output, real_output

    # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
    # for batch matrix multiplicatio
    real_output_soft = real_output_soft.unsqueeze(1)
    model_output_log_prob = model_output_log_prob.unsqueeze(2)
    real_output_log_prob = real_output_log_prob.unsqueeze(2)

    # Compute the loss, and average/sum for the batch.
    cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob) + torch.bmm(real_output_soft, real_output_log_prob)
    if size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
    else:
            cross_entropy_loss = cross_entropy_loss.sum()
    # Return a pair of (loss_output, model_output). Model output will be
    # used for top-1 and top-5 evaluation.
    # model_output_log_prob = model_output_log_prob.squeeze(2)
    return cross_entropy_loss

class LinearTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''
    def __init__(self,
                 model: Module,
                 weight: float = None,
                 max_iter: int = 90*20019,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 lamda: float = 0.1,
                 size_average: bool = True):

        self.model = model
        self.weight = weight
        self.loss_start = max_iter * warm_up
        self.p = p
        self.lamda = lamda
        self.size_average = size_average

        self.temp_decay = LinearTempDecay(max_iter, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, label_smoothing=0.0):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        batch_loss = 0.
        if label_smoothing:
            logsoftmax = nn.LogSoftmax(dim=1)
            n_classes = pred.size(1)
            # convert to one-hot
            tgt = torch.unsqueeze(tgt, 1)
            soft_target = torch.zeros_like(pred)
            soft_target.scatter_(1, tgt.long(), 1)
            # label smoothing
            soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
            batch_loss = torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))
        else:
            for i in range(pred.size(0)):
                numerator = torch.exp(pred[i, tgt[i].data])
                denominator = torch.sum(torch.exp(pred[i, :]))

                loss = -torch.log(numerator / denominator)
                if self.weight:
                    loss = self.weight[tgt[i]] * loss
                batch_loss += loss
            if self.size_average == True:
                batch_loss /= pred.size(0)
        
        self.count += 1

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.model.modules():
                if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                    round_vals = layer.weight_fake_quant.get_soft_weight(layer.weight)
                    round_loss += self.lamda * (1 - ((round_vals - (torch.floor(round_vals)+.5)).abs() * 2).pow(b)).sum()

        total_loss = batch_loss + round_loss
        return total_loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss