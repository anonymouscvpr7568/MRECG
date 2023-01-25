import torch, sys
import numpy as np
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils.hook import PerChannelLoadHook

_version_under_1100 = int(torch.__version__.split('.')[1]) < 10

def _rectified_sigmoid(alpha, quant_min, quant_max):
    """Function to generate rounding mask.

    Args:
        x (torch.Tensor):
        zeta (torch.Tensor):
        gamma (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return alpha.clamp(quant_min, quant_max)


def adaround_forward(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha, zeta, gamma, min_val, hard_value=False):
    if ch_axis != -1:
        new_shape = [1] * len(x.shape)
        new_shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(new_shape)
        zero_point = zero_point.reshape(new_shape)
    for i in range(len(x.size())-1):
        min_val = min_val.unsqueeze(-1)
    zero_x = torch.zeros_like(x)
    min_val = zero_x + min_val
    min_floor = torch.floor(min_val / scale)
    if hard_value:
        # x1 = x.clone().detach() / scale
        x = min_floor + torch.round(_rectified_sigmoid(alpha, quant_min, quant_max))
        # a1 = torch.where((x1 - x).abs()<=1,1,0).sum()
        # a2 = torch.where((x1 - x).abs()<=2,1,0).sum()
        # a3 = torch.where((x1 - x).abs()<=3,1,0).sum()
        # a4 = torch.where((x1 - x).abs()<=4,1,0).sum()
        # a5 = torch.where((x1 - x).abs()<=5,1,0).sum()
        # print("*"*20)
        # print(a1, a2-a1, a3-a2, a4-a3, a5-a4)
        # print("*"*20)
    else:
        x = min_floor + _rectified_sigmoid(alpha, quant_min, quant_max)
    x += zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = (x - zero_point) * scale
    return x


class AdaRoundFakeQuantize(QuantizeBase):
    """This is based on the fixedpointquantize. Because adaround only works at FC and Conv, there is an extra variables
    to define the state and could only serve as weight quantizer.
    self.adaround basicquantize (False) adaroundquantize(True)
    """

    def __init__(self, observer, **observer_kwargs):
        super(AdaRoundFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.adaround = False
        self.load_state_dict_hook = PerChannelLoadHook(self, hook_param=['scale', 'zero_point', 'alpha'])

    def init(self, weight_tensor: torch.Tensor, round_mode='learned_hard_sigmoid', ):
        self.adaround = True
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1
        self.round_mode = round_mode

        # self.soft_targets = False  # delete this
        self.gamma, self.zeta = -0.1, 1.1
        self.init_alpha(x=weight_tensor.data.clone())

    def init_alpha(self, x: torch.Tensor):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        min_val = self.activation_post_process.min_val
        for i in range(len(x.size())-1):
            min_val = min_val.unsqueeze(-1)
        zero_x = torch.zeros_like(x)
        min_val = zero_x + min_val
        min_floor = torch.floor(min_val / scale)
        # x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / scale) - min_floor  # rest of rounding [0, 1)
            # alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            alpha = rest
            self.alpha = Parameter(alpha)
        else:
            raise NotImplementedError
    
    def init_freezealpha(self, spilt):
        if self.alpha is None:
            raise NotImplementedError
        zeros = torch.zeros_like(self.alpha.data)
        ones = torch.ones_like(self.alpha.data)
        quantile = np.quantile(torch.abs(self.alpha.data.cpu()).numpy(), 1 - spilt)
        self.mask = torch.where(torch.abs(self.alpha.data) >= quantile, ones, zeros)

        self.freeze_alpha = self.alpha * self.mask - self.alpha.detach() * self.mask + self.alpha.detach()

    def rectified_sigmoid(self):
        """Function to generate rounding mask.

        Args:
            x (torch.Tensor):
            zeta (torch.Tensor):
            gamma (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.alpha.clamp(self.quant_min, self.quant_max)

    def get_hard_value(self, X):
        X = adaround_forward(X, self.scale.data, self.zero_point.data.float(), self.quant_min,
                             self.quant_max, self.ch_axis, self.alpha, self.zeta, self.gamma, self.activation_post_process.min_val, hard_value=True)
        return X

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if not self.adaround:
                if self.is_per_channel:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale.data, self.zero_point.data.float(), self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, self.scale.item(), int(self.zero_point.item()),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                if self.round_mode == 'learned_hard_sigmoid':
                    X = adaround_forward(X, self.scale.data, self.zero_point.data.float(), self.quant_min,
                                         self.quant_max, self.ch_axis, self.alpha, self.zeta, self.gamma, self.activation_post_process.min_val)
                else:
                    raise NotImplementedError
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List',
                   self.zero_point if self.ch_axis == -1 else 'List')
