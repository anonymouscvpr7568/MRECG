import torch, sys
from collections import OrderedDict
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .mobilenet_v2 import mobilenet_v2
from .ic_automl_mobile_cpu_cls import automl_mobile
from .repoptvgg import repoptvgg
from thop import profile
from byted_nnflow.ic_automl_model.classification.utils.model_utils import load_checkpoint


def load_model(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')

    if "mobilenetv2.pth.tar" in path:
        model.load_state_dict(pretrained_dict['model'])
    else:
        new_state_dict = OrderedDict()
        model_dict = model.state_dict()
        keys = []
        for k, v in model_dict.items():
            keys.append(k)
        for k1,k2 in zip(keys, pretrained_dict):
            new_state_dict[k1] = pretrained_dict[k2]
        print(f'load pretrained checkpoint from: {path}')
        model.load_state_dict(new_state_dict)
    

    inputs = torch.randn(1, 3, 112, 112)
    flops, params = profile(model, (inputs,))
    print('flops: ', flops, 'params: ', params)

    return model
