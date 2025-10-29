# Network

import torch
from torchvision.models import efficientnet_b1

def build_model(opt,in_frames, pred_dim):
    
    model_name = getattr(opt, 'model_name', None) or opt.get('model_name')

    if model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.features[0][0].out_channels, 
            kernel_size  = model.features[0][0].kernel_size, 
            stride       = model.features[0][0].stride, 
            padding      = model.features[0][0].padding, 
            bias         = model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features   = model.classifier[1].in_features,
            out_features  = pred_dim
        )

    else:
        raise("Unknown model.")
    
    return model

