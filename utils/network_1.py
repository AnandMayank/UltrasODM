# utils/network.py
from .remamba import Remamba
from .optical_flow import FlowNet

def build_model(opt, in_frames, pred_dim):
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flow_net = FlowNet()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                Remamba(64),
                nn.MaxPool2d(2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                Remamba(32),
                nn.Conv2d(32, pred_dim, kernel_size=1),
            )

        def forward(self, frames):
            # Compute optical flow between consecutive frames
            flow_features = []
            for i in range(frames.shape[1]-1):
                flow = self.flow_net(frames[:,i], frames[:,i+1])
                flow_features.append(flow)
            flow_features = torch.stack(flow_features, dim=1)
            
            # Combine with frame features
            x = self.encoder(frames)
            x = torch.cat([x, flow_features], dim=1)
            return self.decoder(x)
    
    return CustomModel()