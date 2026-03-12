import torch
import torch.nn as nn
import torchvision.models.video as models


class X3DEncoder:

    def __init__(self, device="cuda"):

        self.device = device

        # load pretrained video model
        self.model = models.r3d_18(pretrained=True)

        # remove classification head
        self.model.fc = nn.Identity()

        self.model = self.model.to(device)
        self.model.eval()

    def encode(self, clip):

        # input expected shape
        # (B, C, T, H, W)

        clip = clip.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            features = self.model(clip)

        return features