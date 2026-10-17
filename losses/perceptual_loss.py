import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg_pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device)
        self.vgg_pretrained.eval()
        self.vgg_pretrained.requires_grad_(False)
        self.net = nn.Sequential(*[self.vgg_pretrained[i] for i in range(35)])

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.loss_fn = nn.L1Loss()
        self.to(self.device).eval()

    def forward(self, generated, ground_truth):
        generated_feature = self.net((generated - self.mean) / self.std)  # ⑤ 整条链走一遍
        truth_feature = self.net((ground_truth - self.mean) / self.std)
        generated_feature_1 = self.ada(generated_feature)
        truth_feature_1 = self.ada(truth_feature)
        loss= self.loss_fn(generated_feature_1, truth_feature_1)
        return loss