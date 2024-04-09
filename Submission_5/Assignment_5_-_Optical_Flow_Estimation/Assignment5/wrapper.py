import matplotlib.pyplot as plt
import torch
from inout import imwrite
import numpy as np
import torchvision.transforms as tf

import torch.nn.functional as F
from models.FlowNetS import FlowNetS
from models.FlowNetC import FlowNetC

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()

class ModelWrapper:

    def __init__(self, device: str):
        self.net = None
        self.model = None
        self.device = device
        self.image_transform = tf.Compose([
            ArrayToTensor(),
            tf.Normalize(mean=[0,0,0], std=[255,255,255]),
            tf.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
        ])

    def load_net(self, path: str):
        state_dict = torch.load(path)
        self.model = state_dict["model_type"]
        if self.model == 'FlowNetS':
            self.net = FlowNetS()
        elif self.model == 'FlowNetC':
            self.net = FlowNetC()
        else:
            raise ValueError("Invalid Checkpoint!")
        self.net.load_state_dict(state_dict["state_dict"], strict=True)
        self.net.to(self.device)

    @torch.no_grad()
    def prepare_inputs(self, image1, image2):
        image1, image2 = self.image_transform(image1).to(self.device), self.image_transform(image2).to(self.device)
        return image1[None,...], image2[None,...]

    def eval(self, image1, image2):
        self.net.eval()
        image1, image2 = self.prepare_inputs(image1, image2)
        # Note: the 20 here is the same as due to numerical issues
        # The model was trained with 1/20 times the gt
        pred = 20 * self.net(image1, image2)
        return pred.squeeze()

    @torch.no_grad()
    def test_correlation(self, image1, image2, gt_path):
        assert self.model == "FlowNetC", "Correlation Layer is only implemented in the FlowNetC model!"

        image1, image2 = self.prepare_inputs(image1, image2)
        corr_pred = self.net.correlate(image1, image2)

        corr_gt = torch.load(gt_path).to(corr_pred.device)


        #changed the code here to ccheck the MSE and how the image looks
        print( torch.allclose(corr_gt, corr_pred, rtol=0, atol=1e-4))
        print(F.mse_loss(corr_gt, corr_pred).item())
