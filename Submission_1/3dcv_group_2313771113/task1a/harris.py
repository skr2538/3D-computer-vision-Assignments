#!/usr/bin/env python3

import cv2
import numpy as np
import os

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torch.linalg import svd


class Harris:
    """
    pytorch implementation of Harris corner detector
    """

    def compute_score(self, I, sigma=1.0, kernel_size=5, k=0.02):
        """
        Compute the score map of the harris corner detector.

        Inputs:
         - I: 2D array, input image
         - k: The k parameter from the score formula, typically in range [0, 0.2]
         - sigma: Std deviation used for structure tensor

        Returns:
         - R: Score map of size H, W
        """

        assert len(I.shape) == 2, "Image dimensions mismatch, need a grayscale image"
        device = I.device
        w, h = I.shape
        blur_kernel = GaussianBlur(kernel_size, sigma)

        # Apply blur kernel to obtain smooth derivatives
        image_blur_kernel = GaussianBlur(5, 1.0)
        I = image_blur_kernel(I.unsqueeze(0)).squeeze()

        ######################################################################################################
        # TODO Q1: Compute harris corner score of a given image                                              #
        # See lecture 2 Part A slides 18 and 20                                                              #
        # You can use the jupyter notebook to visualize the result                                           #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute derivatives
        # gradient filters
        gradient_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=torch.float32)
        gradient_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=torch.float32)

        # compute derivatives
        Ix = F.conv2d(I.view((1, 1, w, h)), gradient_kernel_x.view((1, 1, 3, 3)), padding='valid').squeeze()
        Iy = F.conv2d(I.view((1, 1, w, h)), gradient_kernel_y.view((1, 1, 3, 3)), padding='valid').squeeze()

        # pad to original size
        Ix = F.pad(Ix, [1] * 4)
        Iy = F.pad(Iy, [1] * 4)

        # Stack the derivatives
        gradient = torch.stack((Ix, Iy))

        # Compute structure tensor entries
        structure_tensor = torch.zeros((w, h, 2, 2), device=device)
        for i in range(w):
            for j in range(h):
                structure_tensor[i, j] = torch.outer(gradient[:, i, j], gradient[:, i, j])

        # Blur the entries of the structure kernel with blur_kernel
        for i in range(2):
            for j in range(2):
                structure_tensor[:, :, i, j] = blur_kernel(structure_tensor[:, :, i, j].view((1, 1, w, h)))

        # Compute eigenvalues
        # NOTE: you may use torch.linalg.eigvals(...).real
        eigenvalues = torch.zeros((w, h, 2), device=device)
        for i in range(w):
            for j in range(h):
                eigenvalues[i, j] = torch.linalg.eigvals(structure_tensor[i, j]).real

        # Compute score R
        R = torch.zeros((w, h), device=device)
        for i in range(w):
            for j in range(h):
                R[i, j] = eigenvalues[i, j, 0] * eigenvalues[i, j, 1] - k * (eigenvalues[i, j, 0] + eigenvalues[i, j, 1])**2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return R


    def detect_keypoints(self,  I, threshold=0.2, sigma=1.0, kernel_size=5, k=0.02, window_size=5):
        """
        Perform harris keypoint detection.

        Inputs:
        - I: 2D array, input image
        - threshold: score threshold
        - k: The k parameter for corner_harris, typically in range [0, 0.2]
        - sigma: std. deviation of blur kernel

        Returns:
        - keypoints:tensor(N, 2)
        """

        w, h = I.shape
        R = self.compute_score(I, sigma, kernel_size, k)
        R[R<R.max()*threshold] = 0

        # ######################################################################################################
        # TODO Q2:Non Maximal Suppression for removing adjacent corners.                                       #
        # See lecture 2 Part A slides 22                                                                       #
        # You can use the jupyter notebook to visualize the result                                             #
        # ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Build a tensor that for each pixel contains all the pixels of the window
        window_tensor = torch.zeros((w, h, window_size, window_size), device=I.device)
        padding = window_size // 2
        R_padded = F.pad(R, [padding] * 4)
        window_tensor = R_padded.unfold(0, window_size, 1).unfold(1, window_size, 1)

        # Find the maximum value in each window
        max_tensor = torch.zeros((w, h), device=R.device)
        for i in range(w):
            for j in range(h):
                max_tensor[i, j] = window_tensor[i, j].max()

        # Extract keypoints where the maximum value is at the center of the window
        keypoints = ((R > 0) & (R == max_tensor)).nonzero()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return keypoints


if __name__ == "__main__":
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h = Harris()

    _img1 = cv2.imread("../data/Chess.png")
    _color1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2RGB)
    _gray1 = cv2.cvtColor(_color1, cv2.COLOR_RGB2GRAY)

    img1 = torch.tensor(_img1, device=device) / 255
    color1 = torch.tensor(_color1, device=device) / 255
    gray1 = torch.tensor(_gray1, device=device) / 255

    I = gray1

    keypoints = h.detect_keypoints(I, sigma=1.0, threshold=0.1, k=0.05, window_size=11)

    np.savetxt("harris.out", keypoints.numpy())
    print("Saved result to harris.out")
