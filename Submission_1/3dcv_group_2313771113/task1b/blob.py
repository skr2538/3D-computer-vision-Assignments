##!/usr/bin/env python3

import cv2
import numpy as np
import os
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from torch.linalg import svd


def compute_orientation(j,k, grad_x , grad_y, window_size=15, num_bins=36):
    # Step 1: Take a window around the keypoint
    window = I[j-window_size//2: j+window_size//2+1,k-window_size//2: k+window_size//2+1 ]

    # Step 2: Compute the angle of the image gradient in each pixel
    grad_x_window = grad_x[j-window_size//2: j+window_size//2+1,k-window_size//2: k+window_size//2+1 ]
    grad_y_window = grad_y_window = grad_x[j-window_size//2: j+window_size//2+1,k-window_size//2: k+window_size//2+1 ]
    angles = torch.arctan2(grad_y_window, grad_x_window)

    # Step 3: Build a histogram over all angles
    histogram, bin_edges = np.histogram(angles.cpu(), bins=num_bins, range=(-np.pi, np.pi))

    # Step 4: Entry with the maximum number of samples is the orientation
    dominant_orientation = bin_edges[np.argmax(histogram)]

    return dominant_orientation


def conv2d(x, w):
    """
    Helper function to convolve 2D tensor x with a 2D weight mask w.
    """
    sy, sx = w.shape
    padded = F.pad(x.unsqueeze(0), (sx // 2, sx // 2, sy // 2, sy // 2), mode="replicate")
    result = F.conv2d(padded.unsqueeze(0), w.unsqueeze(0).unsqueeze(0), padding='valid').squeeze()
    return result

def convolution_kernel(sigma, device):
    """
    Compute convolution kernel: sigma^2 * laplace(gauss(sigma))

    Inputs:
     - sigma: std. deviation
     - device: device

    Returns:
    - mask:tensor(H, W)
    """

    # Kernel size and width
    ks = math.ceil(sigma) * 6 + 3
    kw = ks // 2

    ######################################################################################################
    # TODO Q1: Precompute the kernel for blob filtering                                                  #
    # See lecture 2 Part A slides 41                                                                     #
    # You can use the jupyter notebook to visualize the result                                           #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # we'll use the formula from https://de.wikipedia.org/wiki/Marr-Hildreth-Operator
    size = ks
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)),
        (size, size)
    )
    kernel = torch.Tensor(kernel / np.sum(kernel))
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    result = (sigma*sigma*(F.conv2d(kernel.unsqueeze(0), laplacian_kernel, padding=1).squeeze())).to(device)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Please note: you may have boundary artifacts for which it is fine to crop the final kernel
    result = result[1:-1, 1:-1]

    return result

class SIFTDetector:
    """
    pytorch implementation of SIFT detector detector
    """

    def detect_keypoints(self, I, sigma_min=1, sigma_max=30, window_size=3, threshold=0.1):
        """
        Detect SIFT keypoints.

        Inputs:
         - I: 2D array, input image

        Returns:
        - keypoints:tensor(N, 4) (x, y, scale, orientation)
        """

        assert len(I.shape) == 2, "Image dimensions mismatch, need a grayscale image"
        device = I.device
        h, w = I.shape

        # Compute the number of blob sizes
        n_sigma = sigma_max - sigma_min + 1

        ######################################################################################################
        # TODO Q2: Implement blob detector                                                                   #
        # See lecture 2 Part A slides 41                                                                     #
        # You can use the jupyter notebook to visualize the result                                           #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        R = torch.zeros((n_sigma, h, w), device=device)
        for sigma in range(sigma_min, sigma_max + 1):
            # Compute score map for scale sigma here and insert into R
            R[sigma - sigma_min, :, :] = conv2d(I, convolution_kernel(sigma, device))


        # Threshold score map
        R[torch.abs(R) < torch.abs(R).max()*threshold] = 0



        # Do nms suppression over both, position and scale space, with a 3x3x3 window

        # Compute derivatives
        kernel_x = torch.FloatTensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]).to(device)
        kernel_y = torch.FloatTensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]]).to(device)
        # Compute gradients using filter2D

        grad_x = F.conv2d(I.unsqueeze(0), kernel_x.unsqueeze(0).unsqueeze(0)).squeeze().to(device)
        grad_y = F.conv2d(I.unsqueeze(0), kernel_y.unsqueeze(0).unsqueeze(0)).squeeze().to(device)



        a = R.shape[0]
        b = R.shape[1]
        c = R.shape[2]
        tensor_windows = torch.zeros((R.shape[0], R.shape[1],R.shape[2], window_size, window_size, window_size))
        ws = int(window_size/2)
        for i in range( ws , a-ws):
            for j in range(ws , b-ws):
              for k in range(ws , c-ws):
                tensor_windows[i,j, k] = R[i-ws:i+ws+1, j-ws:j+ws+1, k-ws:k+ws+1 ]
        # Find the maximum value in each window
        max_each_window = torch.zeros((R.shape[0], R.shape[1], R.shape[2]))
        for i in range( ws , a-ws):
            for j in range(ws , b-ws):
              for k in range(ws , c-ws):
                max_each_window[i,j,k] = torch.max(tensor_windows[i,j ,k, :, : , :])
        # Extract keypoints where the maximum value is at the center of the window
        keypoints = []
        for i in range( ws , a-ws):
            for j in range(ws , b-ws):
              for k in range(ws , c-ws):
                if(R[i,j, k] == max_each_window[i,j, k] and R[i,j,k] > 0):
                  #l =compute_orientation(j,k, grad_x, grad_y)
                  keypoints.append([j, k ,i+sigma_min])
        keypoints = torch.tensor(keypoints)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return keypoints


if __name__ == "__main__":
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = SIFTDetector()

    _img1 = cv2.imread("./data/CheckerWarp.png")
    _color1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2RGB)
    _gray1 = cv2.cvtColor(_color1, cv2.COLOR_RGB2GRAY)

    img1 = torch.tensor(_img1, device=device) / 255
    color1 = torch.tensor(_color1, device=device) / 255
    gray1 = torch.tensor(_gray1, device=device) / 255

    I = gray1

    blobs = sd.detect_keypoints(I, threshold=0.7)

    np.savetxt("keypoints.out", blobs.numpy())
    print("Saved result to keypoints.out")
