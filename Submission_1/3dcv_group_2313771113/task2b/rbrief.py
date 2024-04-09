##!/usr/bin/env python3

import torch
import torch.nn.functional as F
import sys, os
import numpy as np
from torchvision.transforms import GaussianBlur


class RBRIEF:
    """
    Brief descriptor.
    """

    def __init__(self, seed):
        """
        Create rotated brief descriptor.

        Inputs:
        - seed: Random seed for pattern
        """
        self._seed = seed

    def pattern(self, device, patch_size=17, num_pairs=256):
        ######################################################################################################
        # TODO Q1: Generate comparison pattern type I                                                        #
        # See lecture 2 part A slide 54                                                                      #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # NOTE: you can use torch.randint
        # Make sure the seed is set up with self._seed
        assert self._seed is not None, "Random seed not set"

        # The returned tensor should be of dim (num_pairs, 4)
        # where the 4 dimensions are x1, y1, x2, y2
        torch.manual_seed(self._seed)
        point_pairs = torch.randint(low=-8, high=9, size=(num_pairs, 4), dtype = torch.float)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return point_pairs

    def compute_descriptors(self, I, keypoints, device="cpu"):
        """
        Extract rBRIEF binary descriptors for given keypoints in image.

        Inputs:
        - img: 2D array, input image
        - keypoint: tensor(N, 6) with fields x, y, angle, octave, response, size
        - device: where a torch.Tensor is or will be allocated

        Returns:
        - descriptor: tensor(num_keypoint,256)
        """

        assert len(I.shape) == 2, "Image dimensions mismatch"

        # Apply blur kernel to obtain smooth derivatives
        image_blur_kernel = GaussianBlur(5, 2.0)
        I = image_blur_kernel(I.unsqueeze(0)).squeeze()

        # Get pattern
        pattern = self.pattern(device)

        # Get keypoint values
        points = keypoints[:, 0:2] # x, y
        angle = keypoints[:, 2] # clockwise

        ######################################################################################################
        # TODO Q2: Implement the rotated brief descriptor                                                    #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        rotation_matrices = torch.zeros((len(angle), 2,2), dtype = torch.float)
        for i, angle_ in enumerate(angle):
          angle_ = angle_*np.pi/180
          rotation_matrices[i] = torch.Tensor([[torch.cos(angle_), -torch.sin(angle_)],
                                            [torch.sin(angle_), torch.cos(angle_)]])
        # Compute rotated patterns

        # Discard any keypoints that have pixel locations outside the image
        point_pairs = pattern
        point_pairs =  point_pairs.reshape(256, 2, 2)
        # Apply rotation matrices to points using matrix multiplication
        result = []
        for i in range((rotation_matrices.shape[0])): #no of keypoint
          mark_for_deletion = False
          point_pair_rot = torch.zeros(256,4)
          for j in range((point_pairs.shape[0])): # 256
            marked_for_deletion = False
            translated_point = (point_pairs[j]) - torch.tensor([[int(keypoints[i][0]), int(keypoints[i][0])], [int(keypoints[i][1]), int(keypoints[i][1])]] , dtype = torch.float)
            rotated_points = (torch.matmul(torch.floor(rotation_matrices[i]), translated_point))
            rotated_points += torch.tensor([[int(keypoints[i][0]), int(keypoints[i][0])], [int(keypoints[i][1]), int(keypoints[i][1])]] , dtype = torch.float)
            x1,x2,y1,y2 = rotated_points[0,0] ,rotated_points[0,1], rotated_points[1,0] , rotated_points[1,1]
            condition = (
                keypoints[i][0] + max(y1, y2) > I.shape[1] or
                keypoints[i][0] + max(y1, y2) < 0 or
                keypoints[i][0] + min(y1, y2) > I.shape[1] or
                keypoints[i][0] + min(y1, y2) < 0 or
                keypoints[i][1] + max(x1, x2) > I.shape[0] or
                keypoints[i][1] + max(x1, x2) < 0 or
                keypoints[i][1] + min(x1, x2) > I.shape[0] or
                keypoints[i][1] + min(x1, x2) < 0
            )
            if (condition):
              mark_for_deletion = True
              #print("marked for deletion")
              point_pair_rot[j, 0] = point_pairs[j][0][0]
              point_pair_rot[j, 1] = point_pairs[j][1][0]
              point_pair_rot[j, 2] = point_pairs[j][0][1]
              point_pair_rot[j, 3] = point_pairs[j][1][1]
            else:
              point_pair_rot[j, 0] = x1
              point_pair_rot[j, 1] = y1
              point_pair_rot[j, 2] = x2
              point_pair_rot[j, 3] = y2
          desc = []
          for k in range(point_pair_rot.shape[0]):
              # Sample image intensities at line segment starts
              s_x1 = int(keypoints[i][0] + point_pair_rot[k][1])
              s_y1 = int(keypoints[i][1] + point_pair_rot[k][0])
              # Sample image intensities at line segment ends
              s_x2 = int(keypoints[i][0] + point_pair_rot[k][3])
              s_y2 = int(keypoints[i][1] + point_pair_rot[k][2])
              # Compare intensities to form a binary descriptor
              try:
                desc.append(I[s_x1,s_y1] > I[s_x2,s_y2])
              except:
                desc.append(0)
          result.append(desc)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        return torch.Tensor(result)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    import cv2
    import os
    sys.path.append('..')
    sys.path.append(os.getcwd())
    print(os.getcwd())

    from task2a.match import match


    group_id = int(open('./group_id.txt', 'r').read())

    img1 = cv2.imread("./data/Chess.png")
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("./data/ChessRotated.png")
    color2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)

    # Fields in keypoints from SIFT detector:
    # x, y, angle, octave, response, size

    keypoints1 = torch.tensor(np.loadtxt('./task2b/keypoints1.txt'), device=device)
    keypoints2 = torch.tensor(np.loadtxt('./task2b/keypoints2.txt'), device=device)

    brief = RBRIEF(seed=group_id)
    desc1 = brief.compute_descriptors(torch.tensor(gray1, device=device), keypoints1)
    desc2 = brief.compute_descriptors(torch.tensor(gray2, device=device), keypoints2)

    matches = match(
        descriptors1=desc1,
        descriptors2=desc2,
        device=device,
        dist="hamming",
        ratio=0.95,
        threshold=160,
    )

    np.savetxt("rbrief.out", matches.numpy())

