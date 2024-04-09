##!/usr/bin/env python3

import torch
import torch.nn.functional as F
import sys, os
from torchvision.transforms import GaussianBlur
import numpy as np

class RHOG:
    """
    Brief descriptor.
    """

    def compute_descriptors(self, I, keypoints, device="cpu"):
        """
        Extract rotate hog dsecriptors for the keypoints.

        Inputs:
        - img: 2D array, input image
        - keypoint: tensor(N, 6) with fields x, y, angle, octave, response, size
        - device: where a torch.Tensor is or will be allocated

        Returns:
        - descriptor: tensor(num_keypoint,256)
        """

        assert len(I.shape) == 2, "Image dimensions mismatch"
        # Apply blur kernel to obtain smooth derivatives
        image_blur_kernel = GaussianBlur(5, 1.0)
        I = image_blur_kernel(I.unsqueeze(0)).squeeze()

        # Get keypoint values
        points = keypoints[:, 0:2] # x, y
        angle = keypoints[:, 2] # clockwise

        ######################################################################################################
        # TODO Q1: Implement the rotated hog descriptor                                                      #
        ######################################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Compute image derivatives
        kernel_x = torch.FloatTensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        kernel_y = torch.FloatTensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
        I = torch.tensor(I, dtype = torch.float)
        grad_x = F.conv2d(I.unsqueeze(0), kernel_x.unsqueeze(0).unsqueeze(0))
        grad_y = F.conv2d(I.unsqueeze(0), kernel_y.unsqueeze(0).unsqueeze(0))

        # Sample gradients from 4x4 squares that are rotated around the center of the patch by angle
        # Convert angle to radians
        result = []
        for i in range(len(points)):
          angle_rad = angle[i] * (np.pi / 180.0)
          #DEclaring the transformation matrix (translation rotation translation)
          E1 = torch.tensor([
              [np.cos(-angle_rad), - np.sin(-angle_rad),  - points[i][0] *np.cos(-angle_rad)+ points[i][1]*np.sin(-angle_rad)+ points[i][0]],
              [np.sin(-angle_rad), np.cos(-angle_rad),  - points[i][0] *np.sin(-angle_rad) - points[i][1]*np.cos(-angle_rad)+ points[i][1]],
              [0,0, 1]
          ], dtype = torch.float)
          rotated_grad_x = torch.full((grad_x.shape[1], grad_x.shape[2]), -999)
          rotated_grad_x = torch.tensor(rotated_grad_x,dtype = torch.float)
          
          rotated_grad_y = torch.full((grad_y.shape[1], grad_y.shape[2]), -999)
          rotated_grad_y = torch.tensor(rotated_grad_y,dtype = torch.float)
          x = rotated_grad_x.shape[0]
          y = rotated_grad_y.shape[1]
          new_matrix = np.zeros((3, x*y))

          # Populate the first row with the range 0 to x repeated y times
          new_matrix[1, :] = np.tile(np.arange(y), x)

          # Populate the second row with the range 0 to y repeated x times
          new_matrix[0, :] = np.repeat(np.arange(x), y)

          # Populate the third row with constant values (in this case, you can customize it)
          new_matrix[2, :] = np.ones(x * y) * 1  # For example, fill with the constant value 42

          # Reshape the matrix to have three rows
          new_matrix = new_matrix.reshape(3, x * y)
          new_p = torch.matmul(E1, torch.tensor(new_matrix, dtype = torch.float))
          # Round to the nearest integer (assuming you want integer indices)
          new_p = new_p.round().int()
          indices_x, indices_y = new_p[0], new_p[1]

          # Extract integer indices from new_matrix
          indices_matrix_x, indices_matrix_y = new_matrix[0], new_matrix[1]

          #valid_indices_mask = (indices_x >= 0) & (indices_x < X) & (indices_y >= 0) & (indices_y < Y)

          valid_indices_mask = (indices_x >= 0 & (indices_x < rotated_grad_x.shape[0] ) & (indices_y >= 0) & (indices_y < rotated_grad_x.shape[1]))


          valid_indices_mask =  ((valid_indices_mask.reshape(rotated_grad_x.shape[0], rotated_grad_x.shape[1])))
          valid_indices_mask_tensor = torch.tensor(valid_indices_mask, dtype=torch.bool)


          # Use advanced indexing to assign values to rotated grad_x only for valid indices
          rotated_grad_x[valid_indices_mask_tensor] = grad_x[0,valid_indices_mask_tensor]
          rotated_grad_y[valid_indices_mask_tensor] = grad_y[0,valid_indices_mask_tensor]

          # use interpolation for imputing missing
          rotated_grad_x = interpolate_missing_values(rotated_grad_x)
          rotated_grad_y = interpolate_missing_values(rotated_grad_y)


          rotated_grad_x = rotated_grad_x.squeeze()
          rotated_grad_y = rotated_grad_y.squeeze()

          #finding the angle at each pixel
          angle_matrix = torch.atan2(rotated_grad_y, rotated_grad_x)

          #creating the reggion around the keypoint
          x, y = points[i]
          x = int(x)
          y = int(y)
          region_16x16 = angle_matrix[y - 8:y + 8, x - 8:x + 8]
          subregions = [region_16x16[i:i+4, j:j+4] for i in range(0, 16, 4) for j in range(0, 16, 4)]
          # Assemble gradients from the 4x4 patches into orientation bins

          feature_vector = np.array([])
          for subregion in subregions:
              histogram, y = np.histogram(subregion, bins=8, range=[- np.pi, np.pi ])
              #histogram, y = np.histogram(subregion, bins=8, range=[int(np.percentile(angle_matrix, 5)), int(np.percentile(angle_matrix, 5))])
              #orientation_bins = F.conv1d(orientation_bins.unsqueeze(0), torch.ones(1, 1, 3) / 3, padding=1).squeeze(0)
              # Assemble the histograms into a 128d vector
              feature_vector = np.concatenate((feature_vector, histogram))
          # Slightly smooth the orientation histograms spatially
          #smoothed_vector = np.convolve(feature_vector, np.ones(3)/3, mode='valid')
          result.append(feature_vector)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return torch.Tensor(np.array(result))

def interpolate_missing_values(matrix):
    # Create a mask to identify points with value -999
    mask = (matrix == -999).float()

    # Convert matrix to float32
    matrix = matrix.float()

    # Create a grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(matrix.size(0)), torch.arange(matrix.size(1)))
    grid = torch.stack((grid_x, grid_y), dim=-1).float()

    # Flatten the grid and add a channel dimension
    grid_flat = grid.view(-1, 2).unsqueeze(0).unsqueeze(0)

    # Apply the interpolation using bilinear interpolation
    interpolated_matrix = F.grid_sample(matrix.unsqueeze(0).unsqueeze(0), grid_flat, align_corners=False).squeeze(0).squeeze(0)

    # Apply the mask to preserve values where the original matrix had -999

    interpolated_matrix = interpolated_matrix.reshape(matrix.shape[0], matrix.shape[1]) * mask + matrix

    '''plt.imshow(interpolated_matrix, cmap='viridis', interpolation='nearest', origin='upper')
    plt.colorbar()  # Add a colorbar to the plot for reference
    plt.title('2D Matrix Plot')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show() '''
    
    return interpolated_matrix

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    import cv2
    import os
    sys.path.append('..')
    sys.path.append(os.getcwd())
    from task2a.match import match

    img1 = cv2.imread("./data/Chess.png")
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("./data/ChessRotated.png")
    color2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)

    # Fields in keypoints from SIFT detector:
    # x, y, angle, octave, response, size
    keypoints1 = torch.tensor(np.loadtxt('./task2c/keypoints1.txt'), device=device)
    keypoints2 = torch.tensor(np.loadtxt('./task2c/keypoints1.txt'), device=device)

    hog = RHOG()
    desc1 = hog.compute_descriptors(torch.tensor(gray1, device=device), keypoints1)
    desc2 = hog.compute_descriptors(torch.tensor(gray2, device=device), keypoints2)

    matches = match(
        descriptors1=desc1,
        descriptors2=desc2,
        device=device,
        dist="euclidean",
        ratio=0.95,
        threshold=0, # Adjust value
    )

    np.savetxt("rhog.out", matches.numpy())

