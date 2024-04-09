##!/usr/bin/env python3
import torch
import numpy as np
import cv2

def match(descriptors1, descriptors2, device, dist="norm2", threshold=0, ratio=0.5):
    """
    Brute-force descriptor match with Lowe tests and cross consistency check.
    

    Inputs:
    - descriptors1: tensor(N, feature_size),the descriptors of a keypoint
    - descriptors2: tensor(N, feature_size),the descriptors of a keypoint
    - device: where a torch.Tensor is or will be allocated
    - dist: distance metrics, hamming distance for measuring binary descriptor, and norm-2 distance for others
    - threshold: threshold for first Lowe test
    - ratio: ratio for second Lowe test

    Returns:
    - matches: tensor(M, 2), indices of corresponding matches in first and second set of descriptors,
      where matches[:, 0] denote the indices in the first and
      matches[:, 1] the indices in the second set of descriptors.
    """

    # Exponent for norm
    if dist == "hamming":
        p = 0
    else:
        p = 2.0

    ######################################################################################################
    # TODO Q1: Find the indices of corresponding matches                                                 #
    # See slide 48 of lecture 2 part A                                                                   #
    # Use cross-consistency checking and first and second Lowe test                                      #
    ######################################################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Compute distances
    # NOTE: you may use torch.cdist with p=p
    score = []
    score_ = []
    distance_matrix = torch.cdist(descriptors1, descriptors2, p=p)
    # Perform first and second lowe test
    sim_matrix = []
    m_ij = torch.min(distance_matrix, axis = 1)
    m_ij2 = torch.sort(distance_matrix, dim=1)

    sim_matrix = []
    for i in range((distance_matrix.shape[0])):
      if((m_ij[0][i] <= threshold) and (m_ij[0][i] <= ratio* m_ij2[0][i][1])):
        sim_matrix.append([i, int(m_ij2[1][i][1])])
        score.append(m_ij[0][i])
    # Forward backward consistency check
    m_ij2_b = torch.sort(distance_matrix, dim=0)
    matches= []
    for index,i in enumerate(sim_matrix):
      if(int(m_ij2_b[1][0][int(i[1])]) == i[0]):
        matches.append(i)
        score_.append(score[index])
    combined_lists = list(zip(matches, score_))

    # Sort the combined list based on the values of the first list (list1)
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Unzip the sorted list
    sorted_list1, sorted_list2 = zip(*sorted_combined_lists)

    # Sort matches using distances from best to worst

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return (torch.tensor(sorted_list1))


if __name__ == "__main__":
    # test your match function under here by using provided image, keypoints, and descriptors


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img1 = cv2.imread("./data/Chess.png")
    color1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    img2 = cv2.imread("./data/ChessRotated.png")
    color2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)

    keypoints1 = np.loadtxt("./task2a/keypoints1.in")
    keypoints2 = np.loadtxt("./task2a/keypoints2.in")
    keypoints1 = torch.tensor(keypoints1, device=device)
    keypoints2 = torch.tensor(keypoints2, device=device)

    descriptors1 = np.loadtxt("./task2a/descriptors1.in")
    descriptors2 = np.loadtxt("./task2a/descriptors2.in")
    descriptors1 = torch.tensor(descriptors1, device=device)
    descriptors2 = torch.tensor(descriptors2, device=device)

    matches = match(
        descriptors1=descriptors1,
        descriptors2=descriptors2,
        device=device,
        dist="hamming",
        ratio=0.95,
        threshold=160,
    )

    np.savetxt("./output_matches.out", matches.cpu().numpy())
