import torch
import numpy as np

def AEPE(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the average endpoint error between two flow fields
    :param: prediction and target flow field (2 x Height x Width)
    :output: average EPE between the flow fields
    """

    ######################################################################################################
    # Part1 Q1) Implement the average endpoint error class in the file eval.py
    ######################################################################################################

    # ***** START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    aepe = -1 # TODO

    return 1/(prediction.shape[1]*prediction.shape[2]) * torch.linalg.norm(prediction - target)
    # ***** END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return aepe
