import torch
import numpy as np

def color_wheel() -> torch.Tensor:
    """
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    n_columns = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((3,n_columns))
    col = 0

    # RY
    colorwheel[0, 0:RY] = 255
    colorwheel[1, 0:RY] = torch.floor(255 * torch.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[0, col:col + YG,] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[1, col:col + YG] = 255
    col = col + YG
    # GC
    colorwheel[1, col:col + GC] = 255
    colorwheel[2, col:col + GC] = torch.floor(255 * torch.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[1, col:col + CB] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    colorwheel[2, col:col + CB] = 255
    col = col + CB
    # BM
    colorwheel[2, col:col + BM] = 255
    colorwheel[0, col:col + BM] = torch.floor(255 * torch.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[2, col:col + MR] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    colorwheel[0, col:col + MR] = 255
    return colorwheel

def flow_visualize(flow: torch.Tensor, scale=50) -> np.ndarray:
    """
    :param flow: A flow field of dimension (2 x width x height)
    :output: The colour coded flow field according to a colour code
    The colour code can be obtained from the above function and has been adapted from the Matlab implementation Deqing Sun
    """

    colorwheel = color_wheel()  # shape [55x3]
    ncols = colorwheel.shape[1]
    u = flow[0,:,:]
    v = flow[1,:,:]
    u = u / scale
    v = v / scale
    flow_image = torch.zeros((u.shape[0], u.shape[1], 3)).to(torch.uint8)
    a = torch.arctan2(-v, -u) / torch.pi
    fk = (a+1) / 2*(ncols-1)
    #k0 = torch.floor(fk).to(torch.int32)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    rad = np.sqrt(np.square(u) + np.square(v))
    for i in range(colorwheel.shape[0]):
        tmp = colorwheel[i,:]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        col  = 1 - rad * (1-col)
        flow_image[:,:,i] = torch.floor(255 * col)

    return flow_image
