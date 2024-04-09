import torchvision.transforms as tf
from imageio.v3 import imread, imwrite
import numpy as np
import torch


def readTensor(file: str) -> torch.Tensor:
    array = read(file)

    return tf.ToTensor()(np.asarray(array, dtype=np.float32))

def writeTensor(filename: str, im: torch.Tensor):
    if im.shape[0] == 3:
        im = _unnormalize(im).numpy()
        im = np.transpose(im, (1, 2, 0))
        imwrite(filename, im)
    elif im.shape[0] == 2:
        im = im.numpy()
        im = np.transpose(im, (1, 2, 0))
        _writeFlow(filename, im)
    else:
        raise ValueError("Invalid input shape")
def _unnormalize(tensor, mean=torch.tensor([0.411,0.432,0.45], dtype=torch.float32),
                std=torch.tensor([1,1,1], dtype=torch.float32)):
    inverse_norm = tf.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    tensor = inverse_norm(tensor)
    return (tensor*255).to(torch.uint8)


def read(file: str) -> np.ndarray:
    if file.endswith('.flo'):
        return _readFlow(file)
    elif file.endswith('ppm'):
        return _readPPM(file)
    elif file.endswith('.png'):
        return imread(file)
    else:
        raise Exception('Invalid Filetype {}', file)


def _readFlow(file: str) -> np.ndarray:
    f = open(file, 'rb')
    header = np.fromfile(f, np.float32, count=1).squeeze()
    if header != 202021.25:
        raise Exception('Invalid .flo file {}', file)
    w = np.fromfile(f, np.int32, 1).squeeze()
    h = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, w*h*2).reshape((h, w, 2))

    return flow

def _readPPM(file: str) -> np.ndarray:
    return imread(file)

def _writeFlow(filename: str, flow: np.ndarray):
    f = open(filename, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)