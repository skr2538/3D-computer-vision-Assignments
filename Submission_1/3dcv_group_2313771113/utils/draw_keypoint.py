import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


def draw_keypoint(keypoints, img, orientations=None):
    features_img = np.copy(img)
    keypoints = keypoints.cpu().numpy()

    for keypoint in keypoints:
        x = np.round(keypoint)[1]
        y = np.round(keypoint)[0]
        cv2.ellipse(features_img, (x, y), (3, 3), 0, 0, 360, (255, 0, 0), 1)

    # draw orientations of top 5 keypoints
    if orientations is not None:
        orientations = orientations.cpu().numpy()
        for keypoint, ori in zip(keypoints[:10], orientations[:10]):
            x = np.round(keypoint)[1]
            y = np.round(keypoint)[0]
            x_offset = np.round(x + 50 * math.cos(ori)).astype(np.int32)

            y_offset = np.round(y + 50 * math.sin(ori)).astype(np.int32)
            cv2.arrowedLine(features_img, (x, y), (x_offset, y_offset), (0, 128, 0), 3)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(features_img)
