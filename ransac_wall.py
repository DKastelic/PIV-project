"""RANSAC homography demo for the supplied wall images and matched keypoints.

Run: python ransac_wall.py --show
Dependencies: numpy scipy matplotlib opencv-python

U = x (column), V = y (row). Row i in image 1 matches row i in image 2.
Coordinates default to zero-based pixels; use --coordinate-origin 1 for
MATLAB one-based coordinates. A .mat container alone does not imply an origin.
H12 maps wall1 -> wall2; H21 maps wall2 -> wall1 (the reference).
RANSAC and normalized DLT use NumPy only. OpenCV is used only for image
reading and visualization warping, never for homography estimation.
This is a two-image experiment, not the full sequence/court project pipeline.
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def project(H, points):
    """Apply a homography; projections at infinity are reported as NaN."""
    q = np.column_stack((points, np.ones(len(points)))) @ H.T
    valid = np.abs(q[:, 2]) > 1e-12
    result = np.full((len(points), 2), np.nan)
    result[valid] = q[valid, :2] / q[valid, 2, None]
    return result

#The normalize function centers and scales the point coordinates before estimating the homography. This makes the DLT calculation more numerically stable.
def normalize(points):
    """Hartley normalization: centroid zero, mean distance sqrt(2)."""
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance < 1e-12:
        raise ValueError("Coincident points cannot determine a homography.")
    s = np.sqrt(2) / distance
    T = np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1]])
    return (points - center) * s, T



def degenerate_sample(points):
    """Reject minimal samples with duplicate or near-collinear triples."""
    try:
        p, _ = normalize(points)
    except ValueError:
        return True
    for i, j, k in combinations(range(4), 3):
        a, b = p[j] - p[i], p[k] - p[i]
        if abs(a[0] * b[1] - a[1] * b[0]) < 1e-3:
            return True
    return False


def reprojection_errors(H, src, dst):
    """Forward transfer error in destination-image pixels."""
    error = np.linalg.norm(project(H, src) - dst, axis=1)
    return np.where(np.isfinite(error), error, np.inf)


def ransac(data, fit_model, compute_errors, threshold, p, accuracy):
    if not 0 < p < 1 or not 0 < accuracy < 1:
        raise ValueError("p and accuracy must be between 0 and 1.")
    if len(data) < 4:
        raise ValueError("At least 4 correspondences are required.")

    max_iterations = int(
        np.ceil(np.log1p(-accuracy) / np.log1p(-(p**4)))
    )
    print("Maximum iterations:", max_iterations)

    rng = np.random.default_rng(42)
    best_model = None
    best_inliers = np.zeros(len(data), dtype=bool)
    i = 0

    while True:
        indices = rng.choice(len(data), 4, replace=False)
        model = fit_model(data[indices])

        errors = compute_errors(model, data)
        inliers = errors < threshold

        if inliers.sum() > best_inliers.sum():
            best_model = model
            best_inliers = inliers

        i += 1
        if i >= max_iterations:
            break

    return best_model, best_inliers

    if not 0 < p < 1 or not 0 < accuracy < 1:
        raise ValueError("p e accuracy devono essere compresi tra 0 e 1.")
    if len(data) < 4:
        raise ValueError("Servono almeno 4 corrispondenze.")

    max_iterations = int(
        np.ceil(np.log1p(-accuracy) / np.log1p(-(p**4)))
    )
    print("Numero massimo di iterazioni:", max_iterations)

    rng = np.random.default_rng(42)
    best_model = None
    best_inliers = np.zeros(len(data), dtype=bool)
    i = 0

    while True:
        indices = rng.choice(len(data), 4, replace=False)
        model = fit_model(data[indices])

        errors = compute_errors(model, data)
        inliers = errors < threshold

        if inliers.sum() > best_inliers.sum():
            best_model = model
            best_inliers = inliers

        i += 1
        if i >= max_iterations:
            break

    return best_model, best_inliers


def fit_homography(matches):
    A = []

    for x, y, u, v in matches:
        A.append([-x, -y, -1,  0,  0,  0, u*x, u*y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v*x, v*y, v])

    A = np.asarray(A, dtype=float)

    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    H = Vt[-1].reshape(3, 3)

    return H / np.linalg.norm(H)


def homography_errors(H, matches):

    source = matches[:, :2]
    reference = matches[:, 2:]

    source = np.column_stack((source, np.ones(len(source))))

    projected = source @ H.T

    valid = np.abs(projected[:, 2]) > 1e-12
    errors = np.full(len(matches), np.inf)

    projected_xy = projected[valid, :2] / projected[valid, 2, None]
    errors[valid] = np.linalg.norm(
        projected_xy - reference[valid], axis=1
    )

    return errors


def main():
    data = loadmat(Path(__file__).resolve().parent / "wall_data" / "homography_data.mat")
    matches = np.column_stack([data[k].ravel() for k in ("U1", "V1", "U2", "V2")])
    _, inliers = ransac(matches, fit_homography, homography_errors, threshold=3.0, p=0.5, accuracy=0.99)
    if inliers.sum() < 4:
        raise RuntimeError("RANSAC found fewer than four inliers.")
    return fit_homography(matches[inliers])


import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import warp

image = np.array(Image.open(Path(__file__).resolve().parent / "wall_data" / "wall1.jpg"))
H = main()

plt.figure(); plt.imshow(image); plt.title("Before homography"); plt.axis("off"); plt.show()
plt.figure(); plt.imshow(warp(image, inverse_map=np.linalg.inv(H))); plt.title("After homography"); plt.axis("off"); plt.show()