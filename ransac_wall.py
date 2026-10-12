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


def normalize(points):
    """Hartley normalization: centroid zero, mean distance sqrt(2)."""
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance < 1e-12:
        raise ValueError("Coincident points cannot determine a homography.")
    s = np.sqrt(2) / distance
    T = np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1]])
    return (points - center) * s, T


def normalized_dlt(src, dst):
    """Estimate dst ~ H @ src with normalized direct linear transformation."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError("Expected matching N x 2 arrays, N >= 4.")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("Coordinates must be finite.")
    a, T1 = normalize(src)
    b, T2 = normalize(dst)
    x, y = a.T
    u, v = b.T
    zeros, ones = np.zeros(len(a)), np.ones(len(a))
    A = np.empty((2 * len(a), 9))
    A[0::2] = np.column_stack((-x, -y, -ones, zeros, zeros, zeros, u*x, u*y, u))
    A[1::2] = np.column_stack((zeros, zeros, zeros, -x, -y, -ones, v*x, v*y, v))
    # For four pairs A is 8 x 9: full Vt is essential to retain its null vector.
    _, singular, Vt = np.linalg.svd(A, full_matrices=(len(a) == 4))
    if np.count_nonzero(singular > singular[0] * 1e-12) < 8:
        raise ValueError("Degenerate point configuration.")
    H = np.linalg.solve(T2, Vt[-1].reshape(3, 3)) @ T1
    H /= np.linalg.norm(H)
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError("Singular homography.")
    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    return H


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

def ransac_homography(src, dst, threshold=3.0, max_iterations=72,
                      confidence=0.999, seed=42):
    """Four-point RANSAC, adaptive trial budget, and consensus DLT refitting."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    rng = np.random.default_rng(seed)
    best_H, best_mask, best_score = None, None, (-1, -np.inf)
    limit, trials = max_iterations, 0
    while trials < limit:
        trials += 1
        idx = rng.choice(len(src), 4, replace=False)
        if degenerate_sample(src[idx]) or degenerate_sample(dst[idx]):
            continue
        try:
            H = normalized_dlt(src[idx], dst[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = reprojection_errors(H, src, dst)
        mask = errors <= threshold
        count = int(mask.sum())
        score = (count, -float(np.median(errors[mask])) if count else -np.inf)
        if count >= 4 and score > best_score:
            best_H, best_mask, best_score = H, mask, score
            all_inlier_probability = (count / len(src)) ** 4
            required = (1 if all_inlier_probability >= 1 else
                        int(np.ceil(np.log1p(-confidence) / np.log1p(-all_inlier_probability))))
            limit = min(limit, max(trials, required))
    if best_H is None:
        raise ValueError("RANSAC found no valid homography; check the correspondences.")
    # Re-estimate on the consensus and reclassify; preserve a valid model if refit fails.
    for _ in range(10):
        try:
            refined = normalized_dlt(src[best_mask], dst[best_mask])
        except (ValueError, np.linalg.LinAlgError):
            break
        new_mask = reprojection_errors(refined, src, dst) <= threshold
        if new_mask.sum() < 4:
            break
        stable = np.array_equal(best_mask, new_mask)
        best_H, best_mask = refined, new_mask
        if stable:
            break
    errors = reprojection_errors(best_H, src, dst)
    return best_H, errors <= threshold, errors, trials


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=root / 'wall_data' / 'homography_data.mat')
    parser.add_argument('--image1', type=Path, default=root / 'wall_data' / 'wall1.jpg')
    parser.add_argument('--image2', type=Path, default=root / 'wall_data' / 'wall2.jpg')
    parser.add_argument('--output', type=Path, default=root / 'wall_results')
    parser.add_argument('--threshold', type=float, default=3.0, help='Inlier tolerance in wall2 pixels')
    parser.add_argument('--max-iterations', type=int, default=5000)
    parser.add_argument('--confidence', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--coordinate-origin', type=int, choices=[0, 1], default=0)
    parser.add_argument('--show', action='store_true', help='Also display the diagnostic figure')
    args = parser.parse_args()

    import cv2
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = loadmat(args.data)
    missing = {'U1', 'V1', 'U2', 'V2'} - data.keys()
    if missing:
        raise ValueError(f'Missing MAT arrays: {sorted(missing)}')
    p1 = np.column_stack((data['U1'].ravel(), data['V1'].ravel())).astype(float)
    p2 = np.column_stack((data['U2'].ravel(), data['V2'].ravel())).astype(float)
    p1 -= args.coordinate_origin
    p2 -= args.coordinate_origin
    images = []
    for path in (args.image1, args.image2):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f'Could not read {path}')
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im1, im2 = images
    for points, img in zip((p1, p2), images):
        if ((points < 0).any() or (points[:, 0] >= img.shape[1]).any()
                or (points[:, 1] >= img.shape[0]).any()):
            raise ValueError('Keypoints fall outside the image; check image size and coordinate origin.')

    H12, inliers, errors, trials = ransac_homography(
        p1, p2, args.threshold, args.max_iterations, args.confidence, args.seed)
    H21 = np.linalg.inv(H12)
    if abs(H21[2, 2]) > 1e-12:
        H21 /= H21[2, 2]
    args.output.mkdir(parents=True, exist_ok=True)
    savemat(args.output / 'homography_wall.mat', {
        'H12': H12, 'H21': H21, 'H': H21, 'inliers': inliers[:, None],
        'errors_px': errors[:, None], 'points1': p1, 'points2': p2,
        'threshold_px': args.threshold, 'seed': args.seed,
        'input_coordinate_origin': args.coordinate_origin, 'output_coordinate_origin': 0,
    })
    np.savetxt(args.output / 'matches.csv', np.column_stack((
        np.arange(len(p1)), p1, p2, inliers.astype(int), errors)), delimiter=',',
        header='match_index_0based,x1,y1,x2,y2,inlier,error_px', comments='')

    # Warp image 2 back onto image 1, using the inverse of the fitted map.
    size = (im1.shape[1], im1.shape[0])
    warped = cv2.warpPerspective(im2, H21, size)
    valid = cv2.warpPerspective(np.ones(im2.shape[:2], np.uint8), H21, size,
                                flags=cv2.INTER_NEAREST).astype(bool)
    overlay = im1.copy()
    overlay[valid] = (0.5 * im1[valid] + 0.5 * warped[valid]).astype(np.uint8)
    for name, img in [('wall2_warped_to_wall1.png', warped), ('overlay.png', overlay)]:
        if not cv2.imwrite(str(args.output / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
            raise OSError(f'Failed to write {name}')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    canvas = np.zeros((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), np.uint8)
    canvas[:im1.shape[0], :im1.shape[1]] = im1
    canvas[:im2.shape[0], im1.shape[1]:] = im2
    for ax, mask, title, color in [
        (axes[0, 0], inliers, f'Inliers: {inliers.sum()} / {len(inliers)}', '#00ff88'),
        (axes[0, 1], ~inliers, f'Rejected matches: {(~inliers).sum()}', '#ff5577')]:
        ax.imshow(canvas)
        for a, b in zip(p1[mask], p2[mask]):
            ax.plot([a[0], b[0]+im1.shape[1]], [a[1], b[1]], '-o', color=color,
                    linewidth=0.65, markersize=2, alpha=0.8)
        ax.set_title(title + '\nwall1 (left) -> wall2 (right)')
        ax.axis('off')
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('wall2 aligned to wall1: 50/50 overlay')
    axes[1, 0].axis('off')
    axes[1, 1].scatter(np.arange(len(errors)), errors,
                       c=np.where(inliers, '#138a50', '#dd3344'), s=20)
    axes[1, 1].axhline(args.threshold, color='black', linestyle='--', label='Inlier threshold')
    axes[1, 1].set(xlabel='Match index (zero-based)', ylabel='Forward reprojection error (pixels)',
                   yscale='symlog', title='Errors for all supplied correspondences')
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend()
    fig.suptitle('From-scratch RANSAC + normalized DLT', fontsize=16)
    fig.savefig(args.output / 'ransac_diagnostics.png', dpi=160)
    report = (f'Matches: {len(p1)}\nInliers: {inliers.sum()}\nOutliers: {(~inliers).sum()}\n'
              f'Threshold: {args.threshold} px\nSeed: {args.seed}\nTrials: {trials}\n'
              f'Input coordinate origin: {args.coordinate_origin}\n'
              f'Inlier median error: {np.median(errors[inliers]):.4f} px\n'
              f'Inlier RMSE: {np.sqrt(np.mean(errors[inliers]**2)):.4f} px\n'
              f'H12 (wall1 -> wall2, zero-based pixels):\n{H12}\n'
              f'H21 / H (wall2 -> wall1, zero-based pixels):\n{H21}\n'
              'The dominant wall plane is aligned; foreground objects can show parallax.\n')
    (args.output / 'summary.txt').write_text(report, encoding='utf-8')
    print(report)
    print(f'Results: {args.output.resolve()}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
