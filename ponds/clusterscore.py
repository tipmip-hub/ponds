import numpy as np
from numpy.typing import NDArray

from typing import Any

def extract_ponds_masks(
        pd: Any,   # ponds object
    ) -> dict[int, NDArray[np.bool_]]:
    """
    Extracts the shift cluster masks from the ponds object into a dictionary.

    Arguments:
    - pd: ponds object
    Returns:
    - G_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    """
    G_masks: dict[int, NDArray[np.bool_]] = {}
    for i in range(len(pd.data)-1):
        G_masks[i] = pd.data[f"c{i+1}"].values
    return G_masks

def extract_toad_masks(
        td: Any,   # toad object
    ) -> dict[int, NDArray[np.bool_]]:
    """
    Extracts the detected cluster masks from the toad object into a dictionary.

    Arguments:
    - td: toad object
    Returns:
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    """
    cluster_arr: NDArray[np.int_] = td.data.ts_dts_cluster.values
    cluster_ids_all = np.unique(cluster_arr)
    cluster_ids = cluster_ids_all[cluster_ids_all != -1]  # exclude non-clustered points
    #cluster_ids: NDArray[np.int_] = np.unique(cluster_arr)
    #cluster_ids = cluster_ids[cluster_ids != -1]  # exclude non-clustered points
    D_masks: dict[int, NDArray[np.bool_]] = {}
    cid: np.int_
    for cid in cluster_ids:
        D_masks[int(cid)] = (cluster_arr == cid)
    return D_masks

def FB_measure(
        GT_masks: dict[int, NDArray[np.bool_]],
        D_masks: dict[int, NDArray[np.bool_]],
    ) -> float:
    """
    Computes the boolean F-measure between ground truth masks and detected masks.

    Sometimes, also called Dice coefficient.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    Returns:
    - score: float: FB measure score
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. FB measure cannot be computed.")

    # accumulate all the masks in GT_masks and D_masks
    mask1 = np.zeros_like(list(GT_masks.values())[0], dtype=np.bool_)
    for G_mask in list(GT_masks.values()):
        mask1 = np.logical_or(mask1, G_mask)
    mask2 = np.zeros_like(list(GT_masks.values())[0], dtype=np.bool_)
    for D_mask in list(D_masks.values()):
        mask2 = np.logical_or(mask2, D_mask)
    
    # find intersects and differences 
    tp = np.sum(np.logical_and(mask1, mask2))
    fp = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    fn = np.sum(np.logical_and(mask1, np.logical_not(mask2)))

    score = 2*tp / (2*tp + fp + fn)
    return float(score)

def FC_measure(
        GT_masks: dict[int, NDArray[np.bool_]],
        D_masks: dict[int, NDArray[np.bool_]],
    ) -> float:
    """
    Computes the clustering F-measure between ground truth masks and detected masks.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    Returns:
    - score: float: FC measure score
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. FC measure cannot be computed.")
    
    tp = 0   # true positive: number of pairs of objects in same cluster in GT and D
    fn = 0   # false negatives: number of pairs of objects in same cluster in GT but not in D
    fp = 0   # false positives: number of pairs of objects in same cluster in D but not in GT

    # count true positives
    for gt in GT_masks.values():
        for d in D_masks.values():
            intersection = np.logical_and(gt, d)
            n = np.sum(intersection)
            tp += n * (n - 1) / 2
    # count false negatives
    for gt in GT_masks.values():
        gt_size = np.sum(gt)
        fn += gt_size * (gt_size - 1) / 2
    fn -= tp
    # count false positives
    for d in D_masks.values():
        d_size = np.sum(d)
        fp += d_size * (d_size - 1) / 2
    fp -= tp

    #print(f"{tp=}, {fp=}, {fn=}")

    score = 2*tp / (2*tp + fp + fn)
    return float(score)

def RC(
    GT_masks: dict[int, NDArray[np.bool_]],
    D_masks: dict[int, NDArray[np.bool_]],
) -> float:
    """
    Computes the ratio of number of ground truth clusters to number of detected clusters.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    Returns:
    - score: float: RC measure score
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. RC measure cannot be computed.")
    # check if D_masks is not empty
    if len(D_masks) == 0:
        return np.nan
    if len(D_masks) == 1 and np.sum(D_masks[0]) == 0:
        return np.nan

    score = len(GT_masks)/len(D_masks)
    return float(score)

def Frag(
    GT_masks: dict[int, NDArray[np.bool_]],
    D_masks: dict[int, NDArray[np.bool_]],
) -> NDArray[np.floating] :
    """
    Computes the fragmentation score for each ground truth cluster.

    This measure ignores noise cells completely. Only cells that are part of any 
    ground truth cluster are considered.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    Returns:
    - score: NDArray[np.floating]: array of fragmentation scores for each ground truth cluster
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. Frag measure cannot be computed.")
    # check if D_masks is not empty
    if len(D_masks) == 0:
        return np.full((len(GT_masks)), 0.0)

    score = np.zeros(len(GT_masks))

    os = overlap_score(GT_masks, D_masks)
    for i in range(len(GT_masks)):
        #bm = best_match(os, i)
        fs = F_score(np.sum(GT_masks[i]), os[i,:])
        score[i] = fs
    return score

def Uni(
    GT_masks: dict[int, NDArray[np.bool_]],
    D_masks: dict[int, NDArray[np.bool_]],
) -> NDArray[np.floating] :
    """
    Computes the unification score for each ground truth cluster.

    This measure ignores noise cells completely. Only cells that are part of any 
    ground truth cluster are considered.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    Returns:
    - score: NDArray[np.floating]: array of unification scores for each ground truth cluster
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. Uni measure cannot be computed.")
    # check if D_masks is not empty
    if len(D_masks) == 0:
        return np.full(len(GT_masks), 0.0)

    score = np.zeros(len(GT_masks))

    os = overlap_score(GT_masks, D_masks)
    for i in range(len(GT_masks)):
        bm = best_match(os, i)
        us = U_score(np.sum(os[:,bm]), np.sum(GT_masks[i]), os[i,bm])
        score[i] = us
    return score

###### helper functions ######

from numpy.typing import NDArray

def match_time(
        GT_masks: dict[int, NDArray[np.bool_]],
        D_masks: dict[int, NDArray[np.bool_]],
        timestep_uncertainty: int = 0,
    ) -> dict[int, NDArray[np.bool_]]:
    """
    Shifts cells of detected masks in time to best match ground truth masks.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    - timestep_uncertainty: int: maximum allowed time shift (in number of timesteps)
    Returns:
    - D_masks_shifted: dict[int, NDArray[np.bool_]]: dictionary of time-shifted detected cluster masks
    """

    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. match_time cannot be computed.")
    
    # make space for shifted detected masks
    D_masks_shifted: dict[int, NDArray[np.bool_]] = {}

    # extract dimensions
    first_mask = GT_masks[0]
    time_dim, lat_dim, lon_dim = first_mask.shape[0], first_mask.shape[1], first_mask.shape[2]

    #for j in range(len(D_masks)):
    for j, d_mask in enumerate(D_masks):
        # make space for new detection mask
        D_mask_shifted = np.zeros((time_dim, lat_dim, lon_dim), dtype=bool)
        # get indices of detected cells in this D_mask
        ix_D0 = np.where(d_mask)
        for k in range(len(ix_D0[1])):  # loop over all detected cells
            t_Ds = ix_D0[0][k]      # initialize to original position
            lat_D0 = ix_D0[1][k]
            lon_D0 = ix_D0[2][k]

            for _, gt_mask in enumerate(GT_masks):
                ix_GT = np.where(gt_mask)
                for h in range(len(ix_GT[1])): # loop over all cells in GT cluster
                    t_GT = ix_GT[0][h]
                    lat_GT = ix_GT[1][h]
                    lon_GT = ix_GT[2][h]

                    if (lat_GT == lat_D0) and (lon_GT == lon_D0):   # check if coordinates match
                        if abs(t_Ds - t_GT) <= timestep_uncertainty:
                            t_Ds = t_GT     # update time position if within unc. bounds
        
            D_mask_shifted[t_Ds, lat_D0, lon_D0] = 1  # assign detection value to new mask
        D_masks_shifted[j] = D_mask_shifted

    return D_masks_shifted

def overlap_score(
        GT_masks: dict[int, NDArray[np.bool_]],
        D_masks: dict[int, NDArray[np.bool_]],
    ) -> NDArray[np.floating]:
    """
    Returns the overlap score matrix between ground truth masks and detected masks.

    Arguments:
    - GT_masks: dict[int, NDArray[np.bool_]]: dictionary of ground truth cluster masks
    - D_masks: dict[int, NDArray[np.bool_]]: dictionary of detected cluster masks
    Returns:
    - score: NDArray[np.floating]: overlap score matrix
    """
    score = np.zeros((len(GT_masks), len(D_masks)))
    for i, gt_mask in enumerate(GT_masks):
        for j, d_mask in enumerate(D_masks):
            
            score[i, j] = np.sum(np.logical_and(gt_mask, d_mask))   # find number of matching voxels
    return score

def F_score(
        D_ov_mags: NDArray[np.floating],
    ) -> float:
    """
    Penalizes for fragmentation errors - one ground truth cluster being
    detected as multiple clusters (noise cells ignored).

    return 0.0: no overlap or maximal fragmentation
    return 1.0: perfect match

    Arguments:
    - GT_mag (int): number of GT-cluster cells
    - D_ov_mags (np.ndarray): array of numbers of D-cluster cells overlapping with the target GT cluster
    """
    o = np.sum(D_ov_mags)

    if o <= 0:
        return 0.0
    else:
        F = np.sum(D_ov_mags**2) / o**2
        return float(F)

def U_score(
        D_ov_mag: int,
        overlap: int,
    ) -> float:
    """
    Penalizes for unification errors - one detected cluster covering
    multiple ground truth clusters (noise cells ignored).

    return 0.0: no overlap or maximal unification
    return 1.0: perfect match

    Example cases for non-perfect scoring:
    - GT cluster of size 10, detected cluster of size 30, overlap 10
      -> U_score = 0.888...
    - GT cluster of size 10, detected cluster of size 30, overlap 5 (noise from other GT clusters picked up)
      -> U_score = 0.468...

    Arguments:
    - D_ov_mag (int): number of D-cluster cells overlapping with any GT cluster
    - GT_mag (int): number of GT-cluster cells
    - overlap (int): number of D-cluster cells overlapping with target GT cluster
    """
    if overlap <= 0:
        return 0.0

    res = 1
    res *= (1 - (overlap/D_ov_mag)**2)

    return float(1 - res)

def best_match(
        overlap_scores: NDArray[np.floating],
        ix: int
    ) -> int:

    """
    Returns the index of the detected cluster that best matches the ground truth cluster
    at index ix, based on the overlap scores.

    Arguments:
    - overlap_scores: NDArray[np.floating]: overlap score matrix
    - ix: int: index of the ground truth cluster
    Returns:
    - res: int: index of the best matching detected cluster
    """
    res = np.argmax(overlap_scores[ix, :])
    return int(res)