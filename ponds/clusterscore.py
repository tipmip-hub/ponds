import numpy as np

def extract_G_masks(
        pd,   # ponds object
    ):
    """
    Extracts the shift cluster masks from the ponds object into a dictionary.
    """
    G_masks = {}
    for i in range(len(pd.data)-1):
        G_masks[i] = pd.data[f"c{i+1}"].values
    return G_masks

def extract_D_masks(
        td,   # toad object
    ):
    """
    Extracts the detected cluster masks from the toad object into a dictionary.
    """
    cluster_arr = td.data.ts_dts_cluster.values
    cluster_ids = np.unique(cluster_arr)
    cluster_ids = cluster_ids[cluster_ids != -1]  # exclude non-clustered points
    D_masks = {}
    for cid in cluster_ids:
        D_masks[cid] = (cluster_arr == cid)
    return D_masks

def FB_measure(
        GT_masks,
        D_masks,
    ):
    """
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
    TP = np.sum(np.logical_and(mask1, mask2))
    FP = np.sum(np.logical_and(np.logical_not(mask1), mask2))
    FN = np.sum(np.logical_and(mask1, np.logical_not(mask2)))

    score = 2*TP / (2*TP + FP + FN)
    return score

def FC_measure(
        GT_masks,
        D_masks,
    ):
    """
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. RC measure cannot be computed.")
    
    TP = 0   # true positive: number of pairs of objects in same cluster in GT and D
    FN = 0   # false negatives: number of pairs of objects in same cluster in GT but not in D
    FP = 0   # false positives: number of pairs of objects in same cluster in D but not in GT

    # count true positives
    for GT in GT_masks.values():
        for D in D_masks.values():
            intersection = np.logical_and(GT, D)
            n = np.sum(intersection)
            TP += n * (n - 1) / 2
    # count false negatives
    for GT in GT_masks.values():
        GT_size = np.sum(GT)
        FN += GT_size * (GT_size - 1) / 2
    FN -= TP
    # count false positives
    for D in D_masks.values():
        D_size = np.sum(D)
        FP += D_size * (D_size - 1) / 2
    FP -= TP

    #print(f"{TP=}, {FP=}, {FN=}")

    score = 2*TP / (2*TP + FP + FN)
    return score

def RC(
    GT_masks,
    D_masks,
):
    """
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. RC measure cannot be computed.")
    # check if D_masks is not empty
    if len(D_masks) == 0:
        return np.nan

    score = len(GT_masks)/len(D_masks)
    return score

def Frag(
    GT_masks,
    D_masks,
):
    """
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. RC measure cannot be computed.")
    # check if D_masks is not empty
    if len(D_masks) == 0:
        return np.full((len(GT_masks)), 0.0)

    score = np.zeros(len(GT_masks))

    OS = overlap_score(GT_masks, D_masks)
    for i in range(len(GT_masks)):
        BM = best_match(OS, i)
        FS = F_score(np.sum(GT_masks[i]), OS[i,:])
        score[i] = FS
    return score

def Uni(
    GT_masks,
    D_masks,
):
    """
    """
    # check if GT_masks is not empty
    if len(GT_masks) == 0:
        raise ValueError("GT_masks is empty. RC measure cannot be computed.")
    # check if D_masks is not empty
    if len(D_masks) == 0:
        return np.full(len(GT_masks), 0.0)

    score = np.zeros(len(GT_masks))

    OS = overlap_score(GT_masks, D_masks)
    for i in range(len(GT_masks)):
        BM = best_match(OS, i)
        US = U_score(np.sum(OS[:,BM]), np.sum(GT_masks[i]), OS[i,BM])
        score[i] = US
    return score

###### helper functions ######

def match_time(
        GT_masks,
        D_masks,
        timestep_uncertainty=0,
    ):
    """
    """
    # make a copy of the detected masks
    D_masks_shifted = {}
    for j in range(len(D_masks)):
        D_masks_shifted[j] = D_masks[j].copy()

    for j in range(len(D_masks)):
        vals = np.zeros((len(GT_masks), 2 * timestep_uncertainty + 1))    # make space to choose best match from
        for i in range(len(GT_masks)):
            # finde the best temporal alignment within the uncertainty range
            for t_os in range(-timestep_uncertainty, timestep_uncertainty + 1):
                shifted_D_mask = np.roll(D_masks[j] , shift=t_os, axis=0)      # shift in time axis
                vals[i, t_os + timestep_uncertainty] = np.sum(np.logical_and(GT_masks[i], shifted_D_mask))   # find number of matching voxels for shifted mask

        # choose best matching D-index; with no axis specified, argmax flattens the array
        best_ix = np.argmax(vals)%(2 * timestep_uncertainty + 1)
        # change D_masks entry correspondingly
        D_masks_shifted[j] = np.roll(D_masks[j], shift=best_ix - timestep_uncertainty, axis=0)

    return D_masks_shifted

def overlap_score(
        GT_masks,
        D_masks,
    ):
    """
    Returns the overlap score matrix between ground truth masks and detected masks.
    """
    score = np.zeros((len(GT_masks), len(D_masks)))
    for i in range(len(GT_masks)):
        for j in range(len(D_masks)):
            
            score[i, j] = np.sum(np.logical_and(GT_masks[i], D_masks[j]))   # find number of matching voxels
    return score

def F_score(
        GT_mag,
        D_ov_mags,
    ):
    """
    Penalizes for fragmentation errors - one ground truth cluster being
    detected as multiple clusters (noise cells ignored).

    return 0.0: no overlap or maximal fragmentation
    return 1.0: perfect match

    Arguments:
    - GT_mag (int): number of GT-cluster cells
    - D_ov_mags (np.ndarray): array of numbers of D-cluster cells overlapping with the target GT cluster
    """
    O = np.sum(D_ov_mags)
    A = GT_mag
    r = 1#O/float(A)

    if O <= 0:
        return 0.0
    else:
        F = 0#1 - r
        F += r * (1 - np.sum(D_ov_mags**2) / O**2)
        return 1 - F

def U_score(
        D_ov_mag,
        GT_mag,
        overlap,
    ):
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

    res = 1#overlap / GT_mag
    res *= (1 - (overlap/D_ov_mag)**2)

    return 1 - res

def best_match(overlap_score, ix):

    """
    """
    res = np.argmax(overlap_score[ix, :])
    return res