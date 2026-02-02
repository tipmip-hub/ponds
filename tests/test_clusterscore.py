import numpy as np
from ponds.clusterscore import match_time

"""
Testing the PONDS clusterscoring.

Things to Test:
    x test if match_time works as intended
    - test scoring functions


Legend:
    - : to do
    x : done
"""

def test_match_time():
    """
    - test if 0 timestep_uncertainty leeds to output same as input
    - test if positive and negative offsets are handled correctly
    - test if clusters are also matched if they do not have the same cluster ID
    - test if cell in D is not found anywhere in GT, it is ignored
    """

    GT_masks = {
        0: np.array([
            [[0, 0],    # time 0
             [0, 0]],
            
            [[1, 1],    # time 1
             [1, 0]],

            [[0, 0],    # time 2
             [0, 0]],
                    ],dtype=bool),
        1: np.array([
            [[0, 0],    # time 0
             [0, 0]],
            
            [[0, 0],    # time 1
             [0, 0]],

            [[0, 0],    # time 2
             [0, 0]],
                    ],dtype=bool)
    }

    D_masks = {
        0: np.array([
            [[0, 0],    # time 0
             [0, 0]],
            
            [[0, 0],    # time 1
             [0, 1]],               # <- cell is nowhere in GT

            [[0, 0],    # time 2
             [0, 0]],
                    ],dtype=bool),
        1: np.array([               # <- D-cluster ID does not match GT-cluster ID
            [[0, 1],    # time 0    # <- shifted to past
             [0, 0]],
            
            [[1, 0],    # time 1    
             [0, 0]],

            [[0, 0],    # time 2
             [1, 0]],               # <- shifted to future
                    ],dtype=bool)
    }

    # call function with zero uncertainty
    D_masks_matched = match_time(
        GT_masks,
        D_masks,
        timestep_uncertainty=0,)    # no shifts allowed
    
    assert np.array_equal(D_masks_matched[0], D_masks[0]), "Cluster 0 should remain unchanged."
    assert np.array_equal(D_masks_matched[1], D_masks[1]), "Cluster 1 should remain unchanged."

    # call function with uncertainty of 1 timestep
    D_masks_matched = match_time(
        GT_masks,
        D_masks,
        timestep_uncertainty=1,)
    
    assert np.array_equal(D_masks_matched[0], D_masks[0]), "Cluster 0 should remain unchanged."
    assert np.array_equal(D_masks_matched[1], GT_masks[0]), "Cluster 1 should be shifted to match GT cluster 0."
