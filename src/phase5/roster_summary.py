"""12-d roster composition summary from L1 player vectors.

Shared by the Phase B cache builder (training) and the Phase5 predictor
(inference) so both sides compute exactly the same features.
"""

import numpy as np


def compute_roster_summary(
    abilities: np.ndarray, uncertainties: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Compute 12-d roster summary from L1 vectors.

    Features:
      0: mean ability norm
      1: max ability norm (star power)
      2: std of ability norms (talent spread)
      3: mean uncertainty
      4: min uncertainty (most certain player)
      5: max uncertainty (least certain player)
      6: n_players / 15 (roster fullness)
      7: top3 ability fraction (star concentration)
      8-11: ability quartile norms (Q25, Q50, Q75, Q100 of ability norms)
    """
    n_valid = mask.sum()
    if n_valid == 0:
        return np.zeros(12, dtype=np.float32)

    valid_abilities = abilities[mask.astype(bool)]
    valid_uncertainties = uncertainties[mask.astype(bool)]

    ability_norms = np.linalg.norm(valid_abilities, axis=1)
    unc_means = valid_uncertainties.mean(axis=1)

    sorted_norms = np.sort(ability_norms)[::-1]  # descending

    # Top-3 fraction
    top3_sum = sorted_norms[:3].sum()
    total_sum = sorted_norms.sum()
    top3_frac = top3_sum / max(total_sum, 1e-8)

    # Quartiles of ability norms
    quartiles = np.percentile(ability_norms, [25, 50, 75, 100])

    summary = np.array(
        [
            ability_norms.mean(),  # 0: mean ability norm
            ability_norms.max(),  # 1: max ability norm
            ability_norms.std(),  # 2: std ability norm
            unc_means.mean(),  # 3: mean uncertainty
            unc_means.min(),  # 4: min uncertainty
            unc_means.max(),  # 5: max uncertainty
            n_valid / 15.0,  # 6: roster fullness
            top3_frac,  # 7: star concentration
            quartiles[0],  # 8: Q25 ability norm
            quartiles[1],  # 9: Q50 ability norm
            quartiles[2],  # 10: Q75 ability norm
            quartiles[3],  # 11: Q100 ability norm
        ],
        dtype=np.float32,
    )
    return summary
