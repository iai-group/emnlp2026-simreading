"""Shared matplotlib font styling for paper figures.

All figures in the paper share Figure 1's font sizes so they look
consistent: 20 pt axis labels, 15 pt tick numbers, and a 15 pt legend.
Import ``apply_paper_style`` at the top of a plotting routine and pass
``LEGEND_SIZE`` to ``legend(fontsize=...)``.
"""

import matplotlib.pyplot as plt

AXIS_LABEL_SIZE = 20   # x/y-axis labels
TICK_LABEL_SIZE = 15   # tick numbers
LEGEND_SIZE = 15       # legend text


def apply_paper_style() -> None:
    """Set rcParams so paper figures share Figure 1's font sizes.

    Axis labels (and any titles) inherit ``font.size``/``axes.labelsize``
    at 20 pt; tick numbers are set to 15 pt. Legends must still be given
    ``fontsize=LEGEND_SIZE`` explicitly at each ``legend()`` call.
    """
    plt.rcParams.update({
        'font.size': AXIS_LABEL_SIZE,
        'axes.labelsize': AXIS_LABEL_SIZE,
        'xtick.labelsize': TICK_LABEL_SIZE,
        'ytick.labelsize': TICK_LABEL_SIZE,
    })
