import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from coordinated import configs

plt.rcParams.update({'figure.max_open_warning': 0})


def make_scatter_plot(slot2xy: Dict[str, np.ndarray],
                      what: str,
                      step: int,
                      cat_id2coordinate: Dict[int, List[float]],
                      exclude_slot_y: bool = True,
                      ) -> plt.Figure:

    # fig
    fig, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Hidden Unit 1', fontsize=configs.Fig.axis_fs)
    ax.set_ylabel('Hidden Unit 2', fontsize=configs.Fig.axis_fs)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_yticks([-1.0, 0, 1.0])

    ax.axhline(y=0, color='grey', linestyle=':', zorder=1)
    ax.axvline(x=0, color='grey', linestyle=':', zorder=1)
    plt.title(f'{what}\nstep={step}\n')

    # plot target locations
    for coordinates in cat_id2coordinate.values():
        x, y = coordinates
        ax.scatter(x, y, s=30, c='black', zorder=3, marker='v')

    # plot embeddings
    for slot, embeddings in slot2xy.items():
        assert embeddings.shape[1] == 2
        if exclude_slot_y and slot == 'y':
            continue
        x, y = embeddings[:, 0], embeddings[:, 1]
        ax.scatter(x, y, label=slot, s=10)

    plt.legend(loc='upper right')
    plt.tight_layout()

    return fig
