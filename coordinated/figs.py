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
    ax.set_xlabel('X', fontsize=configs.Fig.axis_fs)
    ax.set_ylabel('Y', fontsize=configs.Fig.axis_fs)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks([-1.0, 0, 1.0])
    ax.set_yticks([-1.0, 0, 1.0])

    ax.axhline(y=0, color='grey', linestyle=':')
    ax.axvline(x=0, color='grey', linestyle=':')
    plt.title(f'Project=Coordinated\n{what}\nstep={step}\nmax_init_weight={configs.Training.max_init_weight}')

    # plot target locations
    for coordinates in cat_id2coordinate.values():
        x, y = coordinates
        ax.scatter(x, y, s=8, c='black', zorder=1, marker='v')

    # plot embeddings
    for slot, embeddings in slot2xy.items():
        assert embeddings.shape[1] == 2
        if exclude_slot_y and slot == 'y':
            continue
        x, y = embeddings[:, 0], embeddings[:, 1]
        ax.scatter(x, y, label=slot, s=4)

    plt.legend(loc='upper right')
    print('plotting at step', step)

    return fig
