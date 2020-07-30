from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = root / 'coordinated'
    images = root / 'images'


class Training:
    max_init_weight = 0.1


class Eval:
    verbose = False
    eval_interval = 100
    plot_interval = 100


class Fig:
    title_label_fs = 8
    axis_fs = 12
    leg_fs = 8
    fig_size = (6, 6)
    dpi = 163 // 2
    line_width = 2

    show_embeddings = 0
    show_hiddens1 = 0
    show_hiddens2 = 1
