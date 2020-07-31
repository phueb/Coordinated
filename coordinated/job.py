import attr
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from preppy import SlidingPrep

from coordinated import configs
from coordinated.rnn import RNN
from coordinated.corpus import Corpus
from coordinated.figs import make_scatter_plot


@attr.s
class Params(object):
    # rnn
    flavor = attr.ib(validator=attr.validators.instance_of(str))
    # toy corpus
    doc_size = attr.ib(validator=attr.validators.instance_of(int))
    delay = attr.ib(validator=attr.validators.instance_of(int))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    starvation = attr.ib(validator=attr.validators.instance_of(tuple))
    sample_b = attr.ib(validator=attr.validators.instance_of(tuple))
    sample_a = attr.ib(validator=attr.validators.instance_of(tuple))
    incongruent_a = attr.ib(validator=attr.validators.instance_of(tuple))
    incongruent_b = attr.ib(validator=attr.validators.instance_of(tuple))
    size_a = attr.ib(validator=attr.validators.instance_of(tuple))
    size_b = attr.ib(validator=attr.validators.instance_of(tuple))
    drop_a = attr.ib(validator=attr.validators.instance_of(tuple))
    drop_b = attr.ib(validator=attr.validators.instance_of(tuple))
    # training
    optimizer = attr.ib(validator=attr.validators.instance_of(str))
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)


def main(param2val=None):

    if param2val is None:  # if running in IDE
        save_path = Path('saves')
    else:
        save_path = Path(param2val['save_path'])

    # this allows running main() directly in IDE instead of terminal which does not show figures by default
    if param2val is None:
        param2val = param2val_ide

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    # create toy input
    corpus = Corpus(doc_size=params.doc_size,
                    delay=params.delay,
                    num_types=params.num_types,
                    num_fragments=3,  # must be 3 so that categories can be equidistant in 2d space
                    starvation=params.starvation,
                    sample_b=params.sample_b,
                    sample_a=params.sample_a,
                    incongruent_a=params.incongruent_a,
                    incongruent_b=params.incongruent_b,
                    size_a=params.size_a,
                    size_b=params.size_b,
                    drop_a=params.drop_a,
                    drop_b=params.drop_b,
                    )
    prep = SlidingPrep([corpus.doc],
                       reverse=False,
                       num_types=None,  # None ensures that no OOV symbol is inserted and all types are represented
                       slide_size=params.batch_size,
                       batch_size=params.batch_size,
                       context_size=corpus.num_words_in_window - 1)

    # pre compute unique windows
    all_contexts = prep.reordered_windows[:, :-1]
    b_ids = [prep.store.w2id[w] for w in corpus.b]
    bool_ids = np.isin(all_contexts[:, -1], b_ids)
    all_contexts_ending_in_b = all_contexts[bool_ids]
    unique_contexts_ending_in_b_all, counts = np.unique(all_contexts_ending_in_b, axis=0, return_counts=True)
    print('stats about window counts')
    print(counts.min())
    print(counts.mean())
    print(counts.max())
    print(f'num sequences all={len(unique_contexts_ending_in_b_all):,}')

    # define 3 locations in 2d space equidistant from one another and from origin
    cat_id2coordinate = {
        0: [+1.0, +0.0],
        1: [-0.5, +0.866],
        2: [-0.5, -0.866],
    }

    rnn = RNN('srn', input_size=params.num_types)

    criterion = torch.nn.MSELoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(rnn.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(rnn.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer')

    eval_steps = []
    name2col = {}

    # train loop
    for step, batch in enumerate(prep.generate_batches()):

        # make target coordinates
        contexts = np.zeros((params.batch_size, corpus.num_words_in_window - 1))
        target_coordinates = np.zeros((params.batch_size, 2))
        for n, window in enumerate(batch):
            # figure out what target coordinate to make by figuring out what last slot is
            target_word_id = window[-1]
            target_word = prep.store.types[target_word_id]
            if target_word in corpus.y:  # supervisory signal should come from category structure in Y
                xw_id = window[1]
                xi = prep.store.types[xw_id]
                cat_id = corpus.xi2cat_id[xi]
                contexts[n] = window[:-1]
                target_coordinates[n] = cat_id2coordinate[cat_id]
            else:
                continue
                # TODO make coordinates sensitive to corpus parameter like incongruent_a when last slot != 'b'
                # target_coordinates[n] = cat_id2coordinate[cat_id]
                # target_coordinates[n] = [0.0, 0.0]

        # prepare input, targets
        inputs = torch.cuda.LongTensor(contexts)  # embedding requires 64bit LongTensor
        targets = torch.cuda.FloatTensor(target_coordinates)

        # feed forward - only hidden layer at last time step is, because there is no output layer
        rnn.train()
        optimizer.zero_grad()  # zero the gradient buffers
        last_encodings = rnn(inputs)
        mse = criterion(last_encodings, targets)  # MSE not XE

        # EVAL /////////////////////////////////////////////////////////////////////////

        # collect mse
        if step % configs.Eval.eval_interval == 0:
            eval_steps.append(step)
            mse_npy = mse.detach().cpu().numpy().item()
            name2col.setdefault('mse', []).append(mse_npy)

            # console
            print(f'step={step:>6,}/{prep.num_mbs:>6,}: mse={mse_npy:.4f}', flush=True)
            print()

        # visualize hidden states + embeddings
        if step % configs.Eval.plot_interval == 0:

            slot2hiddens1 = {}
            slot2hiddens2 = {}
            slot2embeddings = {}
            for slot, words in zip(corpus.slots,
                                   [corpus.a, corpus.x, corpus.b, corpus.y]):
                word_ids = [prep.store.w2id[w] for w in words]

                # compute embeddings
                embeddings = rnn.embed.weight.detach().cpu().numpy()[word_ids]
                slot2embeddings[slot] = embeddings

                # compute hiddens1 - just hidden state for single time step
                hiddens = rnn(torch.cuda.LongTensor(np.array([word_ids])).T)
                slot2hiddens1[slot] = hiddens.detach().cpu().numpy()

            # compute hiddens2 - last hidden state for complete sequence
            # note:not all possible contexts may be represented in corpus, but close
            hiddens = rnn(torch.cuda.LongTensor(unique_contexts_ending_in_b_all))
            slot2hiddens2['axb'] = hiddens.detach().cpu().numpy()

            # make fig
            fig_e1 = make_scatter_plot(slot2embeddings, 'embeddings', step, cat_id2coordinate)
            fig_h1 = make_scatter_plot(slot2hiddens1, 'hidden states', step, cat_id2coordinate)
            fig_h2 = make_scatter_plot(slot2hiddens2, 'hidden states', step, cat_id2coordinate)

            fig_e1.savefig(save_path / 'embeddings' / f'{step:0>6}.png')
            fig_h1.savefig(save_path / 'hiddens1' / f'{step:0>6}.png')
            fig_h2.savefig(save_path / 'hiddens2' / f'{step:0>6}.png')

            if configs.Fig.show_embeddings:
                fig_e1.show()

            if configs.Fig.show_hiddens1:
                fig_h1.show()

            if configs.Fig.show_hiddens2:
                fig_h2.show()

        # TRAIN /////////////////////////////////////////////////////////////////////////

        mse.backward()
        optimizer.step()

    # return performance as pandas Series
    series_list = []
    for name, col in name2col.items():
        print(f'Making pandas series with name={name} and length={len(col)}')
        s = pd.Series(col, index=eval_steps)
        s.name = name
        series_list.append(s)

    return series_list


if __name__ == '__main__':
    (Path('saves') / 'embeddings').mkdir(parents=True, exist_ok=True)
    (Path('saves') / 'hiddens1').mkdir(parents=True, exist_ok=True)
    (Path('saves') / 'hiddens2').mkdir(parents=True, exist_ok=True)

    param2val_ide = {
            # rnn
            'flavor': 'srn',
            # toy corpus
            'doc_size': 400_000,
            'delay': 200_000,
            'num_types': 128 // 1,  # originally 128
            'starvation': (0.0, 0.0),  # (prob before delay, prob after delay)
            'sample_a': ('super', 'super'),
            'sample_b': ('item', 'item'),
            'incongruent_a': (0.0, 0.0),  # probability that Ai is category incongruent
            'incongruent_b': (0.0, 0.0),
            'size_a': (1.0, 1.0),  # proportion of set size of A
            'size_b': (1.0, 1.0),
            'drop_a': (0.0, 0.0),
            'drop_b': (0.0, 0.0),
            # training
            'optimizer': 'sgd',
            'lr': 0.05,
            'batch_size': 64,
        }

    main()
