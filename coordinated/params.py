

param2requests = {
    'batch_size': [64],
    'lr': [0.4],

    'incongruent_a': [(1.0, 1.0)],  # , (0.0, 0.0)],
    'incongruent_b': [(1.0, 1.0)],  # , (0.0, 0.0)],

}

param2debug = {
    'doc_size': 1_000,
    'delay': 500,
}


param2default = {
    # rnn
    'flavor': 'srn',
    # toy corpus
    'doc_size': 100_000,
    'delay': 50_000,
    'num_types': 128,
    'starvation': (0.0, 0.0),  # (prob before delay, prob after delay)
    'sample_a': ('super', 'super'),
    'sample_b': ('super', 'super'),
    'incongruent_a': (0.0, 0.0),  # probability that Ai is category incongruent
    'incongruent_b': (0.0, 0.0),
    'size_a': (1.0, 1.0),  # proportion of set size of A
    'size_b': (1.0, 1.0),
    'drop_a': (0.0, 0.0),
    'drop_b': (0.0, 0.0),
    # training
    'optimizer': 'sgd',
    'lr': 0.4,  # 0.01 for adagrad, 0.5 for sgd
    'batch_size': 64,
}
