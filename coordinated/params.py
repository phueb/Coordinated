from dataclasses import dataclass

param2requests = {
    'incongruent_a': [(1.0, 1.0)],  # , (0.0, 0.0)],
    'incongruent_b': [(1.0, 1.0)],  # , (0.0, 0.0)],

}


param2default = {
    # rnn
    'flavor': 'srn',
    # toy corpus
    'doc_size': 400_000,
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
    'lr': 0.1,
    'batch_size': 64,
}


@dataclass
class Params:
    # rnn
    flavor: str
    # toy corpus
    doc_size: int
    delay: int
    num_types: int
    starvation: tuple
    sample_b: tuple
    sample_a: tuple
    incongruent_a: tuple
    incongruent_b: tuple
    size_a: tuple
    size_b: tuple
    drop_a: tuple
    drop_b: tuple
    # training
    optimizer: str
    batch_size: int
    lr: float

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)