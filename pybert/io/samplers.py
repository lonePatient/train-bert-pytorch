import random
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler

class ShuffleBatchSampler(BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.
    The ``ShuffleBatchSampler`` adds ``shuffle`` on top of
    ``torch.utils.data.sampler.BatchSampler``.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.
    Example:
        >>> import random
        >>> from torchnlp.samplers import SortedSampler
        >>>
        >>> random.seed(123)
        >>>
        >>> list(ShuffleBatchSampler(SortedSampler(range(10)), batch_size=3, drop_last=False))
        [[6, 7, 8], [9], [3, 4, 5], [0, 1, 2]]
        >>> list(ShuffleBatchSampler(SortedSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [6, 7, 8], [3, 4, 5]]
    """

    def __init__(
            self,
            sampler,
            batch_size,
            drop_last,
            shuffle=True,
    ):
        self.shuffle = shuffle
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        # NOTE: This is not data
        batches = list(super().__iter__())
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def _identity(e):
        return e

    def __init__(self, data, sort_key=_identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)