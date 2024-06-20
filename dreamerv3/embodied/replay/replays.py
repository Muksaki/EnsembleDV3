from . import generic
from . import selectors
from . import limiters
import random
import embodied
import numpy as np

class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )



class KFoldUniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, k=5, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0
      ):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    self._k = k
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )
    self._split_tables = split_dict_into_k_parts(self.table, self._k)



  def _sample(self):
    dur = generic.wait(self.limiter.want_sample, 'Replay sample is waiting')
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur > 0)
    seqs = []
    if self.online:
      # Online has not been modified
      try:
        seq = self.online_queue.popleft()
      except IndexError:
        seq = self.table[self.sampler()]
    else:
      for i in range(self._k):
        uuid = self.sampler()
        while uuid in self._split_tables[i]:
          uuid = self.sampler()
          # print(uuid)
        seq = self.table[uuid]
        seq = {k: [step[k] for step in seq] for k in seq[0]}
        seq = {k: embodied.convert(v) for k, v in seq.items()}
        if 'is_first' in seq:
          seq['is_first'][0] = True
        seqs.append(seq)
    final_seq = {k: np.stack([dict[k] for dict in seqs]) for k in seqs[0].keys()}
    return final_seq
  

def split_dict_into_k_parts(data_dict, k):
    keys = list(data_dict.keys())
    random.shuffle(keys)
    n = len(keys)
    part_size = n // k
    parts = []
    for i in range(k):
        start_index = i * part_size
        end_index = start_index + part_size if i < k - 1 else n
        part_keys = keys[start_index:end_index]
        part_dict = {key: data_dict[key] for key in part_keys}
        parts.append(part_dict)
    return parts