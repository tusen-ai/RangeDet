from __future__ import division
from __future__ import print_function

from threading import Thread, Lock

import mxnet as mx
import numpy as np
from six.moves.queue import Queue


class DetectionAugmentation(object):
    def __init__(self):
        pass

    def apply(self, input_record):
        pass


class PostMergeBatchLoader(mx.io.DataIter):
    """
    LowCostWorkerLoader is a 3-thread design for heavy cost worker.
    Compared to Loader, job of collecting DataBatch is now responsible for collector.
    """

    def __init__(self, roidb, transform, data_name, label_name, batch_size=1,
                 shuffle=False, num_worker=None, num_collector=None,
                 worker_queue_depth=None, collector_queue_depth=None, rank=0, num_partition=1):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size:
        :param shuffle: bool
        :return: Loader
        """
        super(PostMergeBatchLoader, self).__init__(batch_size=batch_size)

        self.rank = rank
        self.num_partition = num_partition

        # data processing utilities
        self.transform = transform

        # save parameters as properties
        self.roidb = roidb
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=5)

        # infer properties from roidb
        self.total_index = np.arange(len(self.roidb))
        self.partition_count = (len(roidb) + self.num_partition - 1) // self.num_partition
        if self.rank == self.num_partition - 1:
            self.index = self.total_index[-self.partition_count:]
        else:
            self.index = self.total_index[self.rank * self.partition_count: (self.rank + 1) * self.partition_count]
        print("loader rank:{}, partition count:{}".format(self.rank, self.partition_count))

        # decide data and label names
        self.data_name = data_name
        self.label_name = label_name

        # status variable for synchronization between get_data and get_label
        self._cur = 0

        self.data = None
        self.label = None

        # multi-thread settings
        self.num_worker = num_worker
        self.num_collector = num_collector
        self.index_queue = Queue()
        self.data_queue = Queue(maxsize=worker_queue_depth)
        self.result_queue = Queue(maxsize=collector_queue_depth)
        self.lock = Lock()
        self.workers = None
        self.collectors = None

        # get first batch to fill in provide_data and provide_label
        self._thread_start()
        self.load_first_batch()
        self.reset()

    @property
    def total_record(self):
        return len(self.index) // self.batch_size * self.batch_size

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def _insert_queue(self):
        for i in range(0, len(self.index), self.batch_size):
            batch_index = self.index[i:i + self.batch_size]
            if len(batch_index) == self.batch_size:
                self.index_queue.put(batch_index)

    def _thread_start(self):
        self.workers = \
            [Thread(target=self.worker, args=[self.roidb, self.index_queue, self.data_queue])
             for _ in range(self.num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self.collectors = [Thread(target=self.collector, args=[]) for _ in range(self.num_collector)]

        for c in self.collectors:
            c.daemon = True
            c.start()

    def reset(self):
        self._cur = 0
        if self.shuffle:
            self.random_state.shuffle(self.total_index)
            if self.rank == self.num_partition - 1:
                self.index = self.total_index[-self.partition_count:]
            else:
                self.index = self.total_index[self.rank * self.partition_count: (self.rank + 1) * self.partition_count]

        self._insert_queue()

    def iter_next(self):
        return self._cur + self.batch_size <= len(self.index)

    def load_first_batch(self):
        self.index_queue.put(range(self.batch_size))
        self.next()

    def load_batch(self):
        self._cur += self.batch_size
        result = self.result_queue.get()
        return result

    def next(self):
        if self.iter_next():
            result = self.load_batch()
            self.data = result.data
            self.label = result.label
            return result
        else:
            raise StopIteration

    def worker(self, roidb, index_queue, data_queue):
        while True:
            batch_index = index_queue.get()
            records = []
            for index in batch_index:
                roi_record = roidb[index].copy()
                for trans in self.transform:
                    trans.apply(roi_record)
                records.append(roi_record)
            data_queue.put(records)

    def collector(self):
        while True:
            records = self.data_queue.get()

            data_batch = {}
            for name in self.data_name + self.label_name:
                try:
                    data_batch[name] = np.stack([r[name] for r in records])
                except Exception as e:
                    raise Exception('{} raise an error'.format(name))

            data = [mx.nd.array(data_batch[name]) for name in self.data_name]
            label = [mx.nd.array(data_batch[name]) for name in self.label_name]
            provide_data = [(k, v.shape) for k, v in zip(self.data_name, data)]
            provide_label = [(k, v.shape) for k, v in zip(self.label_name, label)]

            data_batch = mx.io.DataBatch(data=data,
                                         label=label,
                                         provide_data=provide_data,
                                         provide_label=provide_label)
            self.result_queue.put(data_batch)

    def __len__(self):
        return self.total_record
