from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler


class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, shuffle, batch_size, num_nodes_limit, drop_last=False):
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset
        self.num_nodes_limit = num_nodes_limit
        self.actual_batches = None  # different for every epoch
        self.length = None  # this will be the same across different epochs, lack will be added, extra will be discarded

    def __iter__(self):
        actual_batches_list = self.get_actual_batch()
        print('current batches length is {0}'.format(len(actual_batches_list)))
        for i in range(len(actual_batches_list)):
            yield actual_batches_list[i]
        # reset, be ready for new batches, should execute only once
        print('resetting batches')
        self.actual_batches = None

    def __len__(self):
        actual_batches_list = self.get_actual_batch()
        return len(actual_batches_list)

    def get_actual_batch(self):
        # NOTE can only be called once per epoch
        if self.actual_batches is None:
            original_batch_list = list(self.sampler)
            res = []
            current_total_nodes = 0
            for index in original_batch_list:
                if self.batch_size != -1 and len(res) > 0 and len(res[-1]) == self.batch_size:
                    current_total_nodes = 0
                current_num_nodes = len(self.dataset[index].y)
                if self.num_nodes_limit != -1 and current_num_nodes > self.num_nodes_limit:
                    raise ValueError('single {0}th data point of size {1} exceeding limit of {2}'.format(index, current_num_nodes, self.num_nodes_limit))
                if self.num_nodes_limit != -1 and current_total_nodes + current_num_nodes > self.num_nodes_limit:
                    current_total_nodes = 0
                if current_total_nodes == 0:
                    res.append([])
                res[-1].append(index)
                current_total_nodes += current_num_nodes
            self.actual_batches = res
        return self.actual_batches
