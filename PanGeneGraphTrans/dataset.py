import numpy as np
import torch
import pandas as pd
import random
import time
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self):
        """
            graph = {'input_ids': input_ids,
                     'attention_mask':attention_mask
                     'pos_ids':pos_ids}
        """
        self.graphs = []
        self.labels = []

    def __getitem__(self, index):
        return self.graphs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def save_dataset(data_dir):
    since_begin = time.time()  # 开始时间

    label_file = data_dir + 'label.txt'
    pos_file = data_dir + 'pos.txt'
    root_file = data_dir + 'root.txt'

    # all the nodes of the graph
    x = torch.tensor(np.array(pd.read_csv(label_file, header=None)), dtype=torch.long).view(-1, )
    pos = torch.tensor(np.array(pd.read_csv(pos_file, header=None)), dtype=torch.long).view(-1, )
    root = pd.read_csv(root_file, header=None)[0].values.tolist()

    n = x.shape[0]
    subgraph_size = 128
    vocab_size = 4096

    trainDataset = GraphDataset()
    testDataset = GraphDataset()
    print('token number: ' + str(n))

    sampleId = 0
    current_node = 0
    while True:  # 每一个子图
        if current_node >= n:
            break

        num_nodes = random.randint(subgraph_size // 2, subgraph_size - 2)  # 随机最大子图大小
        # [L,R)
        L = current_node  # 左边界
        R = min(n, current_node + num_nodes)  # 右边界

        for i in range(R - 1, L - 1, -1):  # 保证在同一个连通子图里
            if root[i] == root[L]:
                R = i + 1
                break

        num_nodes = R - L  # 该子图实际大小
        current_node += num_nodes

        if num_nodes <= 10:  # 如果子图太小
            continue

        cls_id = 1
        sep_id = 2
        pad_id = 3
        mask_id = 4
        input_ids = pad_id * torch.ones(subgraph_size, dtype=torch.long)
        input_ids[0] = cls_id
        input_ids[1] = sep_id
        input_ids[2:2 + num_nodes] = x[L:L + num_nodes]

        # mask
        perm = 2 + torch.tensor(np.random.permutation(num_nodes), dtype=torch.long)

        train_num, mask_num, change_num = int(num_nodes * 0.15), int(num_nodes * 0.12), int(num_nodes * 0.03)
        train_idx, mask_idx, change_idx = perm[:train_num], perm[:mask_num], perm[mask_num: mask_num + change_num]

        label = -100 * torch.ones(subgraph_size, dtype=torch.long)
        label[train_idx] = input_ids[train_idx]

        input_ids[mask_idx] = mask_id
        input_ids[change_idx] = torch.randint(5, vocab_size, (change_idx.shape[0],), dtype=torch.long)

        attention_mask = torch.zeros(subgraph_size, dtype=torch.bool)
        attention_mask[:2 + num_nodes] = True

        pos_ids = (subgraph_size - 1) * torch.ones(subgraph_size, dtype=torch.long)  # dep[[PAD]] = subgraph_size-1
        pos_ids[0] = 0  # dep[[CLS]] = 0
        pos_ids[1] = 1  # dep[[SEP]] = 1
        pos_ids[2:2+num_nodes] = pos[L:L+num_nodes] - pos[L] + 2

        if torch.any(pos_ids < 0) or torch.any(pos_ids >= subgraph_size):
            print("\nError! Invalid data!(pos_ids)\n")
            return

        graph = {'input_ids': input_ids.cpu(),
                 'attention_mask': attention_mask.cpu(),
                 'pos_ids': pos_ids.cpu()}

        if sampleId % 20 == 0:
            testDataset.graphs.append(graph)
            testDataset.labels.append(label.cpu())
        else:
            trainDataset.graphs.append(graph)
            trainDataset.labels.append(label.cpu())

        sampleId += 1
        print('\r' + str(current_node) + '/' + str(n), end='')

    print('\n' + f'num subgraph: {sampleId}')
    print(f'Data total time: {time.time() - since_begin:.2f}')

    train_path = data_dir + "graph_dataset_train.pth"
    test_path = data_dir + "graph_dataset_test.pth"

    torch.save(trainDataset, train_path)
    torch.save(testDataset, test_path)


def load_dataset(data_dir_load):
    train_path = data_dir_load + "graph_dataset_train.pth"
    test_path = data_dir_load + "graph_dataset_test.pth"
    trainDataset = torch.load(train_path)
    testDataset = torch.load(test_path)
    return trainDataset, testDataset


if __name__ == "__main__":
    save_dataset('../data/pretrain/')
