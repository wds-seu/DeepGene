import torch
from dataset import GraphDataset
import tokenizers


def load_finetune_dataset(file_dir):
    dataset = GraphDataset()
    tokenizer = tokenizers.Tokenizer.from_file("../data/vocab/tokenizer.json")
    seqs = []
    seq_max_len = 0
    max_position_embeddings = 5120

    with open(file_dir) as f:
        for line in f:
            line = line.split(',')
            if line[0] == 'sequence':
                continue

            seq = torch.tensor(tokenizer.encode(line[0], add_special_tokens=False).ids, dtype=torch.long)
            if seq.shape[0] + 2 >= max_position_embeddings:
                seq = seq[0:max_position_embeddings-2]
            seqs.append(seq)
            seq_max_len = max(seq_max_len, seq.shape[0])
            dataset.labels.append(torch.tensor(int(line[1]), dtype=torch.long))

    seq_max_len += 2

    for x in seqs:
        num_nodes = x.shape[0]

        cls_id = 1
        sep_id = 2
        pad_id = 3
        input_ids = pad_id * torch.ones(seq_max_len, dtype=torch.long)
        input_ids[0] = cls_id
        input_ids[1] = sep_id
        input_ids[2:2 + num_nodes] = x

        attention_mask = torch.zeros(seq_max_len, dtype=torch.bool)
        attention_mask[:2 + num_nodes] = True

        pos_ids = (seq_max_len - 1) * torch.ones(seq_max_len, dtype=torch.long)  # dep[[PAD]] = seq_max_len-1
        pos_ids[0] = 0  # dep[[CLS]] = 0
        pos_ids[1] = 1  # dep[[SEP]] = 1
        for i in range(2, 2 + num_nodes):
            pos_ids[i] = i

        graph = {'input_ids': input_ids.cpu(),
                 'attention_mask': attention_mask.cpu(),
                 'pos_ids': pos_ids.cpu()}

        dataset.graphs.append(graph)

    return dataset
