import tokenizers
import re
from queue import Queue
import torch
import sys

sys.setrecursionlimit(1000000)


class UnionSet:
    def __init__(self, node):
        self.parent = {}
        self.rank = {}
        for i, val in enumerate(node):
            self.parent[val] = val
            self.rank[val] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


class Tarjan:
    def __init__(self, node, edge):
        self.node = node
        self.edge = edge
        self.sets = {}  # sets[5] = ['s1+','s2+']
        self.dfn = {}
        self.low = {}
        self.color = {}
        self.index = 0
        self.col = 0
        self.stack = []

    def dfs(self, cur):
        self.index += 1
        self.dfn[cur], self.low[cur] = self.index, self.index

        self.stack.append(cur)
        for nxt in self.edge[cur]:
            if nxt not in self.dfn:
                self.dfs(nxt)
                self.low[cur] = min(self.low[cur], self.low[nxt])
            elif nxt not in self.color:
                self.low[cur] = min(self.low[cur], self.low[nxt])

        if self.low[cur] == self.dfn[cur]:
            self.color[cur] = self.col
            self.sets[self.col] = [cur]
            while self.stack[-1] != cur:
                self.color[self.stack[-1]] = self.col
                self.sets[self.col].append(self.stack[-1])
                self.stack.pop()
            self.stack.pop()
            self.col += 1
        return

    def run_tarjan(self):
        for i in self.node:
            if i not in self.dfn:
                self.dfs(i)


def load_dataset(out_dir):
    tokenizer = tokenizers.Tokenizer.from_file("tokenizer.json")
    gfa_file = 'hprc-v1.0-minigraph-grch38.gfa'

    edge_from = []
    edge_to = []

    seqs = {}  # seqs['s1'] = 'AGAT'
    seqs_with_op = {}  # seqs_with_op['s1+'] = 'AGAT'
    edge_with_op = {}  # edge['s1+'] = ['s2+','s3+']

    with open(gfa_file) as f:
        for line in f:
            line = line.strip().split()
            if line[0] == 'S':
                seqs[line[1]] = line[2]
            elif line[0] == 'L':
                s = line[1] + line[2]
                t = line[3] + line[4]
                edge_from.append(s)
                edge_to.append(t)
                for i in range(2):
                    name, op = line[i * 2 + 1], line[i * 2 + 2]
                    if name + op not in seqs_with_op:  # 没有出现过 (eg.'s1+')
                        if name not in seqs:
                            print("Error! Invalid data!")
                            return
                        seq = seqs[name]
                        seq = re.sub(r'N+', 'N', seq)  # 使用正则表达式替换连续出现的N为一个N
                        seq = re.sub(r'[^ATCGN]', '', seq)  # 使用正则表达式去除除了ATCGN以外的其他碱基
                        if op == '-':
                            translation_table = str.maketrans('ATCG', 'TAGC')
                            seq = seq.translate(translation_table)[::-1]  # ATCG -> CGAT
                        seqs_with_op[name + op] = seq
                        edge_with_op[name + op] = []

            else:
                print("Error! Invalid data!")
                return

    union = UnionSet(seqs_with_op)  # 并查集

    for i, j in zip(edge_from, edge_to):
        edge_with_op[i].append(j)
        union.unite(i, j)

    tarjan_obj = Tarjan(seqs_with_op, edge_with_op)
    tarjan_obj.run_tarjan()

    edge_col = [[] for i in range(tarjan_obj.col)]  # 邻接表
    degree = [0 for i in range(tarjan_obj.col)]  # 入度

    for i in seqs_with_op:
        for nxt in edge_with_op[i]:
            if tarjan_obj.color[i] != tarjan_obj.color[nxt]:
                edge_col[tarjan_obj.color[i]].append(tarjan_obj.color[nxt])
                degree[tarjan_obj.color[nxt]] += 1

    que = Queue()
    topological_nodes = []  # 拓扑序

    for node_col in range(tarjan_obj.col):  # 入度为0
        if degree[node_col] == 0:
            que.put(node_col)

    while not que.empty():  # 拓扑序
        node_col = que.get()
        topological_nodes.append(node_col)
        for nxt_col in edge_col[node_col]:
            degree[nxt_col] -= 1
            if degree[nxt_col] == 0:
                que.put(nxt_col)

    print('topological seqs num: ' + str(len(topological_nodes)))

    label = []
    dep = []
    root = []
    pre_max_dep = [0 for i in range(tarjan_obj.col)]

    for id_, node_col in enumerate(topological_nodes):
        token_number = 0  # 这个强连通分量token总数
        for j in range(len(tarjan_obj.sets[node_col])):  # 枚举拓扑序节点

            name_with_op = tarjan_obj.sets[node_col][j]  # 枚举该强连通分量的每个序列
            root_of_seq = union.find(name_with_op)  # 该序列所在的连通图
            node_feat = tokenizer.encode(seqs_with_op[name_with_op], add_special_tokens=False).ids  # 该序列分解得到的token

            for val in node_feat:
                label.append(val)  # 每一个token
                root.append(root_of_seq)

            for i in range(len(node_feat)):
                dep.append(pre_max_dep[node_col] + token_number + i)
            token_number += len(node_feat)

        for nxt_col in edge_col[node_col]:
            pre_max_dep[nxt_col] = max(pre_max_dep[nxt_col], pre_max_dep[node_col] + token_number)

        print('\r' + str(id_ + 1) + '/' + str(len(topological_nodes)), end='')

    n = len(label)

    print('\nSort start')
    # 优先使用subgraph_id排序，接着使用dep排序
    idx = torch.tensor(sorted(range(n), key=lambda tmp: (root[tmp], dep[tmp])), dtype=torch.long)
    print('Sort end')

    with open(f"{out_dir}label.txt", 'w') as out_label_file, \
            open(f"{out_dir}pos.txt", 'w') as out_pos_file, \
            open(f"{out_dir}root.txt", 'w') as out_root_file:

        for i in range(n):
            out_label_file.write(str(label[idx[i]])+'\n')
            out_pos_file.write(str(dep[idx[i]])+'\n')
            out_root_file.write(str(root[idx[i]])+'\n')
            if i % 2000000 == 0:
                print('\r' + str(i + 1) + '/' + str(n), end='')


if __name__ == "__main__":
    load_dataset('../data/pretrain/')
