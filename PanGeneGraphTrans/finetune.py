import time
import argparse
import random
import numpy as np
import torch
from parse_finetune import parse_finetune_method, parser_finetune_add_main_args
from dataset_finetune import load_finetune_dataset
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.utils.data import DataLoader
import os
import warnings

gpus = [0]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Parse args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_finetune_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load method
model = parse_finetune_method(args)
model = torch.nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

# print('MODEL:', model)
total = sum([param.nelement() for param in model.module.parameters()])
print('Model Size:', total)

# load dataset
trainDataset = load_finetune_dataset(args.data_dir + 'train.csv')
devDataset = load_finetune_dataset(args.data_dir + 'dev.csv')
testDataset = load_finetune_dataset(args.data_dir + 'test.csv')
print("running")
trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
devLoader = DataLoader(devDataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

warmup_steps = 50
warmup_lr_init = 0
warmup_lr_end = args.lr
optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], weight_decay=args.weight_decay,
                              lr=args.lr)


def rule(step):
    if step < warmup_steps:
        lamda = warmup_lr_init + (warmup_lr_end - warmup_lr_init) * step / warmup_steps
    else:
        lamda = warmup_lr_end
    return lamda / args.lr


warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule, last_epoch=-1)


# Training loop
def eval_model(dataLoader):
    eval_since_begin = time.time()
    model.eval()
    eval_loss_sum = 0

    out_tot = torch.tensor([]).cpu()
    label_tot = torch.tensor([]).cpu()

    with torch.no_grad():
        for eval_i, (eval_graph, eval_label) in enumerate(dataLoader):
            eval_since_begin_batch = time.time()

            eval_outputs = model(**eval_graph, labels=eval_label)

            print('\r', f'  ----Eval,'
                        f'{eval_i + 1}/{len(dataLoader)}, '
                        f'Loss: {eval_outputs.loss.mean().item():.8f}, '
                        f'time: {time.time() - eval_since_begin_batch:.2f} ', end='')

            eval_loss_sum += eval_outputs.loss.mean().item()
            eval_label = eval_label.long().detach().cpu()
            label_tot = torch.cat((label_tot, eval_label), dim=0)

            eval_out = F.log_softmax(eval_outputs.logits, dim=1)
            eval_out = eval_out.argmax(dim=-1, keepdim=True).reshape(-1, ).cpu()
            out_tot = torch.cat((out_tot, eval_out), dim=0)

        if args.metric == 'mcc':
            result = matthews_corrcoef(label_tot, out_tot)
        elif args.metric == 'f1':
            result = f1_score(label_tot, out_tot, average="macro", zero_division=0)
        else:
            raise ValueError('Invalid metric')

        print(f'\n  ----End of the eval:'
              f'\033[1;31;40mLoss_average: {eval_loss_sum / len(dataLoader):.8f}\033[0m'
              f','
              f'\033[1;31;40m{args.metric}: {result * 100:.3f}%\033[0m'
              f',Eval time: {time.time() - eval_since_begin:.2f} ')


step_tot = 0
loss_sum = 0
since_begin_part = time.time()
for epoch in range(args.epochs):

    accumulation_steps = args.accumulation_steps  # 设置梯度累积的步数

    for i, (graph, label) in enumerate(trainLoader):
        model.train()
        since_begin_batch = time.time()
        loss = model(**graph, labels=label).loss.mean()

        # 计算累积梯度
        loss /= accumulation_steps
        loss_sum += loss.item()
        loss.backward()

        # 在指定的累积步数之后进行优化步骤
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler.step()

        print('\r', f'Epoch: {epoch}, '
                    f'{i + 1}/{len(trainLoader)}, '
                    f'Loss: {loss.item() * accumulation_steps:.8f}, '
                    f'time: {time.time() - since_begin_batch:.2f}, '
                    f'LR= {lr:.9f} ', end='')

        step_tot += 1
        if step_tot % args.eval_step == 0:
            print('\n----step' + str(step_tot))
            print(f'----Part of the Epoch: {epoch}, '
                  f'\033[1;31;40mLoss_average: {loss_sum / args.eval_step * accumulation_steps:.8f}\033[0m'
                  f',Time: {time.time() - since_begin_part:.2f} ')

            eval_model(devLoader)
            eval_model(testLoader)
            since_begin_part = time.time()
            loss_sum = 0

    if i % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

print('\n----step' + str(step_tot))
eval_model(devLoader)
eval_model(testLoader)
