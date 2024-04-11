import time
import math
import argparse
import random
import numpy as np
import torch
from dataset import load_dataset, GraphDataset
from parse import parse_method, parser_add_main_args
import os
from torch.utils.data import DataLoader
import warnings
import gc

gpus = [0, 1, 2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Parse args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load method
model = parse_method(args)
model = torch.nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

# print('MODEL:', model)
total = sum([param.nelement() for param in model.module.parameters()])
print('Model Size:', total)

# Load and preprocess data
trainDataset, testDataset = load_dataset(args.data_dir)
trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
testLoader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': args.lr}],
                              weight_decay=args.weight_decay, lr=args.lr)

total_step = args.epochs * len(trainLoader)
warmup_steps = 12 * len(trainLoader)
warmup_lr_init = 0
warmup_lr_end = args.lr


def rule(step):
    step += args.load_epoch * len(trainLoader)
    if step < warmup_steps:
        lamda = warmup_lr_init + (warmup_lr_end - warmup_lr_init) * step / warmup_steps
    else:
        lamda = warmup_lr_end - (warmup_lr_end - warmup_lr_init) * (step - warmup_steps) / (total_step - warmup_steps)
    return lamda / args.lr


warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule, last_epoch=args.load_epoch - 1)

# Training loop
for epoch in range(args.load_epoch, args.epochs):
    since_begin_epoch = time.time()
    model.train()
    loss_sum = 0
    accumulation_steps = args.accumulation_steps  # 设置梯度累积的步数

    for i, (graph, label) in enumerate(trainLoader):

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
                    f'Loss: {accumulation_steps * loss.item():.8f}, '
                    f'time: {time.time() - since_begin_batch:.2f}, '
                    f'LR= {lr:.9f} ', end='')

    if i % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    print(f'\nEnd of the Epoch: {epoch}, '
          f'\033[1;31;40mLoss_average: {accumulation_steps * loss_sum / len(trainLoader):.8f}\033[0m'
          f',Epoch total time: {time.time() - since_begin_epoch:.2f} ')

    # Save model
    if (epoch + 1) % args.save_step == 0:
        param_file = args.model_dir + 'pretrain_params_epoch_' + str(epoch + 1)
        model.module.save_pretrained(param_file)

    # Eval model
    if (epoch + 1) % args.eval_step == 0:
        since_begin_eval = time.time()
        model.eval()

        loss_sum = 0

        with torch.no_grad():
            for i, (graph, label) in enumerate(testLoader):
                since_begin_batch = time.time()

                loss = model(**graph, labels=label).loss.mean()

                loss_sum += loss.item()

                print('\r', f'Eval, '
                            f'{i + 1}/{len(testLoader)}, '
                            f'Loss: {loss.item():.8f}, '
                            f'time: {time.time() - since_begin_batch:.2f} ', end='')

        PPL = math.exp(loss_sum / len(testLoader))

        print(f'\nEnd of the eval, '
              f'\033[1;31;40mPPL: {PPL:.3f}\033[0m'
              f',Eval time: {time.time() - since_begin_eval:.2f} ')

        torch.cuda.empty_cache()
        gc.collect()
