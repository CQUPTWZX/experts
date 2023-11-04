from transformers import AutoTokenizer
from fairseq.data import data_utils
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model.optim import ScheduledOptim, Adam
from tqdm import tqdm
import argparse
import os
from eval import evaluate
from model.contrast import ContrastModel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils


class BertDataset(Dataset):
    def __init__(self, max_token=512, device='cpu', pad_idx=0, data_path=None):
        self.device = device
        super(BertDataset, self).__init__()
        self.data = data_utils.load_indexed_dataset(
            data_path + '/tok', None, 'mmap'
        )
        self.labels = data_utils.load_indexed_dataset(
            data_path + '/Y', None, 'mmap'
        )
        self.max_token = max_token
        self.pad_idx = pad_idx

    def __getitem__(self, item):
        data = self.data[item][:self.max_token - 2].to(
            self.device)
        labels = self.labels[item].to(self.device)
        return {'data': data, 'label': labels, 'idx': item, }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            return batch['data'], batch['label'], batch['idx']
        label = torch.stack([b['label'] for b in batch], dim=0)
        data = torch.full([len(batch), self.max_token], self.pad_idx, device=label.device, dtype=batch[0]['data'].dtype)
        idx = [b['idx'] for b in batch]
        for i, b in enumerate(batch):
            data[i][:len(b['data'])] = b['data']
        return data, label, idx


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='WebOfScience', choices=['WebOfScience', 'nyt', 'rcv1'],
                    help='Dataset.')
parser.add_argument('--batch', type=int, default=4, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=10, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast', default=1, type=int, help='Whether use contrastive model.')
parser.add_argument('--graph', default=1, type=int, help='Whether use graph encoder.')
parser.add_argument('--layer', default=1, type=int, help='Layer of Graphormer.')
parser.add_argument('--multi', default=True, action='store_false',
                    help='Whether the task is multi-label classification.')
parser.add_argument('--lamb', default=1, type=float, help='lambda')
parser.add_argument('--thre', default=0.02, type=float,
                    help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=3, type=int, help='Random seed.')
parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging.')
parser.add_argument('--experts', type=int, default=3, help='Number of experts')
parser.add_argument('--ta', default=0.5, type=float,
                    help='If prefix weight ≤ tau , the loss of expert m on the sample will be eliminated.')
parser.add_argument('--eta', default=0.91, type=float,
                    help='eta is a temperature factor that adjusts the sensitivity of prefix weights.')

def get_root(path_dict, n):
    ret = []
    while path_dict[n] != n:
        ret.append(n)
        n = path_dict[n]
    ret.append(n)
    return ret


def get_final_output(x, y):  # 计算最终的输出
    index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    index.scatter_(1, y.data.view(-1, 1), 1)  # 根据标签y创建一个与输入x形状相同的索引张量index
    index_float = index.float()
    batch_m = torch.matmul(m_list[None, :],
                           index_float.transpose(0, 1))  # self.m_list与index_float的转置相乘，m_list是一个包含了每个类别的m值的张量
    batch_m = batch_m.view((-1, 1))
    x_m = x - batch_m  # 如果样本的标签与输出的标签匹配，就会减去相应的m值。这样做的目的是减小预测概率与标签概率之间的差距，增强模型对正确分类的自信度
    return torch.exp(
        torch.where(index, x_m, x))  # torch.where根据index判断是否应用减去m值的调整，如果是，则返回减去调整后的值；然后使用torch.exp将输出转换为概率值


def to(self, device):
    super().to(device)
    self.m_list = self.m_list.to(device)
    if self.per_cls_weights_enabled is not None:
        self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
    if self.per_cls_weights_enabled_diversity is not None:
        self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
    return self


def _hook_before_epoch(self, epoch):
    if self.reweight_epoch != -1:
        epoch = epoch
        if epoch > self.reweight_epoch:  # 如果当前时期大于reweight_epoch，则进入重新加权阶段
            self.per_cls_weights_base = self.per_cls_weights_enabled
            self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
        else:
            self.per_cls_weights_base = None
            self.per_cls_weights_diversity = None


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly = True
    args = parser.parse_args()
    device = args.device
    print(args)
    if args.wandb:
        import wandb
        wandb.init(config=args, project='htc')  # 调用 wandb.init 初始化一个新的运行配置
    utils.seed_torch(args.seed)  # 设置随机种子
    args.name = args.data + '-' + args.name
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_path = os.path.join('data', args.data)
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)
    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)

    ta = args.ta
    eta = args.eta
    reweight_temperature = eta
    annealing = 500
    experts = args.experts
    print(experts)

    # 创建一个字典用于存储每个类别的样本数量
    class_count = {}

    # 遍历数据集中的样本
    for item in dataset:
        label = item['label'].tolist()
        for i, element in enumerate(label):
            if i not in class_count:
                class_count[i] = {}

            if element not in class_count[i]:
                class_count[i][element] = 1
            else:
                class_count[i][element] += 1
    # 将字典转换为列表，并按类别顺序排序
    class_count_list = []
    for class_dict in class_count.values():
        counts = list(class_dict.values())
        class_count_list.append(counts)
    class_count_list = [class_count[i][1] for i in sorted(class_count.keys())]

    # 进行后续操作，如计算 m_list
    reweight_epoch = -2
    reweight_factor = 0.05
    m_list = 1. / np.sqrt(np.sqrt(class_count_list))
    max_m = 0.5
    m_list = m_list * (max_m / np.max(m_list))
    m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
    m_list = m_list.to(device)

    if reweight_epoch != -1:  # 表示需要在指定的epoch后进行权重调整
        idx = 1
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], class_count_list)  # effective_num每个类别的有效样本数，betas是表示权重调整的参数的列表
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)  # per_cls_weights表示每个类别的权重
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_count_list)
        per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)

    else:
        per_cls_weights_enabled = None
    per_cls_weights_enabled = per_cls_weights_enabled.to(device)
    class_count_list = np.array(class_count_list) / np.sum(class_count_list)  # 相对权重

    C = len(class_count_list)

    per_cls_weights = C * class_count_list * reweight_factor + 1 - reweight_factor
    per_cls_weights = per_cls_weights / np.max(per_cls_weights)  # 将权重缩放到范围[0, 1]，确保最大权重为1，确保最重要的类别具有最大的权重，以便在损失计算中更加重视。
    T = (reweight_epoch + annealing) / reweight_factor  # 控制在训练过程中逐渐增加多样性权重的程度
    # save diversity per_cls_weights
    per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).to(
        device)  # 保存多样性的张量
    per_cls_weights_diversity = per_cls_weights_enabled_diversity

    if reweight_epoch != -1:
        per_cls_weights_base = per_cls_weights_enabled
        # per_cls_weights_diversity = per_cls_weights_enabled_diversity
    else:
        per_cls_weights_base = None
        # per_cls_weights_diversity = None

    # 创建一个列表来存储模型、优化器和对应的Saver
    models = []
    optimizers = []
    savers = []

    # 循环创建模型、优化器和Saver
    for i in range(experts):  # 创建三个模型，可以根据需要更改数量
        model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                              contrast_loss=args.contrast, graph=args.graph,
                                              layer=args.layer, data_path=data_path, multi_label=args.multi,
                                              lamb=args.lamb, threshold=args.thre, tau=args.tau).to(device)
        models.append(model)

        if args.warmup > 0:
            optimizer = ScheduledOptim(Adam(model.parameters(),
                                            lr=args.lr), args.lr,
                                       n_warmup_steps=args.warmup)
        else:
            optimizer = Adam(model.parameters(),
                             lr=args.lr)
        optimizers.append(optimizer)

        saver = Saver(model, optimizer, None, args)
        savers.append(saver)

    # 循环监视模型
    if args.wandb:
        for model in models:
            wandb.watch(model)

    split = torch.load(os.path.join(data_path, 'split.pt'))  # 训练集和验证集的划分
    train = Subset(dataset, split['train'])
    dev = Subset(dataset, split['val'])

    train = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn)  # train是数据集对象
    dev = DataLoader(dev, batch_size=args.batch, shuffle=False, collate_fn=dataset.collate_fn)

    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    log_file = open(os.path.join('checkpoints', args.name, 'log.txt'), 'w')

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:  # 记录连续没有性能改进的轮次数
            print("Early stop!")
            break

        # 设置模型的训练状态
        for model in models:
            model.train()

        i = 0
        loss = 0
        # Train
        pbar = tqdm(train)
        for data, label, idx in pbar:
            outputs = []
            padding_mask = data != tokenizer.pad_token_id
            # 循环遍历模型列表并计算输出
            for model in models:
                output = model(data, padding_mask, labels=label, return_dict=True)
                outputs.append(output)
            xis = [None] * len(outputs)
            # evidential
            for i in range(len(outputs)):
                xis[i] = outputs[i]['logits']
            num_classes = outputs[0]['num_labels']
            w = [torch.ones(len(xis[0]), dtype=torch.bool, device=xis[0].device)]
            b0 = None

            for xi in xis:
                alpha = torch.exp(xi) + 1
                S = alpha.sum(dim=1, keepdim=True)
                b = (alpha - 1) / S
                u = num_classes / S.squeeze(-1)

                if b0 is None:
                    C = 0
                else:
                    bb = b0.view(-1, b0.shape[1], 1) @ b.view(-1, 1, b.shape[1])
                    C = bb.sum(dim=[1, 2]) - bb.diagonal(dim1=1, dim2=2).sum(dim=1)
                b0 = b
                w.append(w[-1] * u / (1 - C))  # 前置权重

            # dynamic reweighting
            exp_w = [torch.exp(wi / eta) for wi in w]
            exp_w = exp_w[:-1]
            exp_w_sum = sum(exp_w)
            normalized_list = [x / exp_w_sum for x in exp_w]
            exp_w = normalized_list
            exp_w = [wi.unsqueeze(-1) for wi in exp_w]
            reweighted_outs = []
            for i in range(len(xis)):
                reweighted_outs.append(xis[i] * exp_w[i])
            xi = torch.mean(torch.stack(reweighted_outs), dim=0)

            yi = outputs[0]['labels']
            # 将独热编码还原为标签
            y = torch.argmax(yi, dim=1)

            # 创建空列表来存储l和kl值
            l_values = []
            kl_values = []

            for i in range(len(xis)):
                alpha = get_final_output(xis[i], y)  # 获取模型输出结果alpha
                S = alpha.sum(dim=1, keepdim=True)

                l = F.nll_loss(torch.log(alpha) - torch.log(S), y, weight=per_cls_weights_enabled,
                               reduction="none")  # 使用负对数似然损失函数

                # KL adjusted parameters of D(p|alpha)
                alpha_tilde = yi + (1 - yi) * (alpha + 1)
                S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

                kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(alpha_tilde.shape[1])) - torch.lgamma(
                    alpha_tilde).sum(dim=1, keepdim=True) \
                     + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1,
                                                                                                       keepdim=True)

                l += epoch / T * kl.squeeze(-1)

                l_values.append(l)
                kl_values.append(kl)
            # diversity
            # if per_cls_weights_diversity is not None:
            #     diversity_temperature = per_cls_weights_diversity.view((1, -1))
            #     temperature_mean = diversity_temperature.mean().item()
            # else:
            #     diversity_temperature = 1
            #     temperature_mean = 1
            # output_dist = F.log_softmax(xi1 / diversity_temperature, dim=1)  # 输出分布
            # with torch.no_grad():
            #     x = (xi1 + xi2)/2
            #     mean_output_dist = F.softmax(x / diversity_temperature, dim=1)  # 平均输出分布
            # l1 -= 0.01 * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist,reduction="none").sum(dim=1)



            w = w[:-1]
            wmax = [0] * 5
            batch_num_elements = w[0].numel()
            for i in range(batch_num_elements):
                wmax[i] = max(tensor[i].item() for tensor in w)

            # 循环处理每个 w 和 l
            for i in range(len(w)):
                # 归一化处理
                for j in range(batch_num_elements):
                    w[i][j] = w[i][j] / wmax[j]  # 归一化处理，将范围缩放到0到1之间
                # 使用阈值进行筛选
                w[i] = torch.where(w[i] > ta, True, False)
                # 计算 l
                if w[i].sum() == 0:
                    l_values[i] = 0
                else:
                    l_values[i] = (w[i] * l_values[i]).sum() / w[i].sum()

            loss /= args.update  # 将累计的损失值除以args.update，即梯度累积的步数，以获得平均损失
            outputloss = 0
            for i in range(len(outputs)):
                outputloss = outputs[i]['loss'] + l_values[i] + outputloss
            outputloss.backward()  # 执行反向传播计算梯度
            loss += outputloss.item()  # 累加当前批次的损失值
            i += 1
            if i % args.update == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                if args.wandb:
                    wandb.log({'train_loss': loss})
                pbar.set_description('loss:{:.4f}'.format(loss))
                i = 0
                loss = 0
                # torch.cuda.empty_cache()
        pbar.close()

        # 设置模型的状态
        for model in models:
            model.eval()

        pbar = tqdm(dev)
        with torch.no_grad():  # 上下文管理器，关闭梯度计算，以减少内存消耗并加快推理速度
            truth = []
            pred = []
            for data, label, idx in pbar:
                outputs = []
                padding_mask = data != tokenizer.pad_token_id
                # 循环遍历模型列表并计算输出
                for model in models:
                    output = model(data, padding_mask, labels=label, return_dict=True)
                    outputs.append(output)
                xis = [None] * len(outputs)
                # evidential
                for i in range(len(outputs)):
                    xis[i] = outputs[i]['logits']
                num_classes = outputs[0]['num_labels']
                w = [torch.ones(len(xis[0]), dtype=torch.bool, device=xis[0].device)]
                b0 = None

                for xi in xis:
                    alpha = torch.exp(xi) + 1
                    S = alpha.sum(dim=1, keepdim=True)
                    b = (alpha - 1) / S
                    u = num_classes / S.squeeze(-1)

                    if b0 is None:
                        C = 0
                    else:
                        bb = b0.view(-1, b0.shape[1], 1) @ b.view(-1, 1, b.shape[1])
                        C = bb.sum(dim=[1, 2]) - bb.diagonal(dim1=1, dim2=2).sum(dim=1)
                    b0 = b
                    w.append(w[-1] * u / (1 - C))  # 前置权重

                # dynamic reweighting
                exp_w = [torch.exp(wi / eta) for wi in w]
                exp_w = exp_w[:-1]
                exp_w_sum = sum(exp_w)
                normalized_list = [x / exp_w_sum for x in exp_w]
                exp_w = normalized_list
                exp_w = [wi.unsqueeze(-1) for wi in exp_w]
                reweighted_outs = []
                for i in range(len(xis)):
                    reweighted_outs.append(xis[i] * exp_w[i])
                xi = torch.mean(torch.stack(reweighted_outs), dim=0)
                for l in label:
                    t = []  # 存储真实标签为1
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)

                for l in xi:
                    pred.append(torch.sigmoid(l).tolist())

        pbar.close()
        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        print('macro', macro_f1, 'micro', micro_f1)
        print('macro', macro_f1, 'micro', micro_f1, file=log_file)
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1, 'best_macro': best_score_macro,
                       'best_micro': best_score_micro})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            for saver in savers:
                extra = f'_macro{i}'  # 根据索引添加后缀
                checkpoint_path = os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt')
                saver(macro_f1, best_score_macro, checkpoint_path)

            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            for saver in savers:
                extra = f'_micro{i}'  # 根据索引添加后缀
                checkpoint_path = os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt')
                saver(micro_f1, best_score_micro, checkpoint_path)

            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.name, 'checkpoint_{:d}.pt'.format(epoch)))
        # save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
    log_file.close()
