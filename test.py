from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
from train import BertDataset
from eval import evaluate
from model.contrast import ContrastModel

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--batch', type=int, default=128, help='Batch size.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
extra_choices = ['_macro1', '_micro1', '_macro2', '_micro2']
# 创建参数并添加到一个列表
extra_args = []
experts = 3  # 专家数量
for i in range(experts):
    extra_arg = f'--extra{i+1}'
    extra_choices_arg = f'--extra{i+1}_choices'
    parser.add_argument(extra_arg, default=f'_micro{i+1}', choices=extra_choices, help=f'An extra string in the name of checkpoint.')
    parser.add_argument(extra_choices_arg, choices=extra_choices)
    extra_args.append(extra_arg)
    extra_args.append(extra_choices_arg)
args = parser.parse_args()

if __name__ == '__main__':
    checkpoints = []

    for i in range(1, experts + 1):
        extra = f'_macro{i}' if i % 2 == 1 else f'_micro{i}'
        checkpoint = torch.load(os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt'),
                                map_location='cpu')
        checkpoints.append(checkpoint)
    batch_size = args.batch
    device = args.device
    extra = args.extra1
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    data_path = os.path.join('data', args.data)

    if not hasattr(args, 'graph'):
        args.graph = False
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)

    models = []
    model_checkpoints = []
    for i in range(1, experts + 1):
        extra = f'_macro{i}' if i % 2 == 1 else f'_micro{i}'
        checkpoint = torch.load(os.path.join('checkpoints', args.name, f'checkpoint_best{extra}.pt'),
                                map_location='cpu')
        model_checkpoints.append(checkpoint)

    for i in range(experts):
        checkpoint = model_checkpoints[i]
        model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                              contrast_loss=args.contrast, graph=args.graph,
                                              layer=args.layer, data_path=data_path, multi_label=args.multi,
                                              lamb=args.lamb, threshold=args.thre).to(device)
        model.load_state_dict(checkpoint['param'])
        model.eval()
        models.append(model)

    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []
    reweight_temperature = 0.91
    eta = reweight_temperature
    pbar = tqdm(test)
    with torch.no_grad():
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
    precision = scores['precision']
    recall = scores['recall']
    print('precision', precision, 'recall', recall, 'macro', macro_f1, 'micro', micro_f1)
