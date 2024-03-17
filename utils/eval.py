from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.tools import data_2_device, is_number


def calc_auc(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    return auc(fpr, tpr)


def sampled_f1_score(
        actual: List[List[int]],
        pred: List[List[int]]
):
    # converting the multi-label classification to a binary output
    mlb = MultiLabelBinarizer()
    actual = mlb.fit_transform(actual)
    pred = mlb.fit_transform(pred)
    f1 = f1_score(actual, pred, average="samples")
    return f1


def precision_k(
        actual: List[int],
        pred: List[int],
        k: int
):
    if k == 0:
        return 0
    actual_set = set(actual)
    pred_set = set(pred[:k])
    # 求预测值与真实值得交集
    common_values = actual_set.intersection(pred_set)
    return len(common_values) / len(pred[:k])


def average_precision_k(
        actual: List[int],
        pred: List[int],
        k: int
):
    precision_ = []
    for i in range(1, k + 1):
        precision_.append(precision_k(actual, pred, i))
    # return 0 if there are no values in the list
    if len(precision_) == 0:
        return 0
    return np.mean(precision_)


def mean_avg_precision_k(
        actual: List[List[int]],
        pred: List[List[int]],
        k: int
):
    average_precision = []
    for i in range(len(actual)):
        ap = average_precision_k(actual[i], pred[i], k)
        average_precision.append(ap)
    return np.mean(average_precision)


@torch.no_grad()
def mcwf_eval_fn(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    predictions, ground_truths = [], []
    total_loss, n_samples = 0, 0

    for batch in tqdm(dataloader, total=len(dataloader), desc='validating', dynamic_ncols=True):
        output = model(**data_2_device(batch, device))  # (bs, 2)
        n_samples += batch['batch_size']

        predict_labels = torch.argmax(output['logits'], dim=1).tolist()

        total_loss += output['loss'] * batch['batch_size']
        predictions.extend(predict_labels)
        ground_truths.extend(batch['label'].tolist())

    mean_loss = total_loss / n_samples
    auc_score = calc_auc(ground_truths, predictions)
    acc = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)
    p = precision_score(ground_truths, predictions)
    r = recall_score(ground_truths, predictions)
    eval_result = {
        'loss': mean_loss,
        'auc': auc_score,
        'acc': acc,
        'f1': f1,
        'precision': p,
        'recall': r
    }
    print('\n', eval_result)
    return eval_result


@torch.no_grad()
def misa_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    predict_list: List[int] = []
    ground_list: List[int] = []

    loss, cls, diff, cmd, recon = 0, 0, 0, 0, 0
    total_samples = 0

    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc="评估..."):
        output: dict = model(**data_2_device(batch, device))

        total_samples += batch['batch_size']

        predict_list.extend(output['logits'].argmax(dim=1).tolist())
        ground_list.extend(batch['label'].tolist())

        loss += output['loss'] * batch['batch_size']
        cls += output['cls'] * batch['batch_size']
        diff += output['diff'] * batch['batch_size']
        cmd += output['cmd'] * batch['batch_size']
        recon += output['recon'] * batch['batch_size']

    f1 = f1_score(ground_list, predict_list)
    acc = accuracy_score(ground_list, predict_list)
    p = precision_score(ground_list, predict_list)
    r = recall_score(ground_list, predict_list)
    auc = calc_auc(ground_list, predict_list)

    eval_result = {
        'loss': loss / total_samples,
        'cls': cls / total_samples,
        'diff': diff / total_samples,
        'cmd': cmd / total_samples,
        'recon': recon / total_samples,
        'f1': f1,
        'acc': acc,
        'recall': r,
        'auc': auc,
        'precision': p
    }
    print('\n', eval_result)
    return eval_result


@torch.no_grad()
def cubemlp_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    return mcwf_eval_fn(model, dataloader, device)


@torch.no_grad()
def unimse_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    predict = []
    truth = []
    total_samples = 0
    total_loss = 0

    for batch in tqdm(dataloader, dynamic_ncols=True, total=len(dataloader), desc=f'validating'):
        total_samples += batch['batch_size']
        outputs = model(**data_2_device(batch, device))

        loss = outputs["loss"]
        total_loss += loss.item()

        output_ids = model.generate(**data_2_device(batch, device))
        pred_token: List[str] = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        truth.extend(batch['raw_label'])
        for token in pred_token:
            if is_number(token):
                predict.append(int(token))
            else:
                predict.append(token)

    return {
        'loss': total_loss / total_samples,
        'prediction': predict,
        'truth': truth
    }


@torch.no_grad()
def dip_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    total_samples = 0
    total_loss = 0
    predictions = []
    ground_truths = []

    for batch in tqdm(dataloader, dynamic_ncols=True, total=len(dataloader), desc='Evaluating'):
        outputs = model(**data_2_device(batch, device))
        # (bs, )
        binary_output = outputs['logits'].detach().sigmoid().flatten()

        total_samples += batch['batch_size']
        total_loss += outputs['loss'].item() * batch['batch_size']
        # 然后将 sigmoid 输出大于 0.5 的位置设置为 1，小于 0.5 的位置设置为 0
        binary_output[binary_output >= 0.5] = 1
        binary_output[binary_output < 0.5] = 0

        predictions.extend(binary_output.tolist())
        ground_truths.extend(batch['label'].tolist())

    mean_loss = total_loss / total_samples
    auc_score = calc_auc(ground_truths, predictions)
    acc = accuracy_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)
    p = precision_score(ground_truths, predictions)
    r = recall_score(ground_truths, predictions)
    eval_result = {
        'loss': mean_loss,
        'auc': auc_score,
        'acc': acc,
        'f1': f1,
        'precision': p,
        'recall': r
    }
    print('\n', eval_result)
    return eval_result


@torch.no_grad()
def sks_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    total_samples = 0
    total_loss, total_sarcasm_loss, total_humor_loss = 0, 0, 0

    humor_predict, humor_label = [], []
    sarcasm_predict, sarcasm_label = [], []

    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc='validating'):
        outputs = model(**data_2_device(batch, device))

        total_samples += batch['batch_size']

        total_loss += outputs['loss'].item() * batch['batch_size']
        total_humor_loss += outputs['humor_loss'].item() * batch['batch_size']
        total_sarcasm_loss += outputs['sarcasm_loss'].item() * batch['batch_size']

        humor_predict.extend(outputs['humor_predict'])
        humor_label.extend(outputs['humor_label'])

        sarcasm_predict.extend(outputs['sarcasm_predict'])
        sarcasm_label.extend(outputs['sarcasm_label'])

    loss = total_loss / total_samples
    sarcasm_loss = total_sarcasm_loss / total_samples
    humor_loss = total_humor_loss / total_samples
    # 单独 humor
    humor_f1 = f1_score(humor_label, humor_predict)
    humor_recall = recall_score(humor_label, humor_predict)
    humor_auc = calc_auc(humor_label, humor_predict)
    humor_precision = precision_score(humor_label, humor_predict)
    humor_acc = accuracy_score(humor_label, humor_predict)
    # 单独 sarcasm
    sarcasm_f1 = f1_score(sarcasm_label, sarcasm_predict)
    sarcasm_recall = recall_score(sarcasm_label, sarcasm_predict)
    sarcasm_auc = calc_auc(sarcasm_label, sarcasm_predict)
    sarcasm_precision = precision_score(sarcasm_label, sarcasm_predict)
    sarcasm_acc = accuracy_score(sarcasm_label, sarcasm_predict)
    # 整体
    all_f1 = f1_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_recall = recall_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_auc = calc_auc(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_precision = precision_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_acc = accuracy_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)

    eval_result = {
        'loss': loss,
        'humor_loss': humor_loss,
        'sarcasm_loss': sarcasm_loss,

        'all_f1': all_f1,
        'all_recall': all_recall,
        'all_auc': all_auc,
        'all_precision': all_precision,
        'all_acc': all_acc,

        'humor_f1': humor_f1,
        'humor_recall': humor_recall,
        'humor_auc': humor_auc,
        'humor_precision': humor_precision,
        'humor_acc': humor_acc,

        'sarcasm_f1': sarcasm_f1,
        'sarcasm_recall': sarcasm_recall,
        'sarcasm_auc': sarcasm_auc,
        'sarcasm_precision': sarcasm_precision,
        'sarcasm_acc': sarcasm_acc
    }
    print(eval_result)
    return eval_result


@torch.no_grad()
def mcwfml_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    total_samples = 0
    total_loss, total_sarcasm_loss, total_humor_loss = 0, 0, 0

    humor_predict, humor_label = [], []
    sarcasm_predict, sarcasm_label = [], []

    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc='validating'):
        outputs = model(**data_2_device(batch, device))

        total_samples += batch['batch_size']

        total_loss += outputs['loss'].item() * batch['batch_size']
        total_humor_loss += outputs['humor_loss'].item() * batch['batch_size']
        total_sarcasm_loss += outputs['sarcasm_loss'].item() * batch['batch_size']

        humor_predict.extend(outputs['humor_predict'])
        humor_label.extend(outputs['humor_label'])

        sarcasm_predict.extend(outputs['sarcasm_predict'])
        sarcasm_label.extend(outputs['sarcasm_label'])

    loss = total_loss / total_samples
    sarcasm_loss = total_sarcasm_loss / total_samples
    humor_loss = total_humor_loss / total_samples
    # 单独 humor
    humor_f1 = f1_score(humor_label, humor_predict)
    humor_recall = recall_score(humor_label, humor_predict)
    humor_auc = calc_auc(humor_label, humor_predict)
    humor_precision = precision_score(humor_label, humor_predict)
    humor_acc = accuracy_score(humor_label, humor_predict)
    # 单独 sarcasm
    sarcasm_f1 = f1_score(sarcasm_label, sarcasm_predict)
    sarcasm_recall = recall_score(sarcasm_label, sarcasm_predict)
    sarcasm_auc = calc_auc(sarcasm_label, sarcasm_predict)
    sarcasm_precision = precision_score(sarcasm_label, sarcasm_predict)
    sarcasm_acc = accuracy_score(sarcasm_label, sarcasm_predict)
    # 整体
    all_f1 = f1_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_recall = recall_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_auc = calc_auc(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_precision = precision_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_acc = accuracy_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)

    eval_result = {
        'loss': loss,
        'humor_loss': humor_loss,
        'sarcasm_loss': sarcasm_loss,

        'all_f1': all_f1,
        'all_recall': all_recall,
        'all_auc': all_auc,
        'all_precision': all_precision,
        'all_acc': all_acc,

        'humor_f1': humor_f1,
        'humor_recall': humor_recall,
        'humor_auc': humor_auc,
        'humor_precision': humor_precision,
        'humor_acc': humor_acc,

        'sarcasm_f1': sarcasm_f1,
        'sarcasm_recall': sarcasm_recall,
        'sarcasm_auc': sarcasm_auc,
        'sarcasm_precision': sarcasm_precision,
        'sarcasm_acc': sarcasm_acc
    }
    print(eval_result)
    return eval_result


@torch.no_grad()
def cubemlpml_eval_fn(model: nn.Module, dataloader: DataLoader, device: str):
    return mcwfml_eval_fn(model, dataloader, device)


@torch.no_grad()
def dipmlpml_eval_fn(model: nn, dataloader: DataLoader, device: str):
    return mcwfml_eval_fn(model, dataloader, device)


@torch.no_grad()
def misaml_eval_fn(model: nn, dataloader: DataLoader, device: str):
    total_samples = 0
    total_sarcasm_loss, total_humor_loss = 0, 0
    total_loss, total_cls, total_diff, total_cmd, total_recon = 0, 0, 0, 0, 0

    humor_predict, humor_label = [], []
    sarcasm_predict, sarcasm_label = [], []

    for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc="评估..."):
        output: dict = model(**data_2_device(batch, device))

        outputs = model(**data_2_device(batch, device))

        total_samples += batch['batch_size']

        total_loss += outputs['loss'].item() * batch['batch_size']
        total_humor_loss += outputs['humor_loss'].item() * batch['batch_size']
        total_sarcasm_loss += outputs['sarcasm_loss'].item() * batch['batch_size']

        total_cls += output['cls'] * batch['batch_size']
        total_diff += output['diff'] * batch['batch_size']
        total_cmd += output['cmd'] * batch['batch_size']
        total_recon += output['recon'] * batch['batch_size']

        humor_predict.extend(outputs['humor_predict'])
        humor_label.extend(outputs['humor_label'])

        sarcasm_predict.extend(outputs['sarcasm_predict'])
        sarcasm_label.extend(outputs['sarcasm_label'])

    loss = total_loss / total_samples
    cls = total_cls / total_samples
    diff = total_diff / total_samples
    cmd = total_cmd / total_samples
    recon = total_recon / total_samples
    sarcasm_loss = total_sarcasm_loss / total_samples
    humor_loss = total_humor_loss / total_samples
    # 单独 humor
    humor_f1 = f1_score(humor_label, humor_predict)
    humor_recall = recall_score(humor_label, humor_predict)
    humor_auc = calc_auc(humor_label, humor_predict)
    humor_precision = precision_score(humor_label, humor_predict)
    humor_acc = accuracy_score(humor_label, humor_predict)
    # 单独 sarcasm
    sarcasm_f1 = f1_score(sarcasm_label, sarcasm_predict)
    sarcasm_recall = recall_score(sarcasm_label, sarcasm_predict)
    sarcasm_auc = calc_auc(sarcasm_label, sarcasm_predict)
    sarcasm_precision = precision_score(sarcasm_label, sarcasm_predict)
    sarcasm_acc = accuracy_score(sarcasm_label, sarcasm_predict)
    # 整体
    all_f1 = f1_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_recall = recall_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_auc = calc_auc(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_precision = precision_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)
    all_acc = accuracy_score(humor_label + sarcasm_label, humor_predict + sarcasm_predict)

    eval_result = {
        'loss': loss,
        'humor_loss': humor_loss,
        'sarcasm_loss': sarcasm_loss,
        'cls': cls,
        'diff': diff,
        'cmd': cmd,
        'recon': recon,

        'all_f1': all_f1,
        'all_recall': all_recall,
        'all_auc': all_auc,
        'all_precision': all_precision,
        'all_acc': all_acc,

        'humor_f1': humor_f1,
        'humor_recall': humor_recall,
        'humor_auc': humor_auc,
        'humor_precision': humor_precision,
        'humor_acc': humor_acc,

        'sarcasm_f1': sarcasm_f1,
        'sarcasm_recall': sarcasm_recall,
        'sarcasm_auc': sarcasm_auc,
        'sarcasm_precision': sarcasm_precision,
        'sarcasm_acc': sarcasm_acc
    }
    print(eval_result)
    return eval_result


@torch.no_grad()
def mcwfforcl_eval_fn(model: nn, dataloader: DataLoader, device: str):
    ...



@torch.no_grad()
def cubemlpforcl_eval_fn(model: nn, dataloader: DataLoader, device: str):
    ...


@torch.no_grad()
def dipforcl_eval_fn(model: nn, dataloader: DataLoader, device: str):
    ...


@torch.no_grad()
def misaforcl_eval_fn(model: nn, dataloader: DataLoader, device: str):
    ...
