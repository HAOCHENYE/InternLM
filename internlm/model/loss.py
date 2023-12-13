#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
import os
import jsonlines



def _save_loss_info(loss, dataset_path, sample_index):
    file_handlers = {}
    train_folder = os.path.join(os.getenv('TRAIN_FOLDER'), 'cn')
    ckpt_foler = os.getenv('SAVE_CKPT_FOLDER')

    save_root = os.path.join(ckpt_foler, 'analyze_loss')

    rel_path = os.path.relpath(dataset_path, train_folder)
    filename = os.path.splitext(os.path.basename(dataset_path))[0]

    data_rank = gpc.get_local_rank(ParallelMode.DATA)
    pipline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    saved_path = os.path.join(save_root, rel_path, f'dp{data_rank}_pp{pipline_rank}', f'{filename}.jsonl')

    if saved_path not in file_handlers:
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        file_handlers[saved_path] = jsonlines.open(saved_path, mode='w', flush=True)
    
    data_world = gpc.get_world_size(ParallelMode.DATA)
    cur_iter = data_world * int(os.environ['training_steps']) + data_rank
    file_handlers[saved_path].write(
        {'loss': loss.item(), 'index': sample_index, 'dataset_path': dataset_path, 'steps': cur_iter})



class FlashGPTLMLoss(nn.Module):
    """
    Loss function for flash GPT Language Model.
    """

    def __init__(self, parallel_output=True, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None:
            if label_smoothing != 0:
                if gpc.is_rank_for_log():
                    print(f"use label_smoothing: {label_smoothing}")
        else:
            label_smoothing = 0
        self.label_smoothing = label_smoothing

        if parallel_output:
            self.loss_fn = FlashCrossEntropyLoss(
                reduction="mean",
                inplace_backward=True,
                process_group=gpc.get_group(ParallelMode.TENSOR),
                label_smoothing=label_smoothing,
            )  # The loss in this place is bound to the gather_output initialized by VocabParallelClassifier1D
        else:
            # Here, the output will gather output is set in the model, so use ordinary loss
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

    def forward(self, *args, data=None):
        if len(args) == 3:
            # residual is to match prenorm
            logits, _, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        if os.getenv('ANALYZE_LOSS') is not None:
            cu_seqlens = data['cu_seqlens']
            losses = []
            for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                losses.append(self.loss_fn(shift_logits[start:end], shift_labels[start:end]))
            loss = sum(losses) / len(losses)
            _save_loss_info(data, loss)
        else:
            # There is no need to consider the ignore_index problem here, because the loss calculation will be
            # calculated through the calculation range, and -100 must be outside this range, so there is no problem
            loss = self.loss_fn(shift_logits, shift_labels)
        return loss
