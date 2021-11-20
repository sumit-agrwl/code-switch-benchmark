import datetime
import json
import logging
import os
import sys
import time
from copy import deepcopy
from typing import List, Union

class DatasetArgs:
    def __init__(self,
                 dataset_folder: str = None,
                 augment_train_datasets: List[str] = None,
                 **kwargs):
        super(DatasetArgs, self).__init__(**kwargs)

        self.dataset_folder = dataset_folder
        self.augment_train_datasets = augment_train_datasets

        self.validate_dataset_folder()

    def validate_dataset_folder(self):
        assert os.path.exists(self.dataset_folder)


class ModelArgs:
    def __init__(self,
                 pretrained_name_or_path: str,
                 task_type: str = "classification",
                 target_label_field: str = None,
                 max_len: int = 200,
                 prompt_type: str = "soft",
                 n_prompt_tokens: int = 20,
                 init_prompt_from_vocab: bool = False,
                 **kwargs):
        super(ModelArgs, self).__init__(**kwargs)

        self.task_type = task_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.target_label_field = target_label_field
        self.max_len = max_len
        self.prompt_type = prompt_type
        self.n_prompt_tokens = n_prompt_tokens
        self.init_prompt_from_vocab = init_prompt_from_vocab


class TraintestArgs:
    def __init__(self,
                 mode: Union[List[str], str] = None,
                 text_field: str = None,
                 max_epochs: int = 5,
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.01,
                 lr_scheduler_type: str = "linear",
                 num_warmup_steps: int = 0,
                 patience: int = 4,
                 batch_size: int = 16,
                 grad_acc=2,
                 checkpoint_using_accuracy: bool = False,
                 save_errors_path: str = None,
                 eval_ckpt_path: str = None,
                 debug: bool = False,
                 **kwargs):
        super(TraintestArgs, self).__init__(**kwargs)

        if isinstance(mode, str):
            mode = [mode, ]
        self.mode = mode
        self.text_field = text_field or "text"
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        
        self.patience = patience
        self.batch_size = batch_size
        self.grad_acc = grad_acc
        self.checkpoint_using_accuracy = checkpoint_using_accuracy
        self.save_errors_path = save_errors_path
        self.debug = debug
        # evaluation
        self.eval_ckpt_path = eval_ckpt_path

        self.validate_mode()

    def validate_mode(self):
        assert all([m in ["train", "test", "interactive"] for m in self.mode])


class Args(DatasetArgs, ModelArgs, TraintestArgs):

    def __init__(self, **kwargs):
        super(Args, self).__init__(**kwargs)
        print