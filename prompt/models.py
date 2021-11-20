import os
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ParameterList, Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel

from datasets import create_path
from helpers import get_model_nparams, merge_subword_encodings_for_sentences, merge_subword_encodings_for_words

""" ############## """
"""     helpers    """
""" ############## """


def load_bert_pretrained_weights(model, load_path, device):
    print(f"\nLoading weights from directory: {load_path}")
    pretrained_dict = torch.load(f"{load_path}/pytorch_model.bin", map_location=torch.device(device))
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    used_dict = {}
    for k, v in model_dict.items():
        if "classifier.weight" in k or "classifier.bias" in k:
            print(f"Ignoring to load '{k}' from custom pretrained model")
            continue
        if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
            used_dict[k] = pretrained_dict[k]
        elif ".".join(k.split(".")[1:]) in pretrained_dict \
                and v.shape == pretrained_dict[".".join(k.split(".")[1:])].shape:
            used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
        elif "bert." + ".".join(k.split(".")[1:]) in pretrained_dict \
                and v.shape == pretrained_dict["bert." + ".".join(k.split(".")[1:])].shape:
            used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[1:])]
        elif "bert." + ".".join(k.split(".")[3:]) in pretrained_dict \
                and v.shape == pretrained_dict["bert." + ".".join(k.split(".")[3:])].shape:
            used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[3:])]
        elif "roberta." + ".".join(k.split(".")[1:]) in pretrained_dict \
                and v.shape == pretrained_dict["roberta." + ".".join(k.split(".")[1:])].shape:
            used_dict[k] = pretrained_dict["roberta." + ".".join(k.split(".")[1:])]
        elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
            used_dict[k] = pretrained_dict["bert." + k]
        elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
            used_dict[k] = pretrained_dict["roberta." + k]
    unused_dict = {k: v for k, v in model_dict.items() if k not in used_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(used_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. print unused_dict
    print(f"WARNING !!! Following {len([*unused_dict.keys()])} keys are not loaded from {load_path}/pytorch_model.bin")
    print(f"      →→ {[*unused_dict.keys()]}")
    return model


def features_to_device(feats_dict, device):
    for k in feats_dict:
        v = feats_dict[k]
        if v is None:
            continue
        if isinstance(v, list):
            v = [x.to(device) for x in v]
        elif isinstance(v, torch.Tensor):
            v = v.to(device)
        else:
            raise NotImplementedError(f"{type(v)}")
        feats_dict.update({k: v})
    return feats_dict

class BertAttributes:
    def __init__(self,
                 pretrained_name_or_path: str = "bert-base-multilingual-cased",
                 input_representation: str = "subwords",
                 output_representation: str = "cls",
                 freeze_bert: bool = True,
                 n_token_type_ids: int = None,
                 device: str = "cpu" if not torch.cuda.is_available() else "cuda"):
        assert input_representation in ["subwords", "charcnn"]
        assert output_representation in ["cls", "meanpool"]
        assert device in ["cpu", "cuda"]

        self.input_representation = input_representation
        self.output_representation = output_representation
        self.pretrained_name_or_path = pretrained_name_or_path
        self.finetune_bert = not freeze_bert
        self.n_token_type_ids = n_token_type_ids
        self.device = device



class TaskAttributes:
    def __init__(self,
                 name: str,  # "classification" or "seq_tagging"
                 nlabels: int,
                 ignore_index: int = -1,
                 device: str = "cpu" if not torch.cuda.is_available() else "cuda"):

        self.name = name
        self.nlabels = nlabels
        self.ignore_index = ignore_index
        self.device = device


class PromptAttributes:
    def __init__(self,
                 name: str,  # "soft" or "hard",
                 init_from_vocab: bool = False,
                 n_tokens: int = None,
                 random_range: float = 0.5,
                 device: str = "cpu" if not torch.cuda.is_available() else "cuda"):

        self.name = name
        self.init_from_vocab = init_from_vocab
        self.n_tokens = n_tokens
        self.random_range = random_range
        self.device = device
        
        
""" ################ """
""" resource classes """
""" ################ """

class Resource:
    def __init__(self):
        super(Resource, self).__init__()
        self.name = None
        self.attr = None
        self.optimizer = None
        self.requires_bert_optimizer = False

class EncoderResource(Resource, nn.Module):
    def __init__(self,
                 name: str,
                 attr: BertAttributes,
                 config=None,
                 model=None):
        super(EncoderResource, self).__init__()

        self.name = name
        self.attr = attr
        self.config = config
        self.model = model
        self.outdim = None

        self.dropout_hidden_state = nn.Dropout(p=0.25)
        self.dropout_pooler = nn.Dropout(p=0.25)

    def init_bert(self):
        attr = self.attr
        pretrained_name_or_path = attr.pretrained_name_or_path
        bert_config = AutoConfig.from_pretrained(pretrained_name_or_path)
        bert_model = AutoModel.from_pretrained(pretrained_name_or_path)
        if not attr.finetune_bert:
            for param in bert_model.parameters():
                param.requires_grad = False
        bert_model.to(attr.device)
        self.outdim = bert_config.hidden_size
        self.config = bert_config
        self.model = bert_model
        self.requires_bert_optimizer = True
        return

    def forward(self, inputs, batch_splits):

        output = self.model(**inputs, return_dict=True)
        last_hidden_state, pooler_output = output["last_hidden_state"], output["pooler_output"]
        last_hidden_state = [
            merge_subword_encodings_for_words(bert_seq_encodings, seq_splits, device=self.attr.device)
            for bert_seq_encodings, seq_splits in zip(last_hidden_state, batch_splits)
        ]
        last_hidden_state = pad_sequence(last_hidden_state, batch_first=True, padding_value=0).to(self.attr.device)
        
        if last_hidden_state is not None:
            last_hidden_state = self.dropout_hidden_state(last_hidden_state)

        if pooler_output is not None:
            pooler_output = self.dropout_pooler(pooler_output)

        return last_hidden_state, pooler_output


class TaskResource(Resource, nn.Module):
    def __init__(self,
                 name: str,
                 attr: TaskAttributes,
                 mlp_layer=None,
                 criterion=None
                 ):
        super(TaskResource, self).__init__()

        self.name = name
        self.attr = attr
        self.mlp_layer = mlp_layer
        self.criterion = criterion

    def init_mlp(self, indim):
        attr = self.attr
        self.adapter_layer = None
        self.mlp_layer = nn.Linear(indim, attr.nlabels)
        self.mlp_layer.to(attr.device)
        self.set_criterion()
        return

    def set_criterion(self):
        attr = self.attr
        if attr.name == "classification":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif attr.name == "seq_tagging":
            self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=attr.ignore_index)
        else:
            raise ValueError

    def forward(self, last_hidden_state, pooler_output, batch_lengths):
        return last_hidden_state, pooler_output
    

class PromptResource(Resource, nn.Module):
    def __init__(self,
                 name: str,
                 attr: PromptAttributes):
        super(PromptResource, self).__init__()
        
        self.name = name
        self.attr = attr
    
    def init_prompt(self, 
                    wte: nn.Embedding):
        
        self.wte = wte
        if self.attr.init_from_vocab:
            indices = np.random.permutation(range(5000))[:self.attr.n_tokens]
            init_prompt_value = wte.weight[indices].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(self.attr.n_tokens, wte.weight.size(1)).uniform_(
                -self.attr.random_range, self.attr.random_range
            )
        self.soft_prompt = nn.Embedding(self.attr.n_tokens, wte.weight.size(1))
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)
    
    def _extend_labels(self, labels, ignore_index=-100):
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]
        prompt_labels = torch.full((n_batches, self.attr.n_tokens), ignore_index).to(self.attr.device)
        return torch.cat([prompt_labels,labels], dim=1)

    def _extend_attention_mask(self, attention_mask):
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        n_batches = attention_mask.shape[0]
        prompt_mask = torch.full((n_batches, self.attr.n_tokens), 1).to(self.attr.device)
        return torch.cat([prompt_mask, attention_mask], dim=1)
    
    def forward(self, input_ids):
        inputs_embeds = self.wte(input_ids)
        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1).to(self.attr.device)
        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

""" ############## """
""" unified model  """
""" ############## """


class Model(nn.Module):
    def __init__(self,
                 encoder: str,
                 encoder_attribute: dict,
                 task_attribute: dict,
                 prompt_attribute: dict
                 ):
        """
        :param encoders: ["bert", "lstm"] or ["bert"] or "bert",  see docs for details
        :param encoder_attributes: [{...}, {...}], see docs for details
        :param task_attributes: [{...}, {"ignore_index": ignore_index, ...}], see docs for details
        :param n_fields_fusion: number of fusable fileds in the batch
                eg. 2 for a bs=16, where first 8 are for normalized and next 8 for raw texts
        :param fusion_strategy: how to fuse the final encodings before computing logits
        """
        super(Model, self).__init__()
        
        enc_attr = BertAttributes(**encoder_attribute)
        self.encoder_resource = EncoderResource(name=encoder, attr=enc_attr)
        self.encoder_resource.init_bert()
        
        outdim = self.encoder_resource.outdim
        assert outdim > 0

        task_attr = TaskAttributes(**task_attribute)
        self.task_resource = TaskResource(name=task_attr.name, attr=task_attr)
        self.task_resource.init_mlp(outdim)
        
        prompt_attr = PromptAttributes(**prompt_attribute)
        self.prompt_resource = PromptResource(name=prompt_attr.name, attr=prompt_attr)
        self.prompt_resource.init_prompt(self.encoder_resource.model.get_input_embeddings())

    def all_resources(self, return_dict=False):
        if return_dict:
            return {
                "encoder_resource": self._get_encoder_resource(),
                "task_resource": self._get_task_resource(),
                "prompt_resource": self._get_prompt_resource()
            }
        else:
            return [self.encoder_resource, self.task_resource, self.prompt_resource]

    def _get_encoder_resource(self):
        return self.encoder_resource

    def _get_task_resource(self):
        return self.task_resource
    
    def _get_prompt_resource(self):
        return self.prompt_resource
    
    def get_model_nparams(self):
        all_resc = self.all_resources()
        return get_model_nparams(all_resc)

    def save_state_dicts(self, path):
        encoder_resc = self._get_encoder_resource()
        enc_st_dict = encoder_resc.state_dict()
        save_name = "pytorch_model.bin"
        create_path(os.path.join(path, f"encoder"))
        torch.save(enc_st_dict, os.path.join(path, f"encoder", save_name))

        task_resc = self._get_task_resource()
        task_st_dict = task_resc.state_dict()
        save_name = "pytorch_model.bin"
        create_path(os.path.join(path, f"task"))
        torch.save(task_st_dict, os.path.join(path, f"task", save_name))
        
        prompt_resc = self._get_prompt_resource()
        prompt_st_dict = prompt_resc.state_dict()
        save_name = "pytorch_model.bin"
        create_path(os.path.join(path, f"prompt"))
        torch.save(prompt_st_dict, os.path.join(path, f"prompt", save_name))
        return

    def load_state_dicts(self, path, map_location="cpu" if not torch.cuda.is_available() else "cuda"):
        encoder_resc = self._get_encoder_resource()
        save_name = "pytorch_model.bin"
        existing_dict = torch.load(os.path.join(path, f"encoder", save_name), map_location)
        encoder_resc.load_state_dict(existing_dict)

        task_resc = self._get_task_resource()
        save_name = "pytorch_model.bin"
        existing_dict = torch.load(os.path.join(path, f"task", save_name), map_location)
        task_resc.load_state_dict(existing_dict)
        
        prompt_resc = self._get_prompt_resource()
        save_name = "pytorch_model.bin"
        existing_dict = torch.load(os.path.join(path, f"prompt", save_name), map_location)
        prompt_resc.load_state_dict(existing_dict)
        return

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        for resc in self.all_resources():
            resc.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self,
                inputs: dict,
                targets: Union[List[int], List[List[int]]] = None,
                ):
        """
        :param inputs: each item is an output of tokenizer method in `.helpers`
        :param targets: target idxs to compute loss using nn.CrossEntropyLoss
        :return: a dict of outputs and loss
        """

        output_dict = {}
        batch_lengths = inputs["batch_lengths"]
        features = inputs["features"]
        batch_splits = inputs["batch_splits"]
        features = features_to_device(features, self.encoder_resource.attr.device)
        
        bert_feature_dict = {}
        if features["input_ids"] is not None:
            bert_feature_dict["inputs_embeds"] = self.prompt_resource(features["input_ids"])
        '''
        if features["labels"] is not None:
            bert_feature_dict["labels"] = self.prompt_resource._extend_labels(features["labels"])
        '''
        
        if features["attention_mask"] is not None:
            bert_feature_dict["attention_mask"] = self.prompt_resource._extend_attention_mask(features["attention_mask"])
        

        # input encodings
        last_hidden_state, pooler_output = self.encoder_resource(bert_feature_dict, batch_splits)
        # fusion and logits
        logits, loss = [], None
        last_hidden_state, pooler_output = self.task_resource(last_hidden_state, pooler_output, batch_lengths)
        if self.task_resource.name == "classification":
            logits = self.task_resource.mlp_layer(pooler_output)
            if targets:
                target = torch.as_tensor(targets).to(self.task_resource.attr.device)
                loss = self.task_resource.criterion(logits, target)
        elif self.task_resource.name == "seq_tagging":
            logits = self.task_resource.mlp_layer(last_hidden_state)
            if targets:
                target = [torch.tensor(t[:b]) for t, b in zip(target, batch_lengths)]
                target = pad_sequence(target, padding_value=self.task_resource.attr.ignore_index, batch_first=True)
                target = torch.as_tensor(target).to(self.task_resource.attr.device)
                logits = logits.reshape(-1, logits.shape[-1])
                target = target.reshape(-1)
                loss = self.task_resource.criterion(logits, target)

        output_dict.update(
            {
                "loss": loss,
                "logits": logits
            }
        )

        return output_dict

    
    
    def predict(self,
                inputs: dict,
                targets: Union[List[int], List[List[int]]] = None,
                ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output_dict = self.forward(inputs, targets)
            logits = output_dict["logits"]

            probs_list, preds_list, targets_list, acc_num_list, acc_list = [], [], [], [], []
            probs = F.softmax(logits, dim=-1)
            # dims: [batch_size] if task=classification, [batch_size, mxlen] if task=seq_tagging
            argmax_probs = torch.argmax(probs, dim=-1)
            if targets is not None:
                if self.task_resource.name == "classification":
                    preds = argmax_probs.cpu().detach().numpy().tolist()
                    assert len(preds) == len(targets), print(len(preds), len(targets))
                    acc_num = sum([m == n for m, n in zip(preds, targets)])
                    acc = acc_num / len(targets)  # dims: [1]
                elif self.task_resource.name == "seq_tagging":
                    preds = argmax_probs.reshape(-1).cpu().detach().numpy().tolist()
                    ignore_index = self.task_resource.attr.ignore_index
                    batch_lengths = inputs["batch_lengths"]
                    targets = [torch.tensor(t[:b]) for t, b in zip(targets, batch_lengths)]
                    targets = pad_sequence(targets, padding_value=ignore_index, batch_first=True)
                    targets = targets.reshape(-1).cpu().detach().numpy().tolist()
                    assert len(preds) == len(targets), print(len(preds), len(targets))
                    new_preds, new_targets = [], []
                    for ii, jj in zip(preds, targets):
                        if jj != ignore_index:
                            new_preds.append(ii)
                            new_targets.append(jj)
                    acc_num = sum([m == n for m, n in zip(new_preds, new_targets)])
                    acc = acc_num / len(new_targets)
                    preds = new_preds
                    targets = new_targets
            probs = probs.cpu().detach().numpy().tolist()
        output_dict.update(
            {
                "logits": logits,
                "probs": probs,
                "preds": preds,
                "targets": targets,
                "acc_num": acc_num,
                "acc": acc,
            }
        )

        if was_training:
            self.train()

        return output_dict
