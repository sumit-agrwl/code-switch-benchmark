import io
import json
import sys
from collections import namedtuple
from typing import List, Union
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from paths import CHECKPOINTS_PATH
from utils import get_module_or_attr

""" ##################### """
"""       helpers         """
""" ##################### """


def get_model_nparams(model):
    """
    :param model: can be a list of nn.Module or a single nn.Module
    :return: (all, trainable parameters)
    """
    if not isinstance(model, list):
        items = [model, ]
    else:
        items = model
    ntotal, n_gradrequired = 0, 0
    for item in items:
        for param in list(item.parameters()):
            temp = 1
            for sz in list(param.size()):
                temp *= sz
            ntotal += temp
            if param.requires_grad:
                n_gradrequired += temp
    return ntotal, n_gradrequired


def train_validation_split(data, train_ratio, seed=11927):
    len_ = len(data)
    train_len_ = int(np.ceil(train_ratio * len_))
    inds_shuffled = np.arange(len_)
    np.random.seed(seed)
    np.random.shuffle(inds_shuffled)
    train_data = []
    for ind in inds_shuffled[:train_len_]:
        train_data.append(data[ind])
    validation_data = []
    for ind in inds_shuffled[train_len_:]:
        validation_data.append(data[ind])
    return train_data, validation_data


def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of labels and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        yield [data[i] for i in batch_indices]


def progress_bar(value, endvalue, names=[], values=[], bar_length=15):
    assert (len(names) == len(values))
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    string = ''
    for name, val in zip(names, values):
        temp = '|| {0}: {1:.4f} '.format(name, val) if val else '|| {0}: {1} '.format(name, None)
        string += temp
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    if value >= endvalue - 1:
        print()
    return


""" ##################### """
"""    vocab functions    """
""" ##################### """


def load_vocab(path) -> namedtuple:
    return_dict = json.load(open(path))
    # for idx2token, idx2chartoken, have to change keys from strings to ints
    #   https://stackoverflow.com/questions/45068797/how-to-convert-string-int-json-into-real-int-with-json-loads
    if "token2idx" in return_dict:
        return_dict.update({"idx2token": {v: k for k, v in return_dict["token2idx"].items()}})
    if "chartoken2idx" in return_dict:
        return_dict.update({"idx2chartoken": {v: k for k, v in return_dict["chartoken2idx"].items()}})

    # NEW
    # vocab: dict to named tuple
    vocab = namedtuple('vocab', sorted(return_dict))
    return vocab(**return_dict)


def create_vocab(data: List[str],
                 keep_simple=False,
                 min_max_freq: tuple = (1, float("inf")),
                 topk=None,
                 intersect: List = None,
                 load_char_tokens: bool = False,
                 is_label: bool = False,
                 labels_data_split_at_whitespace: bool = False) -> namedtuple:
    """
    :param data: list of sentences from which tokens are obtained as whitespace seperated
    :param keep_simple: retain tokens that have ascii and do not have digits (for preprocessing)
    :param min_max_freq: retain tokens whose count satisfies >min_freq and <max_freq
    :param topk: retain only topk tokens (specify either topk or min_max_freq)
    :param intersect: retain tokens that are at intersection with a custom token list
    :param load_char_tokens: if true, character tokens will also be loaded
    :param is_label: when the inouts are list of labels
    :param labels_data_split_at_whitespace:
    :return: a vocab namedtuple
    """

    if topk is None and (min_max_freq[0] > 1 or min_max_freq[1] < float("inf")):
        raise Exception("both min_max_freq and topk should not be provided at once !")

    # if is_label
    if is_label:

        def split_(txt: str):
            if labels_data_split_at_whitespace:
                return txt.split(" ")
            else:
                return [txt, ]

        # get all tokens
        token_freq, token2idx, idx2token = {}, {}, {}
        for example in tqdm(data):
            for token in split_(example):
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
        print(f"Total tokens found: {len(token_freq)}")
        print(f"token_freq:\n{token_freq}\n")

        # create token2idx and idx2token
        for token in token_freq:
            idx = len(token2idx)
            idx2token[idx] = token
            token2idx[token] = idx

        token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return_dict = {"token2idx": token2idx,
                       "idx2token": idx2token,
                       "token_freq": token_freq,
                       "n_tokens": len(token2idx),
                       "n_all_tokens": len(token2idx),
                       "pad_token_idx": -1}

    else:

        # get all tokens
        token_freq, token2idx, idx2token = {}, {}, {}
        for example in tqdm(data):
            for token in example.split(" "):
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
        print(f"Total tokens found: {len(token_freq)}")

        # retain only simple tokens
        if keep_simple:
            isascii = lambda s: len(s) == len(s.encode())
            hasdigits = lambda s: len([x for x in list(s) if x.isdigit()]) > 0
            tf = [(t, f) for t, f in [*token_freq.items()] if (isascii(t) and not hasdigits(t))]
            token_freq = {t: f for (t, f) in tf}
            print(f"After removing non-ascii and tokens with digits, total tokens retained: {len(token_freq)}")

        # retain only tokens with specified min and max range
        if min_max_freq[0] > 1 or min_max_freq[1] < float("inf"):
            sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
            tf = [(i[0], i[1]) for i in sorted_ if (min_max_freq[0] <= i[1] <= min_max_freq[1])]
            token_freq = {t: f for (t, f) in tf}
            print(f"After min_max_freq selection, total tokens retained: {len(token_freq)}")

        # retain only topk tokens
        if topk is not None:
            sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
            token_freq = {t: f for (t, f) in list(sorted_)[:topk]}
            print(f"After topk selection, total tokens retained: {len(token_freq)}")

        # retain only interection of tokens
        if intersect is not None and len(intersect) > 0:
            tf = [(t, f) for t, f in [*token_freq.items()] if (t in intersect or t.lower() in intersect)]
            token_freq = {t: f for (t, f) in tf}
            print(f"After intersection, total tokens retained: {len(token_freq)}")

        # create token2idx and idx2token
        for token in token_freq:
            idx = len(token2idx)
            idx2token[idx] = token
            token2idx[token] = idx

        # add <<PAD>> special token
        ntokens = len(token2idx)
        pad_token = "<<PAD>>"
        token_freq.update({pad_token: -1})
        token2idx.update({pad_token: ntokens})
        idx2token.update({ntokens: pad_token})

        # add <<UNK>> special token
        ntokens = len(token2idx)
        unk_token = "<<UNK>>"
        token_freq.update({unk_token: -1})
        token2idx.update({unk_token: ntokens})
        idx2token.update({ntokens: unk_token})

        # new
        # add <<EOS>> special token
        ntokens = len(token2idx)
        eos_token = "<<EOS>>"
        token_freq.update({eos_token: -1})
        token2idx.update({eos_token: ntokens})
        idx2token.update({ntokens: eos_token})

        # new
        # add <<SOS>> special token
        ntokens = len(token2idx)
        sos_token = "<<SOS>>"
        token_freq.update({sos_token: -1})
        token2idx.update({sos_token: ntokens})
        idx2token.update({ntokens: sos_token})

        # return dict
        token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return_dict = {"token2idx": token2idx,
                       "idx2token": idx2token,
                       "token_freq": token_freq,
                       "pad_token": pad_token,
                       "pad_token_idx": token2idx[pad_token],
                       "unk_token": unk_token,
                       "unk_token_idx": token2idx[unk_token],
                       "eos_token": eos_token,
                       "eos_token_idx": token2idx[eos_token],
                       "sos_token": sos_token,
                       "sos_token_idx": token2idx[sos_token],
                       "n_tokens": len(token2idx) - 4,
                       "n_special_tokens": 4,
                       "n_all_tokens": len(token2idx)
                       }

        # load_char_tokens
        if load_char_tokens:
            print("loading character tokens as well")
            char_return_dict = create_char_vocab(use_default=True, data=data)
            return_dict.update(char_return_dict)

    # NEW
    # vocab: dict to named tuple
    vocab = namedtuple('vocab', sorted(return_dict))
    return vocab(**return_dict)


""" ##################### """
"""     tokenizers        """
""" ##################### """


def _tokenize_untokenize(input_text: str, bert_tokenizer):
    subtokens = bert_tokenizer.tokenize(input_text)
    output = []
    for subt in subtokens:
        if subt.startswith("##"):
            output[-1] += subt[2:]
        else:
            output.append(subt)
    return " ".join(output)


def _custom_bert_tokenize_sentence(input_text, bert_tokenizer, max_len):
    tokens = []
    split_sizes = []
    text = []
    # for token in _tokenize_untokenize(input_text, bert_tokenizer).split(" "):
    for token in input_text.split(" "):
        word_tokens = bert_tokenizer.tokenize(token)
        if len(tokens) + len(word_tokens) > max_len - 2:  # 512-2 = 510
            break
        if len(word_tokens) == 0:
            continue
        tokens.extend(word_tokens)
        split_sizes.append(len(word_tokens))
        text.append(token)

    return " ".join(text), tokens, split_sizes


def _custom_bert_tokenize_sentence_with_lang_ids(input_text, bert_tokenizer, max_len, input_lang_ids):
    tokens = []
    split_sizes = []
    text = []
    lang_ids = []

    # the 2 is substracted due to added terminal start/end positions
    assert len(input_text.split(" ")) == len(input_lang_ids.split(" ")) - 2, \
        print(len(input_text.split(" ")), len(input_lang_ids.split(" ")) - 2)

    lids = input_lang_ids.split(" ")
    non_terminal_lids = lids[1:-1]

    # cannot use _tokenize_untokenize(input_text) because doing so might change the one-one mapping between
    #   input_text and non_terminal_lids
    for token, lid in zip(input_text.split(" "), non_terminal_lids):
        word_tokens = bert_tokenizer.tokenize(token)
        if len(tokens) + len(word_tokens) > max_len - 2:  # 512-2 = 510
            break
        if len(word_tokens) == 0:
            continue
        tokens.extend(word_tokens)
        split_sizes.append(len(word_tokens))
        text.append(token)
        lang_ids.extend([lid] * len(word_tokens))
    lang_ids = [lids[0]] + lang_ids + [lids[-1]]

    return " ".join(text), tokens, split_sizes, " ".join(lang_ids)


def merge_subword_encodings_for_words(bert_seq_encodings,
                                      seq_splits,
                                      mode='avg',
                                      keep_terminals=False,
                                      device=torch.device("cpu")):
    bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
    bert_cls_enc = bert_seq_encodings[0:1, :]
    bert_sep_enc = bert_seq_encodings[-1:, :]
    bert_seq_encodings = bert_seq_encodings[1:-1, :]
    # a tuple of tensors
    split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
    batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
    if mode == 'avg':
        seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(device)
        out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
    elif mode == "add":
        out = torch.sum(batched_encodings, dim=1)
    elif mode == "first":
        out = batched_encodings[:, 0, :]
    else:
        raise Exception("Not Implemented")

    if keep_terminals:
        out = torch.cat((bert_cls_enc, out, bert_sep_enc), dim=0)
    return out


def merge_subword_encodings_for_sentences(bert_seq_encodings,
                                          seq_splits):
    bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
    bert_cls_enc = bert_seq_encodings[0:1, :]
    bert_sep_enc = bert_seq_encodings[-1:, :]
    bert_seq_encodings = bert_seq_encodings[1:-1, :]
    return torch.mean(bert_seq_encodings, dim=0)
