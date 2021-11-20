import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from helpers import create_vocab, load_vocab, _custom_bert_tokenize_sentence, _custom_bert_tokenize_sentence_with_lang_ids


class Tokenizer:
    def __init__(self,
                 word_vocab=None,
                 tag_input_label_vocab=None,
                 bert_tokenizer=None):

        self.word_vocab = word_vocab
        self.tag_input_label_vocab = tag_input_label_vocab
        self.bert_tokenizer = bert_tokenizer
        self.fastTextVecs = None

        # self.tokenize = None  # assign to right tokenize func from below based on the requirement

    def load_tag_vocab(self, data):
        self.tag_input_label_vocab = create_vocab(data, is_label=False, labels_data_split_at_whitespace=True)

    def save_tag_vocab_to_checkpoint(self, ckpt_dir):
        if not self.tag_input_label_vocab:
            print("`tag_input_label_vocab` is None and need to be loaded first")
            return

        json.dump(self.tag_input_label_vocab._asdict(),
                  open(os.path.join(ckpt_dir, "tag_input_label_vocab.json"), "w"),
                  indent=4)
        return

    def load_tag_vocab_from_checkpoint(self, ckpt_dir):
        assert self.tag_input_label_vocab is None, print("`tag_input_label_vocab` is not None and overwriting it")
        self.tag_input_label_vocab = load_vocab(os.path.join(ckpt_dir, "tag_input_label_vocab.json"))
        return

    def load_word_vocab(self, data):
        self.word_vocab = create_vocab(data, is_label=False, load_char_tokens=True)

    def save_word_vocab_to_checkpoint(self, ckpt_dir):
        if not self.word_vocab:
            print("`word_vocab` is None and need to be loaded first")
            return

        json.dump(self.word_vocab._asdict(),
                  open(os.path.join(ckpt_dir, "word_vocab.json"), "w"),
                  indent=4)
        return

    def load_word_vocab_from_checkpoint(self, ckpt_dir):
        assert self.word_vocab is None, print("`word_vocab` is not None and overwriting it")
        self.word_vocab = load_vocab(os.path.join(ckpt_dir, "word_vocab.json"))
        return

    def bert_subword_tokenize(self,
                              batch_sentences,
                              bert_tokenizer=None,
                              batch_tag_sequences=None,
                              max_len=512,
                              as_dict=True):

        bert_tokenizer = bert_tokenizer or self.bert_tokenizer
        text_padding_idx = bert_tokenizer.pad_token_id
        
        if batch_tag_sequences is not None:
            assert self.tag_input_label_vocab, \
                print(f"`tag_input_label_vocab` is required for processing batch_tag_sequences")

        if batch_tag_sequences is not None:
            assert len(batch_tag_sequences) == len(batch_sentences)
            # adding "other" at ends, and converting them to idxs
            batch_tag_sequences = (
                [" ".join([str(self.tag_input_label_vocab.sos_token_idx)] +
                          [str(self.tag_input_label_vocab.token2idx[tag]) for tag in tag_sequence.split(" ")] +
                          [str(self.tag_input_label_vocab.eos_token_idx)])
                 for tag_sequence in batch_tag_sequences]
            )
            trimmed_batch_sentences = [
                _custom_bert_tokenize_sentence_with_lang_ids(text, bert_tokenizer, max_len, tag_ids)
                for text, tag_ids in zip(batch_sentences, batch_tag_sequences)]
            batch_sentences, batch_tokens, batch_splits, batch_tag_ids = list(zip(*trimmed_batch_sentences))
            batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
            batch_input_ids = pad_sequence(
                [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
                padding_value=text_padding_idx)
            batch_attention_masks = pad_sequence(
                [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts],
                batch_first=True,
                padding_value=0)
            batch_token_type_ids = pad_sequence(
                [torch.tensor([int(sidx) for sidx in tag_ids.split(" ")]) for tag_ids in batch_tag_ids],
                batch_first=True,
                padding_value=self.tag_input_label_vocab.pad_token_idx)
            batch_bert_dict = {
                "attention_mask": batch_attention_masks,
                "input_ids": batch_input_ids,
                "token_type_ids": batch_token_type_ids
            }
        else:
            # batch_sentences = [text if text.strip() else "." for text in batch_sentences]
            trimmed_batch_sentences = [_custom_bert_tokenize_sentence(text, bert_tokenizer, max_len) for text in
                                       batch_sentences]
            batch_sentences, batch_tokens, batch_splits = list(zip(*trimmed_batch_sentences))
            batch_encoded_dicts = [bert_tokenizer.encode_plus(tokens) for tokens in batch_tokens]
            batch_input_ids = pad_sequence(
                [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
                padding_value=text_padding_idx)
            batch_attention_masks = pad_sequence(
                [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts],
                batch_first=True,
                padding_value=0)
            batch_bert_dict = {
                "attention_mask": batch_attention_masks,
                "input_ids": batch_input_ids,
                "labels": batch_input_ids
            }

        batch_lengths = [len(sent.split(" ")) for sent in batch_sentences]  # useful for lstm based downstream layers
        if as_dict:
            return {
                "features": batch_bert_dict,
                "batch_splits": batch_splits,
                "batch_sentences": batch_sentences,
                "batch_lengths": batch_lengths,  # new
                "batch_size": len(batch_sentences),  # new
            }
        # not returning `batch_lengths` for backward compatability
        return batch_sentences, batch_bert_dict, batch_splits