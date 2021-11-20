import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from pytorch_pretrained_bert import BertAdam
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AdamW, get_scheduler

from tokenizer import Tokenizer
from helpers import batch_iter, progress_bar
from helpers import create_vocab, load_vocab
from models import Model
from datasets import read_datasets_jsonl_new, create_path
from args import Args

LOGS_FOLDER = "./logs"
printlog = print

def _get_checkpoints_path():
    if args.eval_ckpt_path:
        assert all([ii in ["test"] for ii in args.mode])
        CKPT_PATH = args.eval_ckpt_path
    else:
        assert args.mode[0] == "train", print("`mode` must first contain `train` if no eval ckpt path is specified")
        CKPT_PATH = os.path.join(args.dataset_folder,
                                 "checkpoints",
                                 f'{str(datetime.datetime.now()).replace(" ", "_")}')
        if os.path.exists(CKPT_PATH):
            msg = f"CKPT_PATH: {CKPT_PATH} already exists. Did you mean to set mode to `test` ?"
            raise Exception(msg)
        create_path(CKPT_PATH)
    printlog(f"CKPT_PATH: {CKPT_PATH}")
    return CKPT_PATH


def _set_logger():
    global printlog

    if args.debug:
        logger_file_name = None
        printlog = print
    else:
        if not os.path.exists(os.path.join(CKPT_PATH, LOGS_FOLDER)):
            os.makedirs(os.path.join(CKPT_PATH,
                                     LOGS_FOLDER))
        logger_file_name = os.path.join(CKPT_PATH,
                                        LOGS_FOLDER,
                                        f'{str(datetime.datetime.now()).replace(" ", "_")}')
        logging.basicConfig(level=logging.INFO, filename=logger_file_name, filemode='a',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
        printlog = logging.info

    print(f"logger_file_name: {logger_file_name}")

    printlog('\n\n\n--------------------------------\nBeginning to log...\n--------------------------------\n')
    printlog(" ".join(sys.argv))
    return logger_file_name



def load_examples():
    train_examples = read_datasets_jsonl_new(os.path.join(args.dataset_folder, f"train.jsonl"), f"dev")
    dev_examples = read_datasets_jsonl_new(os.path.join(args.dataset_folder, f"dev.jsonl"), f"dev")
    test_examples = read_datasets_jsonl_new(os.path.join(args.dataset_folder, f"test.jsonl"), f"test")
    return train_examples, dev_examples, test_examples


def load_target_label_vocab():
    target_label_ckpt_path = os.path.join(CKPT_PATH, f"task")
    create_path(target_label_ckpt_path)
    target_label_ckpt_path = os.path.join(target_label_ckpt_path, "target_label_vocab.json")
    if any(["train" in m for m in args.mode]):
        if not getattr(train_examples[0], args.target_label_field):
            raise ValueError(f"Unable to find {args.target_label_field} target_label_field in the training file")
        data = [getattr(ex, args.target_label_field) for ex in train_examples]
        if any([True if d is None or d == "" else False for d in data]):
            raise ValueError(f"Found empty strings when obtaining {args.target_label_field} target_label_field")
        if args.task_type == "classification":
            vcb = create_vocab(data, is_label=True, labels_data_split_at_whitespace=False)
        elif args.task_type == "seq_tagging":
            vcb = create_vocab(data, is_label=True, labels_data_split_at_whitespace=True)
        else:
            raise ValueError
        json.dump(vcb._asdict(), open(target_label_ckpt_path, "w"), indent=4)
    else:
        vcb = load_vocab(target_label_ckpt_path)
    return vcb


def get_batch_inputs_and_targets(batch_examples):
    # get inputs
    batch_sentences = [getattr(ex, args.text_field) for ex in batch_examples]
    inputs = tokenizer.bert_subword_tokenize(batch_sentences,
                                             max_len=args.max_len)
    # get labels
    batch_targets = [getattr(ex, args.target_label_field) for ex in batch_examples]
    if args.task_type == "classification":
        targets = [target_label_vocab.token2idx[tgt] for tgt in batch_targets]
    elif args.task_type == "seq_tagging":
        targets = [[target_label_vocab.token2idx[token] for token in tgt.split(" ")] for tgt in batch_targets]
    return inputs, targets



def run_train_epoch():
    examples = train_examples
    tot_loss = 0
    n_batches = int(np.ceil(len(examples) / args.batch_size))
    examples_batch_iter = batch_iter(examples, args.batch_size, shuffle=True)
    printlog(f"len of train data: {len(examples)}")
    printlog(f"n_batches of train data: {n_batches}")
    for optimizer in optimizers:
        if optimizer:
            optimizer.zero_grad()
    model.train()
    for batch_id, batch_examples in enumerate(examples_batch_iter):
        st_time = time.time()
        inputs, targets = get_batch_inputs_and_targets(batch_examples)
        output_dict = model(inputs=inputs, targets=targets)
        loss = output_dict["loss"]
        batch_loss = loss.cpu().detach().numpy()
        tot_loss += batch_loss
        if args.grad_acc > 1:
            loss = loss / args.grad_acc
        loss.backward()
        if (batch_id + 1) % args.grad_acc == 0 or batch_id >= n_batches - 1:
            for optimizer in optimizers:
                if optimizer:
                    optimizer.step()
                    optimizer.zero_grad()
        # update progress
        progress_bar(batch_id + 1, n_batches,
                    ["batch_time", "batch_loss", "avg_batch_loss"],
                    [time.time() - st_time, batch_loss, tot_loss / (batch_id + 1)])
    
def run_dev_epoch(mode="dev"):
    examples = dev_examples if mode == "test" else test_examples
    preds, true = [], []
    tot_acc, tot_f1, tot_loss = 0., 0., 0.
    n_batches = int(np.ceil(len(examples) / args.batch_size))
    examples_batch_iter = batch_iter(examples, args.batch_size, shuffle=True)
    printlog(f"len of {mode} data: {len(examples)}")
    printlog(f"n_batches of {mode} data: {n_batches}")
    model.eval()
    sents, labels = [], []
    for batch_id, batch_examples in enumerate(examples_batch_iter):
        st_time = time.time()
        inputs, targets = get_batch_inputs_and_targets(batch_examples)
        output_dict = model.predict(inputs=inputs, targets=targets)
        batch_loss = output_dict["loss"].cpu().detach().numpy()
        tot_loss += batch_loss
        tot_acc += output_dict["acc_num"]
        acc_num = output_dict["acc_num"]
        preds.extend(output_dict["preds"])
        true.extend(output_dict["targets"])
        # update progress
        progress_bar(batch_id + 1, n_batches,
                    ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", 'avg_batch_acc'],
                    [time.time() - st_time, batch_loss, tot_loss / (batch_id + 1), 
                     acc_num / args.batch_size, tot_acc / ((batch_id + 1) * args.batch_size)])
        sents.extend(inputs["batch_sentences"])
        labels.extend(targets)
    tot_acc /= len(examples)  # exact
    tot_loss /= n_batches # approximate
    tot_f1 = f1_score(true, preds, average='weighted')
    return preds, true, tot_loss, tot_acc, tot_f1, sents, labels


def train():
    start_epoch, n_epochs = 0, args.max_epochs
    best_dev_acc, best_dev_acc_epoch = 0., -1
    best_dev_f1, best_dev_f1_epoch = 0., -1
    for epoch_id in range(start_epoch, n_epochs):
        printlog("\n\n################")
        printlog(f"epoch: {epoch_id}")
        """ training """
        run_train_epoch()
        dev_preds, dev_true, dev_loss, dev_acc, dev_f1, _, _ = run_dev_epoch("dev")
        printlog("\n Validation Complete")
        printlog(f"Validation avg_loss: {dev_loss:.4f} and acc: {dev_acc:.4f} and f1: {dev_f1:.4f}")
        printlog("\n" + classification_report(dev_true, dev_preds, digits=4))
        
        """ model saving """
        name = "model.pth.tar"  # "model-epoch{}.pth.tar".format(epoch_id)
        if args.checkpoint_using_accuracy:
            if (start_epoch == 0 and epoch_id == start_epoch) or best_dev_acc < dev_acc:
                best_dev_acc, best_dev_acc_epoch = dev_acc, epoch_id
                model.save_state_dicts(CKPT_PATH)
                printlog("Model(s) saved in {} at the end of epoch(0-base) {}".format(CKPT_PATH, epoch_id))
            else:
                printlog("no improvements in results to save a checkpoint")
                printlog(f"checkpoint previously saved during epoch {best_dev_acc_epoch}(0-base) at: "
                    f"{os.path.join(CKPT_PATH, name)}")
        else:
            if (start_epoch == 0 and epoch_id == start_epoch) or best_dev_f1 < dev_f1:
                best_dev_f1, best_dev_f1_epoch = dev_f1, epoch_id
                best_dev_acc, best_dev_acc_epoch = dev_acc, epoch_id
                model.save_state_dicts(CKPT_PATH)
                printlog("Model(s) saved in {} at the end of epoch(0-base) {}".format(CKPT_PATH, epoch_id))
            else:
                printlog("no improvements in results to save a checkpoint")
                printlog(f"checkpoint previously saved during epoch {best_dev_f1_epoch}(0-base) at: "
                        f"{os.path.join(CKPT_PATH, name)}")
        
                
def save_predictions(preds, truths, sents, labels, mode):
    opfile = open(os.path.join(CKPT_PATH, "predictions_"+mode+".txt"), "w", encoding="utf-8")
    idx = 0
    for i, (sent, label) in enumerate(zip(sents, labels)):
        pred = ""
        truth = ""
        n_labels = 1
        if isinstance (label,list): 
            n_labels = len(label)
        for _ in range(n_labels):
            pred += target_label_vocab.idx2token[preds[idx]] + " "
            truth += target_label_vocab.idx2token[truths[idx]] + " "
            idx += 1
            
        print(sent.strip() + "\t" + truth.strip() + "\t" + pred.strip(), file=opfile)
    opfile.close()
    

def test(dev_only):
    model.load_state_dicts(path=CKPT_PATH)
    printlog("\n\n################")
    printlog(f"testing `dev` file; loading model(s) from {CKPT_PATH}")
    dev_preds, dev_true, dev_loss, dev_acc, dev_f1, dev_sents, dev_labels = run_dev_epoch("dev")
    printlog("\nValidation Complete")
    printlog(f"Validation avg_loss: {dev_loss:.4f} and acc: {dev_acc:.4f} and f1: {dev_f1:.4f}")
    printlog("\n" + classification_report(dev_true, dev_preds, digits=4))
    printlog("")
    printlog(f"\n(NEW!) saving predictions in the folder: {CKPT_PATH}")
    save_predictions(dev_preds, dev_true, dev_sents, dev_labels, "dev")
    if not dev_only:
        printlog("\n\n################")
        printlog(f"testing `test` file; loading model(s) from {CKPT_PATH}")
        test_preds, test_true, test_loss, test_acc, test_f1, test_sents, test_labels = run_dev_epoch("test")
        printlog("\nTest Complete")
        printlog(f"Test avg_loss: {test_loss:.4f} and acc: {test_acc:.4f} and f1: {test_f1:.4f}")
        printlog("\n" + classification_report(test_true, test_preds, digits=4))
        printlog(f"\n(NEW!) saving predictions in the folder: {CKPT_PATH}")
        save_predictions(test_preds, test_true, test_sents, test_labels, "test")

def load_optimizers():
    optims = []
    model_resources = model.all_resources()
    for resc in model_resources:
        for name, param in resc.named_parameters():
            if '.layer.' in name or 'wte' in name or 'pooler' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            printlog(f"parameter name and grad_info: {name, param.requires_grad}")
            
    for resc in model_resources:
        if resc.requires_bert_optimizer:
            params = [param for param in list(resc.named_parameters())]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / args.batch_size / args.grad_acc * args.max_epochs)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
            printlog(f"{len(params)} number of params are being optimized with BertAdam")
        else:
            params = [param for param in list(resc.parameters())]
            if len(params) == 0:
                optimizer = None
                print(f"Warning: optimizer got an empty parameter list in `{resc.name}`")
            else:
                optimizer = torch.optim.Adam(resc.parameters(), lr=0.001)
                printlog(f"{len(params)} number of params are being optimized with Adam")
        optims.append(optimizer)
    printlog(f"number of parameters (all, trainable) in your model: {model.get_model_nparams()}")
    
    
    
    '''
    for resc, resource in model_resources.items():
        if resc == "encoder_resource":
            optimizer_grouped_parameters = [{
                    "params": [p for n, p in resource.named_parameters() if '.layer' not in n],
                    "weight_decay": args.weight_decay
                }]
            params = [param for param in list(resc.named_parameters())]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / train_batch_size / args.grad_acc * n_epochs)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
            printlog(f"{len(params)} number of params are being optimized with BertAdam")
        if resc == "task_resource":
            optimizer_grouped_parameters = [{
                    "params": [p for n, p in resource.named_parameters()],
                    "weight_decay": args.weight_decay
                }]
            if len(optimizer_grouped_parameters[0]['params']) == 0:
                optimizer = None
                lr_scheduler = None
                print(f"Warning: optimizer got an empty parameter list in `{resc}`")
            else:
                printlog(f"{len(optimizer_grouped_parameters[0]['params'])} number of params are being optimized with AdamW")
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
                lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=args.num_warmup_steps,
                    num_training_steps=args.max_epochs
                )
        if resc == "prompt_resource":
            optimizer_grouped_parameters = [{
                    "params": [p for n, p in resource.named_parameters() if n == "soft_prompt.weight"],
                    "weight_decay": args.weight_decay
                }]
        
        lr_schedulers.append(lr_scheduler)
        optims.append(optimizer)
    '''
    
    #printlog(f"number of parameters (all, trainable) in your model: {model.get_model_nparams()}")
    return optims
    
    
def run(argsobj: Args = None, **kwargs):
    global args, CKPT_PATH
    global train_examples, dev_examples, test_examples
    global model, tokenizer, target_label_vocab
    global optimizers, lr_schedulers
    
    args = argsobj or Args(**kwargs)
    
    """ basics """
    CKPT_PATH = _get_checkpoints_path()
    logger_file_name = None
    if not args.debug:
        logger_file_name = _set_logger()

    """ load dataset """
    train_examples, dev_examples, test_examples = load_examples()
    if args.debug:
        printlog("debug mode enabled; reducing dataset size to 40 (train) and 20 (dev/test) resp.")
        train_examples = train_examples[:40]
        dev_examples = dev_examples[:20]
        test_examples = test_examples[:20]
    
    """ load tokenizer """
    tokenizer = Tokenizer(bert_tokenizer=AutoTokenizer.from_pretrained(args.pretrained_name_or_path, use_fast=False))

    """ load targets """
    target_label_vocab = load_target_label_vocab()


    """ define model """
    model = Model(
        encoder="bert",
        encoder_attribute={"pretrained_name_or_path":  args.pretrained_name_or_path},
        task_attribute={"name" : args.task_type, "nlabels" : target_label_vocab.n_all_tokens},
        prompt_attribute={"name" : args.prompt_type, 
                          "n_tokens" : args.n_prompt_tokens, 
                          "init_from_vocab" : args.init_prompt_from_vocab}
    )
    
    if "train" in args.mode:
        """ load optimizers """
        optimizers = load_optimizers()
        train()
    
    if "test" in args.mode:
        test(dev_only=False)
        
    if (not args.debug) and logger_file_name:
        os.system(f"cat {logger_file_name}")