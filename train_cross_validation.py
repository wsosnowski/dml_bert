#!/usr/bin/env python
# -*- coding: utf-8 -*-

import transformers
from numpy import unique
from transformers import AutoTokenizer, Trainer, TrainingArguments, RobertaForSequenceClassification
from RobertaContrastiveLearning import RobertaContrastiveLearning
import numpy as np
import argparse
from utils.tripleentropy_dataset import CLDatasetClassification, CLDatasetNLI
from utils.load_dataset import prepare_dataset
from utils.training import set_seed, compute_metrics
from sklearn.model_selection import KFold
from pytorch_metric_learning import losses as losses_ml
from sentence_transformers.losses import BatchAllTripletLoss

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-ml', '--max-length', default=64, type=int, dest='max_length')
parser.add_argument('--learning-rate', default=1e-5, type=float, dest='learning_rate')
parser.add_argument('--num-warmup-steps', default=10, type=int, dest='num_warmup_steps')
parser.add_argument('--eps', default=1e-08, type=float, dest='eps')
parser.add_argument('--model-name', default="roberta-large", dest='c')
parser.add_argument('--model-type', default="triple-entropy", dest='model_type')
parser.add_argument('--weight-decay', default=0.01, type=float, dest='weight_decay')
parser.add_argument('--la', default=8, type=float, dest='la')
parser.add_argument('--supcon-temp', default=0.1, type=float, dest='supcon_temp')
parser.add_argument('--gamma', default=0.1, type=float, dest='gamma')
parser.add_argument('--margin', default=0.1, type=float, dest='margin')
parser.add_argument('--centers', default=5, type=int, dest='centers')
parser.add_argument('--beta', default=0.4, type=float, dest='beta')
parser.add_argument('--seed', default=2048, type=int, dest='seed')
parser.add_argument('--output-dir', default="./result", dest='output_dir')
parser.add_argument('--save-steps', default=100, type=int, dest='save_steps')
parser.add_argument('--epochs', default=120, type=int, dest='epochs')
parser.add_argument('--num-training-steps', default=120, type=int, dest='num_training_steps')
parser.add_argument('--per-device-train-batch-size', default=64, type=int, dest='per_device_train_batch_size')
parser.add_argument('--per-device-eval-batch-size', default=64, type=int, dest='per_device_eval_batch_size')
parser.add_argument('--sample-size', default=20, type=int, dest='sample_size')
parser.add_argument('--n-split', default=4, type=int, dest='n_split')
parser.add_argument('--dataset-name', default="cr", dest='dataset_name')
parser.add_argument('--softmax-scale', default=1, dest='softmax_scale')
parser.add_argument('--alpha', default=0.1, dest='alpha')
parser.add_argument('--extended-inference', default=0.1, dest='extended_inference')


def cross_validate(args):
    seed = args.seed
    max_length = args.max_length
    learning_rate = args.learning_rate
    num_warmup_steps = args.num_warmup_steps
    num_training_steps = args.num_training_steps
    eps = args.eps
    model_name = args.model_name
    model_type = args.model_type
    weight_decay = args.weight_decay
    la = args.la
    gamma = args.gamma
    margin = args.margin
    centers = args.centers
    beta = args.beta
    save_steps = args.save_steps
    sample_size = args.sample_size
    n_split = args.n_split
    supcon_temp = args.supcon_temp
    np.random.seed(seed)
    dataset_name = args.dataset_name
    softmax_scale = args.softmax_scale
    alpha = args.alpha
    extended_inference = args.extended_inference

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=save_steps
    )

    kf = KFold(n_splits=n_split, random_state=seed, shuffle=True)
    cross_val_res = {}
    df = prepare_dataset(dataset_name)

    if model_name == "roberta-base":
        embedding_size = 768
    else:
        embedding_size = 1024

    for fold_id, (train_index, valid_index) in enumerate(kf.split(df)):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if dataset_name == "mrpc":
            train_dataset = CLDatasetNLI(train_index, df, tokenizer, max_length, sample_size)
            valid_dataset = CLDatasetNLI(valid_index, df, tokenizer, max_length)
        else:
            train_dataset = CLDatasetClassification(train_index, df, tokenizer, max_length, sample_size)
            valid_dataset = CLDatasetClassification(valid_index, df, tokenizer, max_length)
        if model_type == "softriple":
            clf_loss = losses_ml.SoftTripleLoss(num_classes=len(unique(df.label)), embedding_size=embedding_size,
                                             centers_per_class=centers, la=la, gamma=gamma, margin=margin)
        elif model_type == "supcon":
            clf_loss = losses_ml.SupConLoss(temperature=supcon_temp)
        elif model_type == "proxynca":
            clf_loss = losses_ml.ProxyNCALoss(len(unique(df.label)), embedding_size=embedding_size,
                                           softmax_scale=softmax_scale)
        elif model_type == "proxyanchor":
            clf_loss = losses_ml.ProxyAnchorLoss(len(unique(df.label)), embedding_size=embedding_size, margin=margin,
                                              alpha=alpha)
        elif model_type == "npairs":
            clf_loss = losses_ml.NPairsLoss()
        elif model_type == "triplet":
            clf_loss = BatchAllTripletLoss()
        elif model_type == "baseline":
            clf_loss = None
        else:
            raise ValueError(
                f'The model_type: {model_type} is not supported. Choose one of following: triple_entropy, supcon, baseline.')
        model = RobertaContrastiveLearning.from_pretrained(model_name,
                                                           num_labels=len(
                                                               unique(
                                                                   df.label)),
                                                           clf_loss=clf_loss,
                                                           beta=beta,
                                                           extended_inference=extended_inference)
        param_groups = [{"params": model.parameters(),
                         'lr': float(learning_rate)}]

        optimizer = transformers.AdamW(param_groups, eps=eps, weight_decay=weight_decay, correct_bias=True)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=num_training_steps)
        optimizers = optimizer, scheduler
        set_seed(seed)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            optimizers=optimizers,
            compute_metrics=compute_metrics
        )

        trainer.train()
        cross_val_res[fold_id] = trainer.evaluate()
        trainer.save_model(args.output_dir + f'{fold_id}_{model_name}_{model_type}_{dataset_name}/model')

    print(f"Model type: {model_type}, Dataset name: {dataset_name}")
    for measure in ["eval_f1_score", "eval_recall_score", "eval_accuracy_score", "eval_precision_score"]:
        print(
            f'measure: {measure.split("_")[1]}, mean: {np.mean([el[measure] for el in cross_val_res.values()])}, std: {np.std([el[measure] for el in cross_val_res.values()])}')


if __name__ == '__main__':
    args = parser.parse_args()

    cross_validate(args)
