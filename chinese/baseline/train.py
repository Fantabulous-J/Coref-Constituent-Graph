import os
import json
import random
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import metrics
import util
from util import get_predicted_antecedents, evaluate_coref
from conll_dataloader import CoNLLDataLoader

from transformers import BertTokenizer
from model import CorefModel
from optimization import build_optimizer
from poly_lr_decay import PolynomialLRDecay

import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def warmup_linear(optimizer, config, step, num_warmup_steps):
    lr = config['bert_learning_rate'] * (step / num_warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    config = util.initialize_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizer.from_pretrained(config['init_checkpoint_dir'])
    # tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    dataloader = CoNLLDataLoader(config, tokenizer, mode='train')
    train_dataloader = dataloader.get_dataloader(data_sign='train')
    eval_dataloader = dataloader.get_dataloader(data_sign='eval')
    test_dataloader = dataloader.get_dataloader(data_sign='test')

    fh = logging.FileHandler(os.path.join(config['log_dir'], 'coref_log.txt'), mode='w')
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)
    log_dir = config['log_dir']

    best_dev_f1, best_dev_pre, best_dev_recall = 0.0, 0.0, 0.0
    test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = 0.0, 0.0, 0.0

    model = CorefModel(config, tokenizer)
    bert_optimizer, task_optimizer = build_optimizer(model, config)

    num_train_steps = int(config['num_docs'] * config['num_epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    bert_poly_decay_scheduler = PolynomialLRDecay(optimizer=bert_optimizer,
                                                  max_decay_steps=num_train_steps,
                                                  end_learning_rate=0.0,
                                                  power=1.0)
    task_poly_decay_scheduler = PolynomialLRDecay(optimizer=task_optimizer,
                                                  max_decay_steps=num_train_steps,
                                                  end_learning_rate=0.0,
                                                  power=1.0)

    step = 0
    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]
    writer = SummaryWriter(log_dir=log_dir)
    accumulated_loss = 0.0
    model.to(device)
    model.train()
    for epoch in range(config['num_epochs']):
        logger.info("=*=" * 20)
        logger.info("start {} Epoch ... ".format(str(epoch)))

        for i, batch in enumerate(train_dataloader):
            doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
            sentence_map, subtoken_map = batch
            input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
            sentence_map, subtoken_map = \
                input_ids.to(device), input_mask.to(device), text_len.to(device), \
                speaker_ids.to(device), genre.to(device), gold_starts.to(device), \
                gold_ends.to(device), cluster_ids.to(device), sentence_map.to(device), \
                subtoken_map.to(device)
            bert_optimizer.zero_grad()
            task_optimizer.zero_grad()
            _, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                            cluster_ids, sentence_map, subtoken_map)
            accumulated_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            bert_optimizer.step()
            task_optimizer.step()

            if step > 0 and step % report_frequency == 0:
                average_loss = accumulated_loss / report_frequency
                logger.info("[{}] loss={:.2f}".format(step, average_loss))
                writer.add_scalar('Loss', average_loss, step)
                accumulated_loss = 0.0

            if step > 0 and step % eval_frequency == 0:
                coref_p, coref_r, coref_f = evaluate(model, eval_dataloader, config['eval_path'], device)
                logger.info("***** EVAL ON DEV SET *****")
                logger.info(
                    "***** [DEV EVAL COREF] ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(coref_p * 100,
                                                                                                          coref_r * 100,
                                                                                                          coref_f * 100))
                writer.add_scalar('Dev/F1', coref_f, step)
                writer.add_scalar('Dev/Precision', coref_p, step)
                writer.add_scalar('Dev/Recall', coref_r, step)
                if coref_f > best_dev_f1:
                    best_dev_f1 = coref_f
                    best_dev_pre = coref_p
                    best_dev_recall = coref_r
                    test_coref_p, test_coref_r, test_coref_f = \
                        evaluate(model, test_dataloader, config['test_path'], device)
                    test_f1_when_dev_best, test_prec_when_dev_best, test_rec_when_dev_best = test_coref_f, \
                                                                                             test_coref_p, \
                                                                                             test_coref_r
                    logger.info("***** EVAL ON TEST SET *****")
                    logger.info(
                        "***** [TEST EVAL COREF] ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
                            test_coref_p * 100,
                            test_coref_r * 100,
                            test_coref_f * 100))

                    logger.info("***** SAVE MODEL *****")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(log_dir, "model_best.checkpoint"))
                    writer.add_scalar('Best/dev/f1', best_dev_f1, step)
                    writer.add_scalar('Best/dev/p', best_dev_pre, step)
                    writer.add_scalar('Best/dev/r', best_dev_recall, step)
                    writer.add_scalar('Best/test/f1', test_coref_f, step)
                    writer.add_scalar('Best/test/p', test_coref_p, step)
                    writer.add_scalar('Best/test/r', test_coref_r, step)

                model.train()

            step += 1
            if step < num_warmup_steps:
                bert_lr = warmup_linear(bert_optimizer, config, step + 1, num_warmup_steps)
            else:
                bert_poly_decay_scheduler.step(step)
                bert_lr = bert_poly_decay_scheduler.get_last_lr()[0]
            task_poly_decay_scheduler.step()
            task_lr = task_poly_decay_scheduler.get_last_lr()[0]
            writer.add_scalar('Bert Learning Rate', bert_lr, step)
            writer.add_scalar('Task Learning Rate', task_lr, step)

    logger.info("*" * 20)
    logger.info(
        "- @@@@@ BEST DEV F1 : {:.2f}, Precision : {:.2f}, Recall : {:.2f},".format(best_dev_f1 * 100,
                                                                                    best_dev_pre * 100,
                                                                                    best_dev_recall * 100))
    logger.info("- @@@@@ TEST when DEV best F1 : {:.2f}, Precision : {:.2f}, Recall : {:.2f},".format(
        test_f1_when_dev_best * 100, test_prec_when_dev_best * 100, test_rec_when_dev_best * 100))


def evaluate(model, eval_dataloader, data_path, device):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    model.eval()
    mention_tp = 0
    mention_fp = 0
    mention_fn = 0
    epsilon = 1e-10
    coref_evaluator = metrics.CorefEvaluator()
    with torch.no_grad():
        for i, (batch, example) in enumerate(zip(eval_dataloader, examples)):
            doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, \
            cluster_ids, sentence_map, subtoken_map = batch
            assert doc_key[0] == example["doc_key"], (doc_key, example["doc_key"])
            input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
            sentence_map, subtoken_map = \
                input_ids.to(device), input_mask.to(device), text_len.to(device), \
                speaker_ids.to(device), genre.to(device), gold_starts.to(device), \
                gold_ends.to(device), cluster_ids.to(device), sentence_map.to(device), \
                subtoken_map.to(device)

            predictions, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                      cluster_ids, sentence_map, subtoken_map)
            (candidate_cluster_ids, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts,
             top_span_ends, top_antecedents, top_antecedent_scores, top_span_cluster_ids,
             top_span_mention_scores) = [p.detach().cpu() for p in predictions]

            predicted_antecedents = get_predicted_antecedents(top_antecedents.numpy(), top_antecedent_scores.numpy())
            predicted_clusters = evaluate_coref(top_span_starts.numpy(), top_span_ends.numpy(), predicted_antecedents,
                                                example["clusters"],
                                                coref_evaluator)
            # (candidate_mention_scores, candidate_starts, candidate_ends) = [p.detach().cpu() for p in predictions]
            # gold_starts = gold_starts[0].detach().cpu().tolist()
            # gold_ends = gold_ends[0].detach().cpu().tolist()
            # gold_spans = list(zip(gold_starts, gold_ends))
            # candidate_starts = candidate_starts.tolist()
            # candidate_ends = candidate_ends.tolist()
            # candidate_mention_scores = candidate_mention_scores.tolist()
            # for mention_score, candidate_start, candidate_end in zip(candidate_mention_scores, candidate_starts, candidate_ends):
            #     if mention_score > 0:
            #         if (candidate_start, candidate_end) in gold_spans:
            #             mention_tp += 1
            #         else:
            #             mention_fp += 1
            #     else:
            #         if (candidate_start, candidate_end) in gold_spans:
            #             mention_fn += 1
            # mention_tp += torch.logical_and(predict_labels > 0.5, candidate_cluster_ids > 0).sum()
            # mention_fp += torch.logical_and(predict_labels > 0.5, candidate_cluster_ids == 0).sum()
            # mention_fn += torch.logical_and(predict_labels <= 0.5, candidate_cluster_ids > 0).sum()
    coref_p, coref_r, coref_f = coref_evaluator.get_prf()
    # mention_p = mention_tp / (mention_tp + mention_fp + epsilon)
    # mention_r = mention_tp / (mention_tp + mention_fn + epsilon)
    # mention_f = 2 * mention_p * mention_r / (mention_p + mention_r + epsilon)

    # logger.info(
    #     "***** [EVAL MENTION] ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
    #         mention_p * 100,
    #         mention_r * 100,
    #         mention_f * 100))
    #
    return coref_p, coref_r, coref_f
    # return mention_p, mention_r, mention_f


if __name__ == '__main__':
    train()
