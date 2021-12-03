import os
import json
import random

import torch
import numpy as np
from transformers import BertTokenizer

import metrics
import util
from util import get_predicted_antecedents, evaluate_coref

from conll_dataloader import CoNLLDataLoader
import conll

from model import CorefModel
import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate(model, eval_dataloader, data_path, conll_path, prediction_path, device):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    model.eval()
    coref_predictions = {}
    subtoken_maps = {}
    coref_evaluator = metrics.CorefEvaluator(singleton=False)

    with torch.no_grad():
        for i, (batch, example) in enumerate(zip(eval_dataloader, examples)):
            subtoken_maps[example['doc_key']] = example["subtoken_map"]
            doc_key = batch[0]
            assert doc_key == example["doc_key"], (doc_key, example["doc_key"])
            input_ids, input_mask, text_len, genre, gold_starts, gold_ends, cluster_ids, sentence_map, subtoken_map \
                = [b.to(device) for b in batch[1:]]

            predictions, loss = model(input_ids, input_mask, text_len, genre, gold_starts, gold_ends, cluster_ids,
                                      sentence_map, subtoken_map)
            (top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores, candidate_starts, candidate_ends,
             top_span_cluster_ids, top_span_mention_scores, candidate_mention_scores) = \
                [p.detach().cpu() for p in predictions]

            antecedents = get_predicted_antecedents(top_antecedents.numpy(), top_antecedent_scores.numpy())
            clusters = evaluate_coref(top_span_starts.numpy(), top_span_ends.numpy(), antecedents,
                                      example["clusters"], coref_evaluator, top_span_mention_scores, singleton=True)
            coref_predictions[example["doc_key"]] = clusters

    coref_p, coref_r, coref_f = coref_evaluator.get_prf()
    conll_results = conll.evaluate_conll(conll_path, prediction_path, coref_predictions, subtoken_maps,
                                         official_stdout=True)

    return coref_p, coref_r, coref_f, conll_results


if __name__ == '__main__':
    config = util.initialize_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    dataloader = CoNLLDataLoader(config, tokenizer, mode='train')
    eval_dataloader = dataloader.get_dataloader(data_sign='eval')
    test_dataloader = dataloader.get_dataloader(data_sign='test')

    fh = logging.FileHandler(os.path.join(config['log_dir'], 'coref_eval_log.txt'), mode='w')
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)
    log_dir = config['log_dir']

    model = CorefModel(config)
    model.load_state_dict(torch.load(config['best_checkpoint']))
    model.to(device)
    model.eval()

    dev_prediction_file = os.path.join(log_dir, "dev_conll_eval_results.txt")
    # eval on dev set
    coref_p, coref_r, coref_f, dev_conll_results = evaluate(model, eval_dataloader,
                                                            config['eval_path'],
                                                            config['conll_dev_path'],
                                                            dev_prediction_file,
                                                            device)
    logger.info("***** EVAL ON DEV SET *****")
    logger.info(
        "***** [DEV EVAL COREF] ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(coref_p * 100,
                                                                                              coref_r * 100,
                                                                                              coref_f * 100))
    dev_average_f1 = sum(results["f"] for results in dev_conll_results.values()) / len(dev_conll_results)

    logger.info("***** CONLL OFFICIAL DEV RESULTS *****")
    for metric, results in dev_conll_results.items():
        logger.info("***** OFFICIAL RESULTS FOR {} ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
            metric.upper(), results['p'], results['r'], results['f']
        ))
    logger.info("***** [AVERAGE F1] ***** : {:.2f}".format(dev_average_f1))

    # eval on test set
    test_prediction_file = os.path.join(log_dir, "test_conll_eval_results.txt")
    test_coref_p, test_coref_r, test_coref_f, test_conll_results = \
        evaluate(model, test_dataloader, config['test_path'], config['conll_test_path'], test_prediction_file, device)
    logger.info("***** EVAL ON TEST SET *****")
    logger.info(
        "***** [TEST EVAL COREF] ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
            test_coref_p * 100,
            test_coref_r * 100,
            test_coref_f * 100))

    test_average_f1 = sum(results["f"] for results in test_conll_results.values()) / len(test_conll_results)

    logger.info("***** CONLL OFFICIAL TEST RESULTS *****")
    for metric, results in test_conll_results.items():
        logger.info("***** OFFICIAL RESULTS FOR {} ***** : precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
            metric.upper(), results['p'], results['r'], results['f']
        ))
    logger.info("***** [AVERAGE F1] ***** : {:.2f}".format(test_average_f1))
