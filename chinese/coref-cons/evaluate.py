import os
import json
import random

import torch
import numpy as np
from transformers import BertTokenizer

import metrics
import util
from util import get_predicted_antecedents, evaluate_coref
# from analyse import analyse

from conll_dataloader import CoNLLDataLoader
import conll

from model import CorefModel
import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def analysis_document_length(model, eval_dataloader, data_path, lower, upper, device):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]
    test_doc_keys = []
    for example in examples:
        subtoken_map = example['subtoken_map']
        if lower <= len(subtoken_map) < upper:
            test_doc_keys.append(example['doc_key'])
    model.eval()

    coref_evaluator = metrics.CorefEvaluator()
    with torch.no_grad():
        for i, (batch, example) in enumerate(zip(eval_dataloader, examples)):

            doc_key = batch[0]
            if doc_key[0] not in test_doc_keys:
                continue
            assert doc_key[0] == example["doc_key"], (doc_key, example["doc_key"])
            input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
            sentence_map, subtoken_map = [b.to(device) for b in batch[1:]]
            predictions, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                      cluster_ids, sentence_map, subtoken_map)
            (candidate_cluster_ids, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts,
             top_span_ends, top_antecedents, top_antecedent_scores) = [p.detach().cpu().numpy() for p in predictions]

            antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            _ = evaluate_coref(top_span_starts, top_span_ends, antecedents, example["clusters"], coref_evaluator)

    coref_p, coref_r, coref_f = coref_evaluator.get_prf()

    return coref_p, coref_r, coref_f, len(test_doc_keys)


def analysis_mention_length(gold_spans, predicted_spans, num_mentions_by_boundaries,
                            num_predicted_mentions_by_boundaries, mention_tp_by_boundaries):
    mention_length_boundaries = [(1, 2), (3, 4), (5, 7), (8, 10), (11, 10e5)]
    mentions_by_boundaries = [[] for _ in range(len(mention_length_boundaries))]
    predicted_mentions_by_boundaries = [[] for _ in range(len(mention_length_boundaries))]

    for start, end in gold_spans:
        span_length = end - start + 1
        if 1 <= span_length <= 2:
            mentions_by_boundaries[0].append((start, end))
            num_mentions_by_boundaries[0] += 1
        elif 3 <= span_length <= 4:
            mentions_by_boundaries[1].append((start, end))
            num_mentions_by_boundaries[1] += 1
        elif 5 <= span_length <= 7:
            mentions_by_boundaries[2].append((start, end))
            num_mentions_by_boundaries[2] += 1
        elif 8 <= span_length <= 10:
            mentions_by_boundaries[3].append((start, end))
            num_mentions_by_boundaries[3] += 1
        elif 11 <= span_length:
            mentions_by_boundaries[4].append((start, end))
            num_mentions_by_boundaries[4] += 1

    for start, end in predicted_spans:
        span_length = end - start + 1
        if 1 <= span_length <= 2:
            predicted_mentions_by_boundaries[0].append((start, end))
            num_predicted_mentions_by_boundaries[0] += 1
        elif 3 <= span_length <= 4:
            predicted_mentions_by_boundaries[1].append((start, end))
            num_predicted_mentions_by_boundaries[1] += 1
        elif 5 <= span_length <= 7:
            predicted_mentions_by_boundaries[2].append((start, end))
            num_predicted_mentions_by_boundaries[2] += 1
        elif 8 <= span_length <= 10:
            predicted_mentions_by_boundaries[3].append((start, end))
            num_predicted_mentions_by_boundaries[3] += 1
        elif 11 <= span_length:
            predicted_mentions_by_boundaries[4].append((start, end))
            num_predicted_mentions_by_boundaries[4] += 1

    for i, (mentions, predicted_mentions) in enumerate(zip(mentions_by_boundaries, predicted_mentions_by_boundaries)):
        for start, end in predicted_mentions:
            if (start, end) in mentions:
                mention_tp_by_boundaries[i] += 1


def evaluate(model, eval_dataloader, data_path, conll_path, prediction_path, device):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    mention_length_boundaries = [(1, 2), (3, 4), (5, 7), (8, 10), (11, 10e5)]
    num_mentions_by_boundaries = [0 for _ in range(len(mention_length_boundaries))]
    num_predicted_mentions_by_boundaries = [0 for _ in range(len(mention_length_boundaries))]
    mention_tp_by_boundaries = [0 for _ in range(len(mention_length_boundaries))]

    model.eval()
    coref_predictions = {}
    subtoken_maps = {}
    coref_evaluator = metrics.CorefEvaluator()
    predicted_antecedents = []
    predicted_spans = []
    predicted_clusters = []
    with torch.no_grad():
        for i, (batch, example) in enumerate(zip(eval_dataloader, examples)):
            subtoken_maps[example['doc_key']] = example["subtoken_map"]
            doc_key = batch[0]
            assert doc_key == example["doc_key"], (doc_key, example["doc_key"])
            input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, sentence_map, \
            subtoken_map, graph, constituent_starts, constituent_ends, constituent_labels \
                = [b.to(device) for b in batch[1:]]
            predictions, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                      cluster_ids, sentence_map, subtoken_map, graph, constituent_starts,
                                      constituent_ends, constituent_labels)
            (candidate_cluster_ids, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts,
             top_span_ends, top_antecedents, top_antecedent_scores) = [p.detach().cpu() for p in predictions]

            antecedents = get_predicted_antecedents(top_antecedents.numpy(), top_antecedent_scores.numpy())
            clusters = evaluate_coref(top_span_starts.numpy(), top_span_ends.numpy(), antecedents,
                                      example["clusters"], coref_evaluator)
            spans = [(start, end) for start, end in zip(top_span_starts, top_span_ends)]
            coref_predictions[example["doc_key"]] = clusters
            predicted_antecedents.append(antecedents)
            predicted_spans.append(spans)
            predicted_clusters.append(clusters)

            analysis_mention_length(util.flatten(example['clusters']), util.flatten(clusters),
                                    num_mentions_by_boundaries, num_predicted_mentions_by_boundaries,
                                    mention_tp_by_boundaries)

            if i % 10 == 0:
                logger.info("Evaluated {}/{} examples.".format(i + 1, len(examples)))

    mention_proportion_by_length = []
    for i in range(len(mention_length_boundaries)):
        mention_proportion_by_length.append(num_mentions_by_boundaries[i] / sum(num_mentions_by_boundaries))
    logger.info(mention_proportion_by_length)

    for i in range(len(mention_length_boundaries)):
        boundary_start, boundary_end = mention_length_boundaries[i]
        precision = 100 * mention_tp_by_boundaries[i] / num_predicted_mentions_by_boundaries[i]
        recall = 100 * mention_tp_by_boundaries[i] / num_mentions_by_boundaries[i]
        logger.info(
            "{}-{}: precision ({} / {}) {:.2f}".format(boundary_start, boundary_end, mention_tp_by_boundaries[i],
                                                       num_predicted_mentions_by_boundaries[i], precision))
        logger.info(
            "{}-{}: recall ({} / {}) {:.2f}".format(boundary_start, boundary_end, mention_tp_by_boundaries[i],
                                                    num_mentions_by_boundaries[i], recall))
        logger.info("{}-{}: f1: {:.2f}".format(boundary_start, boundary_end,
                                               2 * precision * recall / (precision + recall)))

    precision = 100 * sum(mention_tp_by_boundaries) / sum(num_predicted_mentions_by_boundaries)
    recall = 100 * sum(mention_tp_by_boundaries) / sum(num_mentions_by_boundaries)
    logger.info("mention detection precision: ({} / {}) {:.2f}".format(sum(mention_tp_by_boundaries),
                                                                       sum(num_predicted_mentions_by_boundaries),
                                                                       precision))
    logger.info("mention detection recall: ({} / {}) {:.2f}".format(sum(mention_tp_by_boundaries),
                                                                    sum(num_mentions_by_boundaries),
                                                                    recall))
    logger.info("mention detection f1: {:.2f}".format(2 * precision * recall / (precision + recall)))

    coref_p, coref_r, coref_f = coref_evaluator.get_prf()
    conll_results = conll.evaluate_conll(conll_path, prediction_path, coref_predictions, subtoken_maps,
                                         official_stdout=True)

    # analysis_results = analyse(examples, predicted_clusters, predicted_spans, predicted_antecedents)

    return coref_p, coref_r, coref_f, conll_results


if __name__ == '__main__':
    config = util.initialize_from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizer.from_pretrained(config['init_checkpoint_dir'])
    dataloader = CoNLLDataLoader(config, tokenizer, mode='train')
    eval_dataloader = dataloader.get_dataloader(data_sign='eval')
    test_dataloader = dataloader.get_dataloader(data_sign='test')

    fh = logging.FileHandler(os.path.join(config['log_dir'], 'coref_eval_log.txt'), mode='w')
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)
    log_dir = config['log_dir']

    model = CorefModel(config, dataloader.cons_tag2id)
    model.load_state_dict(torch.load(config['best_checkpoint']))
    model.to(device)
    model.eval()

    # document_length_boundaries = [(0, 128), (128, 256), (256, 512), (512, 768), (768, 1152), (1152, 10e5)]
    #
    # for lower, upper in document_length_boundaries:
    #     coref_p, coref_r, coref_f, num_documents = analysis_document_length(model, eval_dataloader, config['eval_path'],
    #                                                                         lower, upper, device)
    #     logger.info(
    #         "***** [DEV EVAL COREF] ***** : {} - {} #Docs {} precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
    #             lower, upper, num_documents, coref_p * 100, coref_r * 100, coref_f * 100))
    #
    # for lower, upper in document_length_boundaries:
    #     coref_p, coref_r, coref_f, num_documents = analysis_document_length(model, test_dataloader, config['test_path'],
    #                                                                         lower, upper, device)
    #     logger.info(
    #         "***** [TEST EVAL COREF] ***** : {} - {} #Docs {} precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(
    #             lower, upper, num_documents, coref_p * 100, coref_r * 100, coref_f * 100))

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
    # num_gold_pronouns, num_non_gold, num_gold_span, num_total_span, correct, false_non_gold_span_link, \
    # false_non_anaphoric_link, false_new, wrong_link = analysis_results
    # logger.info("***** [Pronoun Analysis Results] ***** : num_gold_pronouns: {}, num_non_gold: {}, num_gold_span: {}, "
    #             "num_total_span: {}, correct: {}, false_non_gold_span_link: {}, false_non_anaphoric_link: {}, "
    #             "false_new: {}, wrong_link: {}".format(num_gold_pronouns, num_non_gold, num_gold_span, num_total_span,
    #                                                    correct, false_non_gold_span_link, false_non_anaphoric_link,
    #                                                    false_new, wrong_link))

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
    # num_gold_pronouns, num_non_gold, num_gold_span, num_total_span, correct, false_non_gold_span_link, \
    # false_non_anaphoric_link, false_new, wrong_link = test_analysis_results
    # logger.info("***** [Pronoun Analysis Results] ***** : num_gold_pronouns: {}, num_non_gold: {}, num_gold_span: {}, "
    #             "num_total_span: {}, correct: {}, false_non_gold_span_link: {}, false_non_anaphoric_link: {}, "
    #             "false_new: {}, wrong_link: {}".format(num_gold_pronouns,num_non_gold, num_gold_span, num_total_span,
    #                                                    correct, false_non_gold_span_link, false_non_anaphoric_link,
    #                                                    false_new, wrong_link))
