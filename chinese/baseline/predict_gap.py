import os
import json
import random

import torch
import numpy as np
from transformers import BertTokenizer

import util
from util import get_predicted_antecedents, get_predicted_clusters

from conll_dataloader import CoNLLDataLoader

from model import CorefModel
import logging

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict(model, eval_dataloader, data_path, output_path, device):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    model.eval()
    with open(output_path, "w") as output_file:
        with torch.no_grad():
            for i, (batch, example) in enumerate(zip(eval_dataloader, examples)):
                doc_key = batch[0]
                assert doc_key[0] == example["doc_key"], (doc_key, example["doc_key"])
                input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
                sentence_map, subtoken_map = [b.to(device) for b in batch[1:]]

                predictions, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                          cluster_ids, sentence_map, subtoken_map)
                (candidate_cluster_ids, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts,
                 top_span_ends, top_antecedents, top_antecedent_scores) = [p.detach().cpu().numpy() for p in
                                                                           predictions]

                predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
                predicted_clusters, _ = get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
                example["predicted_clusters"] = predicted_clusters
                example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
                output_file.write(json.dumps(example))
                output_file.write("\n")


if __name__ == '__main__':
    config = util.initialize_from_env()
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    dataloader = CoNLLDataLoader(config, tokenizer, mode='train')
    test_dataloader = dataloader.get_dataloader(data_sign='test')

    fh = logging.FileHandler(os.path.join(config['log_dir'], 'coref_eval_log.txt'), mode='w')
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)
    log_dir = config['log_dir']
    output_path = os.path.join(log_dir, 'gap-prediction.jsonlines')

    model = CorefModel(config)
    model.load_state_dict(torch.load(config['best_checkpoint']))
    model.to(device)
    model.eval()

    predict(model, test_dataloader, config['test_path'], output_path, device)
