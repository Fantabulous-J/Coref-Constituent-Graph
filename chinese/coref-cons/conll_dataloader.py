import json
import random
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from transformers import BertTokenizer
import dgl

import util
import tree


class CoNLLCorefResolution(object):
    def __init__(self, doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                 cluster_ids, sentence_map, subtoken_map, flatten_sentences, sliding_windows, sentence_masks,
                 token_start_ids, token_end_ids):
        self.doc_key = doc_key
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.text_len = text_len
        self.speaker_ids = speaker_ids
        self.genre = genre
        self.gold_starts = gold_starts
        self.gold_ends = gold_ends
        self.cluster_ids = cluster_ids
        self.sentence_map = sentence_map
        self.subtoken_map = subtoken_map
        self.flatten_sentences = flatten_sentences
        self.sliding_windows = sliding_windows
        self.sentence_masks = sentence_masks
        self.token_start_ids = token_start_ids
        self.token_end_ids = token_end_ids


class CoNLLDataset(Dataset):
    def __init__(self, features: List[CoNLLCorefResolution], tokenizer: BertTokenizer, config,
                 constituents, cons_tag2id, sign="train"):
        self.features = features
        self.config = config
        self.sign = sign
        self.tokenizer = tokenizer
        self.constituents = constituents
        self.cons_tag2id = cons_tag2id

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature: CoNLLCorefResolution = self.features[item]
        example = (feature.doc_key, feature.input_ids, feature.input_mask, feature.text_len, feature.speaker_ids,
                   feature.genre, feature.gold_starts, feature.gold_ends, feature.cluster_ids, feature.sentence_map,
                   feature.subtoken_map, feature.token_start_ids, feature.token_end_ids)
        if self.sign == 'train' and len(example[1]) > self.config["max_training_sentences"]:
            example = truncate_example(*example, self.config)

        example = convert_to_sliding_window(*example, feature.flatten_sentences, feature.sentence_masks,
                                            feature.sliding_windows, self.config['max_segment_len'])
        example = self.create_graph(*example, self.constituents[item])
        return example

    def create_graph(self, doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                     cluster_ids, sentence_map, subtoken_map, constituent):
        graph = dgl.graph([])
        graph.set_n_initializer(dgl.init.zero_initializer)
        sentence_start_idx = sentence_map[0]
        sentence_end_idx = sentence_map[-1]
        num_tokens = sentence_map.size()[0]
        token_range = torch.arange(0, num_tokens, dtype=torch.int64)

        graph.add_nodes(num_tokens)
        graph.ndata['unit'] = torch.zeros(num_tokens)
        graph.ndata['dtype'] = torch.zeros(num_tokens)
        num_prev_nodes = (graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)).size()

        num_new_nodes = (graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)).size()
        assert num_prev_nodes == num_new_nodes, (num_prev_nodes, num_new_nodes)

        # constituents
        constituent = constituent[sentence_start_idx: sentence_end_idx + 1]

        constituent_idx_dict = {}
        for sent_cons in constituent:
            for idx, start, end, label, parent_idx in sent_cons[0]:
                if subtoken_map[0] <= start <= subtoken_map[-1] and subtoken_map[0] <= end <= subtoken_map[-1]:
                    constituent_idx_dict[idx] = len(constituent_idx_dict)

        constituent_start_idx = constituent[0][0][0][0]
        # num_cons = sum([len(sent_cons[0]) for sent_cons in constituent])
        num_cons = len(constituent_idx_dict)
        graph.add_nodes(num_cons)
        node_id_offset = num_tokens
        graph.ndata['unit'][node_id_offset:] = torch.ones(num_cons)
        graph.ndata['dtype'][node_id_offset:] = torch.ones(num_cons)
        prev_num_cons_nodes = (graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)).size()
        constituent_starts = []
        constituent_ends = []
        constituent_labels = []
        prev_root_node_id = None
        forward_edge_type, backward_edge_type = 0, 2
        for high_order_sent_cons in constituent:
            for i, sent_cons in enumerate(high_order_sent_cons):
                for idx, start, end, label, parent_idx in sent_cons:
                    if idx not in constituent_idx_dict:
                        continue
                    idx_nodeid = constituent_idx_dict[idx] + node_id_offset
                    token_start = token_range[subtoken_map == start][0]
                    token_end = token_range[subtoken_map == end][-1]
                    if parent_idx == -1 and parent_idx in constituent_idx_dict:
                        if prev_root_node_id is not None:
                            graph.add_edges(prev_root_node_id, idx_nodeid,
                                            data={'cc_link': torch.tensor([forward_edge_type + i]),
                                                  'dtype': torch.tensor([forward_edge_type + i])})
                            # dual GAT
                            graph.add_edges(idx_nodeid, prev_root_node_id,
                                            data={'cc_link': torch.tensor([backward_edge_type + i]),
                                                  'dtype': torch.tensor([backward_edge_type + i])})
                        prev_root_node_id = idx_nodeid

                    if parent_idx != -1 and parent_idx in constituent_idx_dict:
                        parent_idx_nodeid = constituent_idx_dict[parent_idx] + node_id_offset
                        graph.add_edges(parent_idx_nodeid, idx_nodeid,
                                        data={'cc_link': torch.tensor([forward_edge_type + i]),
                                              'dtype': torch.tensor([forward_edge_type + i])})
                        graph.add_edges(idx_nodeid, parent_idx_nodeid,
                                        data={'cc_link': torch.tensor([backward_edge_type + i]),
                                              'dtype': torch.tensor([backward_edge_type + i])})

                    if i == 0:
                        # self-loop edge
                        graph.add_edges(idx_nodeid, idx_nodeid, data={'cc_link': torch.tensor([4]),
                                                                      'dtype': torch.tensor([4])})
                        # constituent -> token
                        graph.add_edges(idx_nodeid, token_start, data={'ct_link': torch.tensor([5]),
                                                                       'dtype': torch.tensor([5])})
                        graph.add_edges(idx_nodeid, token_end, data={'ct_link': torch.tensor([5]),
                                                                     'dtype': torch.tensor([5])})
                        constituent_starts.append(token_start)
                        constituent_ends.append(token_end)
                        constituent_labels.append(self.cons_tag2id[label])

        current_num_cons_nodes = (graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)).size()
        assert prev_num_cons_nodes == current_num_cons_nodes, (prev_num_cons_nodes, current_num_cons_nodes)
        assert len(constituent_starts) == len(constituent_ends) == len(constituent_labels) == len(constituent_idx_dict)\
            , (len(constituent_starts), len(constituent_ends), len(constituent_labels), len(constituent_idx_dict))
        current_num_token_nodes = (graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)).size()
        assert current_num_token_nodes == num_new_nodes, (current_num_token_nodes, num_new_nodes,
                                                          graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0),
                                                          constituent_starts, constituent_ends, constituent_start_idx)
        constituent_starts = torch.tensor(constituent_starts, dtype=torch.int64)
        constituent_ends = torch.tensor(constituent_ends, dtype=torch.int64)
        constituent_labels = torch.tensor(constituent_labels, dtype=torch.int64)

        return (doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
                sentence_map, subtoken_map, graph, constituent_starts, constituent_ends, constituent_labels)


class CoNLLDataLoader(object):
    def __init__(self, config, tokenizer, mode="train"):
        if mode == "train":
            self.train_batch_size = 1
            self.eval_batch_size = 1
            self.test_batch_size = 1
        else:
            self.test_batch_size = 1

        self.config = config
        self.tokenizer = tokenizer
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.cons_tag2id = util.get_cons_tag_vocab(config['cons_vocab_file'])

    def convert_examples_to_features(self, data_path):
        with open(data_path) as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]

        data_instances = []
        for example in examples:
            data_instances.append(tensorize_example(example, self.config, self.tokenizer, self.genres))

        return data_instances

    def get_dataloader(self, data_sign="train"):
        if data_sign == 'train':
            features = self.convert_examples_to_features(self.config['train_path'])
            constituents = tree.read_constituents(self.config['train_cons_path'])
            dataset = CoNLLDataset(features, self.tokenizer, self.config, constituents, self.cons_tag2id, sign='train')
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size, num_workers=16,
                                    collate_fn=collate_fn)
        elif data_sign == 'eval':
            features = self.convert_examples_to_features(self.config['eval_path'])
            constituents = tree.read_constituents(self.config['eval_cons_path'])
            dataset = CoNLLDataset(features, self.tokenizer, self.config, constituents, self.cons_tag2id, sign='eval')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.eval_batch_size, num_workers=16,
                                    collate_fn=collate_fn)
        else:
            features = self.convert_examples_to_features(self.config['test_path'])
            constituents = tree.read_constituents(self.config['test_cons_path'])
            dataset = CoNLLDataset(features, self.tokenizer, self.config, constituents, self.cons_tag2id, sign='test')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size, num_workers=16,
                                    collate_fn=collate_fn)

        return dataloader


def tensorize_example(example: dict, config: dict, tokenizer: BertTokenizer, genres: dict) -> CoNLLCorefResolution:
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = [0] * len(gold_mentions)
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
    cluster_ids = torch.tensor(cluster_ids, dtype=torch.int64)

    sentences = example["sentences"]
    sentence_masks = example['segment_mask']
    sentence_lens = example['segment_len']
    num_words = sum([length for length in sentence_lens])
    speakers = example["speakers"]
    speaker_dict = util.get_speaker_dict(util.flatten(speakers), config['max_num_speakers'])

    max_sentence_length = config['max_segment_len']
    text_len = torch.tensor(sentence_lens, dtype=torch.int64)

    # speaker mask: -2; [CLS], [SEP] mask: -1; [PAD] mask: 0; sliding window mask: -3
    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, sentence_mask, speaker) in enumerate(zip(sentences, sentence_masks, speakers)):
        sentence = ['[CLS]'] + sentence + ['[SEP]']
        sent_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sent_input_mask = [-1] + sentence_mask + [-1]
        speaker_ids += [speaker_dict.get(s, 3) for s in speaker]
        while len(sent_input_ids) < max_sentence_length:
            sent_input_ids.append(0)
            sent_input_mask.append(0)
        input_ids.append(sent_input_ids)
        input_mask.append(sent_input_mask)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)
    input_mask = torch.tensor(input_mask, dtype=torch.int64)
    assert num_words == torch.sum(input_mask == 1), (num_words, torch.sum(input_mask == 1), example["doc_key"])

    doc_key = example["doc_key"]
    subtoken_map = torch.tensor(example.get("subtoken_map", None), dtype=torch.int64)
    sentence_map = torch.tensor(example['sentence_map'], dtype=torch.int64)
    genre = genres.get(doc_key[:2], 0)
    genre = torch.tensor([genre], dtype=torch.int64)
    gold_starts, gold_ends = tensorize_mentions(gold_mentions)
    token_start_ids, token_end_ids = [], []
    for i in range(len(sentences)):
        if i == 0:
            token_start_ids.append(0)
            token_end_ids.append(len(sentences[i]) - 1)
        else:
            token_start_ids.append(len(sentences[i - 1]) + token_start_ids[-1])
            token_end_ids.append(len(sentences[i]) + token_end_ids[-1])

    flatten_sentences = util.flatten(sentences)
    flatten_sentences = tokenizer.convert_tokens_to_ids(flatten_sentences)
    sentence_masks = util.flatten(sentence_masks)

    sliding_windows = construct_sliding_windows(len(flatten_sentences), max_sentence_length - 2)

    return CoNLLCorefResolution(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                                cluster_ids, sentence_map, subtoken_map, flatten_sentences, sliding_windows,
                                sentence_masks, token_start_ids, token_end_ids)


def convert_to_sliding_window(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                              cluster_ids, sentence_map, subtoken_map, token_start_ids, token_end_ids,
                              flatten_sentences, sentence_masks, sliding_windows, sliding_window_size):
    num_words = text_len.sum().item()
    token_start_idx, token_end_idx = token_start_ids[0], token_end_ids[-1]
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    text_len = []

    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = flatten_sentences[window_start: window_end]
        original_masks = sentence_masks[window_start: window_end]

        window_mask = [0 if (i + window_start < token_start_idx or i + window_start > token_end_idx) and m == 1 else m
                       for i, m in enumerate(window_mask)]
        if sum(window_mask) == 0:
            continue
        window_masks = [-3 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = [101] + original_tokens + [102] + [0] * (
                sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-1] + window_masks + [-1] + [0] * (sliding_window_size - 2 - len(original_tokens))
        token_calculate = [tmp for tmp in one_window_mask if tmp > 0]
        text_len.append(len(token_calculate))
        assert len(one_window_token) == sliding_window_size, (len(one_window_token), sliding_window_size)
        assert len(one_window_mask) == sliding_window_size, (len(one_window_mask), sliding_window_size)
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    assert num_words == sum([i > 0 for j in mask_windows for i in j]), \
        (num_words, sum([i > 0 for j in mask_windows for i in j]), doc_key, token_start_ids, token_end_ids)

    text_len = torch.tensor(text_len, dtype=torch.int64)
    token_windows = torch.tensor(token_windows, dtype=torch.int64)
    mask_windows = torch.tensor(mask_windows, dtype=torch.int64)
    return (doc_key, token_windows, mask_windows, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
            sentence_map, subtoken_map)


def construct_sliding_windows(sequence_length, sliding_window_size):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []

    starts = torch.tensor(starts, dtype=torch.int64)
    ends = torch.tensor(ends, dtype=torch.int64)
    return starts, ends


def truncate_example(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
                     sentence_map, subtoken_map, token_start_ids, token_end_ids, config):
    max_training_sentences = config["max_training_sentences"]
    num_sentences = input_ids.size()[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    sentence_map = sentence_map[word_offset: word_offset + num_words]
    subtoken_map = subtoken_map[word_offset: word_offset + num_words]
    gold_spans = (gold_ends >= word_offset) & (gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]
    speaker_ids = speaker_ids[word_offset: word_offset + num_words]

    token_start_ids = token_start_ids[sentence_offset:sentence_offset + max_training_sentences]
    token_end_ids = token_end_ids[sentence_offset:sentence_offset + max_training_sentences]

    return (doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids,
            sentence_map, subtoken_map, token_start_ids, token_end_ids)


def collate_fn(data):
    data = zip(*data)
    data = [x[0] for x in data]
    return data


if __name__ == '__main__':
    config = util.initialize_from_env()
    tokenizer = BertTokenizer.from_pretrained(config['init_checkpoint_dir'])
    dataloader = CoNLLDataLoader(config, tokenizer, mode='train')
    train_dataloader = dataloader.get_dataloader(data_sign='train')
    eval_dataloader = dataloader.get_dataloader(data_sign='eval')
    test_dataloader = dataloader.get_dataloader(data_sign='test')

    from tqdm import tqdm

    for batch in tqdm(train_dataloader):
        # doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
        # sentence_map, subtoken_map, graph = batch
        # token_node_ids = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        # print(types)
        # if sentence_map.size()[0] != token_node_ids.size()[0]:
        #     print(token_node_ids)
        #     print(speaker_ids)
        doc_key = batch[0]

    for batch in tqdm(eval_dataloader):
        # doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
        # sentence_map, subtoken_map, graph = batch
        # token_node_ids = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        doc_key = batch[0]

    for batch in tqdm(test_dataloader):
        # doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
        # sentence_map, subtoken_map, graph = batch
        # token_node_ids = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 0)
        doc_key = batch[0]
