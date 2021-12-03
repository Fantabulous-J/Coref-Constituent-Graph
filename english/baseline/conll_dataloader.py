import json
import random

import util
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset

from transformers import BertTokenizer


class CoNLLCorefResolution(object):
    def __init__(self, doc_key, input_ids, input_mask, text_len, genre, gold_starts, gold_ends,
                 cluster_ids, sentence_map, subtoken_map, flatten_sentences, sliding_windows, sentence_masks,
                 token_start_ids, token_end_ids):
        self.doc_key = doc_key
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.text_len = text_len
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
    def __init__(self, features: List[CoNLLCorefResolution], config, sign="train"):
        self.features = features
        self.config = config
        self.sign = sign

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature: CoNLLCorefResolution = self.features[item]
        example = (feature.doc_key, feature.input_ids, feature.input_mask, feature.text_len, feature.genre, 
                   feature.gold_starts, feature.gold_ends, feature.cluster_ids, feature.sentence_map,
                   feature.subtoken_map, feature.token_start_ids, feature.token_end_ids)
        if self.sign == 'train' and len(example[1]) > self.config["max_training_sentences"]:
            example = truncate_example(*example, self.config)

        example = convert_to_sliding_window(*example, feature.flatten_sentences, feature.sentence_masks,
                                            feature.sliding_windows, self.config['max_segment_len'])
        return example


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
            dataset = CoNLLDataset(features, self.config, sign='train')
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=collate_fn)
        elif data_sign == 'eval':
            features = self.convert_examples_to_features(self.config['eval_path'])
            dataset = CoNLLDataset(features, self.config, sign='eval')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.eval_batch_size,
                                    collate_fn=collate_fn)
        else:
            features = self.convert_examples_to_features(self.config['test_path'])
            dataset = CoNLLDataset(features, self.config, sign='test')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
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
    input_ids, input_mask= [], []
    for i, (sentence, sentence_mask) in enumerate(zip(sentences, sentence_masks)):
        sentence = ['[CLS]'] + sentence + ['[SEP]']
        sent_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sent_input_mask = [-1] + sentence_mask + [-1]
        while len(sent_input_ids) < max_sentence_length:
            sent_input_ids.append(0)
            sent_input_mask.append(0)
        input_ids.append(sent_input_ids)
        input_mask.append(sent_input_mask)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
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

    return CoNLLCorefResolution(doc_key, input_ids, input_mask, text_len, genre, gold_starts, gold_ends,
                                cluster_ids, sentence_map, subtoken_map, flatten_sentences, sliding_windows,
                                sentence_masks, token_start_ids, token_end_ids)


def convert_to_sliding_window(doc_key, input_ids, input_mask, text_len, genre, gold_starts, gold_ends,
                              cluster_ids, sentence_map, subtoken_map, token_start_ids, token_end_ids,
                              flatten_sentences, sentence_masks, sliding_windows, sliding_window_size):
    num_words = text_len.sum().item()
    token_start_idx, token_end_idx = token_start_ids[0], token_end_ids[-1]
    token_windows = []
    mask_windows = []
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
    return (doc_key, token_windows, mask_windows, text_len, genre, gold_starts, gold_ends, cluster_ids,
            sentence_map, subtoken_map)


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []

    starts = torch.tensor(starts, dtype=torch.int64)
    ends = torch.tensor(ends, dtype=torch.int64)
    return starts, ends


def truncate_example(doc_key, input_ids, input_mask, text_len, genre, gold_starts, gold_ends,
                     cluster_ids, sentence_map, subtoken_map, token_start_ids, token_end_ids, config):
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
    token_start_ids = token_start_ids[sentence_offset:sentence_offset + max_training_sentences]
    token_end_ids = token_end_ids[sentence_offset:sentence_offset + max_training_sentences]

    return (doc_key, input_ids, input_mask, text_len, genre, gold_starts, gold_ends, cluster_ids,
            sentence_map, subtoken_map, token_start_ids, token_end_ids)


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


def collate_fn(data):
    data = zip(*data)
    data = [x[0] for x in data]
    return data
