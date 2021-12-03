import json
import random

import util
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset

from transformers import BertTokenizer, XLNetTokenizer


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
    def __init__(self, features: List[CoNLLCorefResolution], config, sign="train"):
        self.features = features
        self.config = config
        self.sign = sign

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature: CoNLLCorefResolution = self.features[item]
        example = (feature.doc_key, feature.input_ids, feature.input_mask, feature.text_len, feature.speaker_ids,
                   feature.genre, feature.gold_starts, feature.gold_ends, feature.cluster_ids, feature.sentence_map,
                   feature.subtoken_map, feature.token_start_ids, feature.token_end_ids)
        if self.sign == 'train' and len(example[1]) > self.config["max_training_sentences"]:
            example = truncate_example(*example, self.config)

        return convert_to_sliding_window(*example, feature.flatten_sentences, feature.sentence_masks,
                                         feature.sliding_windows, self.config['max_segment_len'])

    def transform(self, doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                  cluster_ids, sentence_map, subtoken_map):
        num_tokens = subtoken_map.size(0)
        token_range = torch.arange(0, num_tokens, dtype=torch.int64)
        word_range = torch.arange(subtoken_map[0], subtoken_map[-1] + 1, dtype=torch.int64)
        word_ids = []
        for i in word_range:
            words = token_range[subtoken_map == i].tolist()
            word_ids.append(words)

        max_word_len = max([len(word) for word in word_ids])
        word_mask = []
        for i, word in enumerate(word_ids):
            word_len = len(word)
            if word_len < max_word_len:
                word_ids[i] += (max_word_len - word_len) * [0]
            word_mask.append([1] * word_len + [0] * (max_word_len - word_len))

        word_ids = torch.tensor(word_ids, dtype=torch.int64)
        word_mask = torch.tensor(word_mask, dtype=torch.int64)

        starts, ends = [], []
        word_start_idx = subtoken_map[0]
        for start, end in zip(gold_starts, gold_ends):
            starts.append(subtoken_map[start] - word_start_idx)
            ends.append(subtoken_map[end] - word_start_idx)

        gold_starts = torch.tensor(starts, dtype=torch.int64)
        gold_ends = torch.tensor(ends, dtype=torch.int64)

        # get candidate mentions
        sentence = []
        speaker_ids = torch.reshape(speaker_ids, [-1])
        flatten_mask = torch.reshape(input_mask, [-1])
        speaker_ids = speaker_ids[flatten_mask > 0]
        speakers = []
        for i in word_range:
            sentence.append(sentence_map[subtoken_map == i][0])
            speakers.append(speaker_ids[subtoken_map == i][0])
        sentence_map = torch.tensor(sentence, dtype=torch.int64)
        speaker_ids = torch.tensor(speakers, dtype=torch.int64)

        num_words = word_ids.size()[0]
        # [num_tokens, max_span_width]
        candidate_starts = torch.arange(0, num_words, dtype=torch.int64).unsqueeze(1).expand(-1, self.config[
            'max_span_width']).contiguous()
        # [num_tokens, max_span_width]
        candidate_ends = candidate_starts + torch.arange(0, self.config['max_span_width'], dtype=torch.int64). \
            unsqueeze(0).expand(num_words, -1).contiguous()
        # [num_tokens * max_span_width]
        candidate_starts = candidate_starts.view(-1).to(sentence_map.device)
        candidate_ends = candidate_ends.view(-1).to(sentence_map.device)
        # [num_tokens * max_span_width]
        candidate_start_sentence_indices = sentence_map[candidate_starts]
        candidate_end_sentence_indices = sentence_map[torch.clamp(candidate_ends, min=0, max=num_words - 1)]
        candidate_mask = torch.logical_and(
            candidate_ends < num_words,
            (candidate_start_sentence_indices - candidate_end_sentence_indices) == 0,
        )

        # [num_candidates]
        candidate_starts = candidate_starts[candidate_mask]
        candidate_ends = candidate_ends[candidate_mask]

        return (doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                cluster_ids, sentence_map, subtoken_map, word_ids, word_mask, candidate_starts, candidate_ends)


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
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == 'eval':
            features = self.convert_examples_to_features(self.config['eval_path'])
            dataset = CoNLLDataset(features, self.config, sign='eval')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.eval_batch_size)
        else:
            features = self.convert_examples_to_features(self.config['test_path'])
            dataset = CoNLLDataset(features, self.config, sign='test')
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader


def tensorize_example(example: dict, config: dict, tokenizer: XLNetTokenizer, genres: dict) -> CoNLLCorefResolution:
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = [0] * len(gold_mentions)
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
    cluster_ids = torch.tensor(cluster_ids, dtype=torch.int64)

    sentences = example["sentences"]
    num_words = sum(len(s) + 2 for s in sentences)
    speakers = example["speakers"]
    speaker_dict = util.get_speaker_dict(util.flatten(speakers), config['max_num_speakers'])

    max_sentence_length = config['max_segment_len']
    text_len = torch.tensor([len(s) for s in sentences], dtype=torch.int64)

    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
        sentence = ['[CLS]'] + sentence + ['[SEP]']
        sent_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sent_input_mask = [-1] + [1] * (len(sent_input_ids) - 2) + [-1]
        speaker_ids += [speaker_dict.get(s, 3) for s in speaker]
        while len(sent_input_ids) < max_sentence_length:
            sent_input_ids.append(tokenizer.pad_token_id)
            sent_input_mask.append(tokenizer.pad_token_id)
        input_ids.append(sent_input_ids)
        input_mask.append(sent_input_mask)
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    input_mask = torch.tensor(input_mask, dtype=torch.int64)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)
    assert num_words == torch.sum((torch.abs(input_mask) != tokenizer.pad_token_id).int()), (num_words, torch.sum(torch.abs(input_mask)))

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
    sentence_masks = [1] * len(flatten_sentences)

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


def truncate_example(doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends,
                     cluster_ids, sentence_map, subtoken_map, token_start_ids, token_end_ids, config):
    max_training_sentences = config["max_training_sentences"]
    num_sentences = input_ids.size()[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0, num_sentences - max_training_sentences)
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    # speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
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


if __name__ == '__main__':
    config = util.initialize_from_env()
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    dataloader = CoNLLDataLoader(config, tokenizer, mode='train')
    train_dataloader = dataloader.get_dataloader(data_sign='train')
    eval_dataloader = dataloader.get_dataloader(data_sign='eval')
    test_dataloader = dataloader.get_dataloader(data_sign='test')

    from tqdm import tqdm

    span_lens = []
    for batch in tqdm(train_dataloader):
        doc_key, input_ids, input_mask, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, \
        sentence_map, subtoken_map = batch
        gold_starts = gold_starts.squeeze(0)
        gold_ends = gold_ends.squeeze(0)
        gold_span_length = (gold_ends - gold_starts + 1).tolist()
        span_lens += gold_span_length

    print('max span len {}'.format(max(span_lens)))
    print('average span len {}'.format(sum(span_lens) / len(span_lens)))

    count = 0
    for length in span_lens:
        if length <= config['max_span_width']:
            count += 1

    print("{}% spans is shorter than {}".format(count / len(span_lens), config['max_span_width']))

    with open(config['train_path']) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    document_lens = []
    for example in examples:
        num_segments = len(example['sentences'])

        document_lens.append(num_segments)

    print("max document len {}".format(max(document_lens)))
    print("average document len {}".format(sum(document_lens) / len(document_lens)))

    max_training_len = config['max_training_sentences']
    count = 0
    for document_length in document_lens:
        if document_length <= max_training_len:
            count += 1

    print("{}% documents is shorter than {}".format(count / len(document_lens), max_training_len))