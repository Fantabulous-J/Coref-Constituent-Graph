from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import re
import os
import sys
import json
import tempfile
import subprocess
import collections

import util
import conll
from transformers import BertTokenizer

SPEAKER_START = '[unused19]'
SPEAKER_END = '[unused73]'


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.speakers = []
        self.segment_info = []
        self.pos_tags = []

    def finalize(self):
        # finalized: segments, segment_subtoken_map
        # populate speakers from info
        subtoken_idx = 0
        for segment in self.segment_info:
            speakers = []
            for i, tok_info in enumerate(segment):
                # if tok_info is None and (i == 0 or i == len(segment) - 1):
                #     speakers.append('[SPL]')
                if tok_info is None:
                    speakers.append(speakers[-1])
                else:
                    speakers.append(tok_info[9])
                    if 'PRP' in tok_info[4] and tok_info[-3] != "-":
                        self.pronouns.append(subtoken_idx)
                subtoken_idx += 1
            self.speakers += [speakers]

        # populate srl from info
        srl = []
        first_subtoken_index = -1
        argument_stacks, argument_buffers = [[]], [[]]
        predicate_buffer = []
        for seg_idx, segment in enumerate(self.segment_info):
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                if tok_info is None:
                    continue
                predicate_sense = tok_info[7]
                args = tok_info[11:-3]
                # print(args)
                if tok_info[-1]:
                    for predicate, args_buffer in zip(predicate_buffer, argument_buffers):
                        for arg_start, arg_end, label in args_buffer:
                            if label in ["V", "C-V"]:
                                continue
                            srl.append((predicate, arg_start, arg_end, label))
                    predicate_buffer = []
                    argument_stacks = [[] for _ in args]
                    argument_buffers = [[] for _ in args]
                for j, arg in enumerate(args):
                    asterisk_idx = arg.find("*")
                    if asterisk_idx >= 0:
                        open_parens = arg[:asterisk_idx]
                        close_parens = arg[asterisk_idx + 1:]
                    else:
                        open_parens = arg[:-1]
                        close_parens = arg[-1]
                    current_idx = open_parens.find("(")
                    while current_idx >= 0:
                        next_idx = open_parens.find("(", current_idx + 1)
                        if next_idx >= 0:
                            label = open_parens[current_idx + 1:next_idx]
                        else:
                            label = open_parens[current_idx + 1:]
                        argument_stacks[j].append((first_subtoken_index, label))
                        current_idx = next_idx
                    for c in close_parens:
                        try:
                            assert c == ")"
                        except AssertionError:
                            # print(first_subtoken_index, arg, argument_buffers, argument_stacks)
                            continue
                        start, label = argument_stacks[j].pop()
                        argument_buffers[j].append((start, first_subtoken_index + tok_info[-2] - 1, label))
                if predicate_sense != '-':
                    # predicate_buffer.append((first_subtoken_index, first_subtoken_index + tok_info[-2] - 1))
                    predicate_buffer.append(first_subtoken_index)

        if len(predicate_buffer) != 0:
            for predicate, args_buffer in zip(predicate_buffer, argument_buffers):
                for arg_start, arg_end, label in args_buffer:
                    if label in ["V", "C-V"]:
                        continue
                    srl.append((predicate, arg_start, arg_end, label))

        # populate ner from info
        first_subtoken_index = -1
        ners = []
        ner_stacks, ner_buffers = [], []
        for seg_idx, segment in enumerate(self.segment_info):
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                if tok_info is None:
                    continue
                ner = tok_info[10]
                if tok_info[-1]:
                    for start, end, label in ner_buffers:
                        ners.append([start, end, label])
                    ner_buffers = []
                    ner_stacks = []
                asterisk_idx = ner.find("*")
                if asterisk_idx >= 0:
                    open_parens = ner[:asterisk_idx]
                    close_parens = ner[asterisk_idx + 1:]
                else:
                    open_parens = ner[:-1]
                    close_parens = ner[-1]
                current_idx = open_parens.find("(")
                while current_idx >= 0:
                    next_idx = open_parens.find("(", current_idx + 1)
                    if next_idx >= 0:
                        label = open_parens[current_idx + 1:next_idx]
                    else:
                        label = open_parens[current_idx + 1:]
                    ner_stacks.append((first_subtoken_index, label))
                    current_idx = next_idx
                for c in close_parens:
                    try:
                        assert c == ")"
                    except AssertionError:
                        continue
                    start, label = ner_stacks.pop()
                    ner_buffers.append((start, first_subtoken_index + tok_info[-2] - 1, label))

        if len(ner_buffers) != 0:
            for start, end, label in ner_buffers:
                ners.append([start, end, label])

        # populate clusters
        first_subtoken_index = -1
        for seg_idx, segment in enumerate(self.segment_info):
            speakers = []
            for i, tok_info in enumerate(segment):
                first_subtoken_index += 1
                coref = tok_info[-3] if tok_info is not None else '-'
                if coref != "-":
                    last_subtoken_index = first_subtoken_index + tok_info[-2] - 1
                    for part in coref.split("|"):
                        if part[0] == "(":
                            if part[-1] == ")":
                                cluster_id = int(part[1:-1])
                                self.clusters[cluster_id].append((first_subtoken_index, last_subtoken_index))
                            else:
                                cluster_id = int(part[1:])
                                self.coref_stacks[cluster_id].append(first_subtoken_index)
                        else:
                            cluster_id = int(part[:-1])
                            start = self.coref_stacks[cluster_id].pop()
                            self.clusters[cluster_id].append((start, last_subtoken_index))
        # merge clusters
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)
        assert len(all_mentions) == len(set(all_mentions))
        num_words = len(util.flatten(self.segments))
        assert num_words == len(util.flatten(self.speakers))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        assert num_words == len(self.pos_tags), (num_words, len(self.pos_tags))
        # expanded_ners = {}
        # for cluster in merged_clusters:
        #     populated_ner_label = "EMPTY"
        #     for start, end in cluster:
        #         for ner_start, ner_end, label in ners:
        #             if ner_start == start and ner_end == end:
        #                 populated_ner_label = label
        #                 break
        #         if populated_ner_label != "EMPTY":
        #             break
        #     for start, end in cluster:
        #         expanded_ners[(start, end)] = populated_ner_label
        #         # expanded_ners.append([start, end, populated_ner_label])
        # # if self.doc_key == "bn/pri/00/pri_0053_0":
        # #     print(ners)
        # #     print(expanded_ners)
        # for start, end, label in ners:
        #     if (start, end) not in expanded_ners:
        #         if self.doc_key == "bn/pri/00/pri_0053_0":
        #             print(start, end, label)
        #         expanded_ners[(start, end)] = label
        #         # expanded_ners.append([start, end, label])
        # if self.doc_key == "bn/pri/00/pri_0053_0":
        #     print(expanded_ners)
        # expanded_ners = sorted((start, end, label) for (start, end), label in expanded_ners.items())
        # assert len(expanded_ners) <= len(util.flatten(merged_clusters)) + len(ners), (len(expanded_ners),
        #                                                                               len(util.flatten(
        #                                                                                   merged_clusters)) + len(ners))
        return {
            "doc_key": self.doc_key,
            "sentences": self.segments,
            "speakers": self.speakers,
            "pos": self.pos_tags,
            "constituents": [],
            "ner": ners,
            "srl": srl,
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(document_state, max_segment_len, constraints1, constraints2):
    current = 0
    previous_token = 0
    not_constraints1_satisfied = False
    while current < len(document_state.subtokens):
        end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
        while end >= current and not constraints1[end]:
            end -= 1
        if end < current:
            not_constraints1_satisfied = True
            end = min(current + max_segment_len - 1 - 2, len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")
        # document_state.segments.append(['[CLS]'] + document_state.subtokens[current:end + 1] + ['[SEP]'])
        document_state.segments.append(document_state.subtokens[current:end + 1])
        subtoken_map = document_state.subtoken_map[current: end + 1]
        # document_state.segment_subtoken_map.append([previous_token] + subtoken_map + [subtoken_map[-1]])
        document_state.segment_subtoken_map.append(subtoken_map)
        info = document_state.info[current: end + 1]
        # document_state.segment_info.append([None] + info + [None])
        document_state.segment_info.append(info)
        if hasattr(document_state, "speaker_mask"):
            document_state.segment_mask.append(document_state.speaker_mask[current: end + 1])
            document_state.segment_len.append(len([i for i in document_state.speaker_mask[current: end + 1] if i >= 0]))
        current = end + 1
        # previous_token = subtoken_map[-1]

    if not_constraints1_satisfied:
        print(document_state.subtokens)
        print(constraints1)


def get_sentence_map(segments, sentence_end):
    current = 0
    sent_map = []
    sent_end_idx = 0
    assert len(sentence_end) == sum([len(s) for s in segments])
    for segment in segments:
        # sent_map.append(current)
        for i in range(len(segment)):
            sent_map.append(current)
            current += int(sentence_end[sent_end_idx])
            sent_end_idx += 1
        # sent_map.append(current)
    return sent_map


def get_document(document_lines, tokenizer: BertTokenizer, language, segment_len):
    document_state = DocumentState(document_lines[0])
    word_idx = -1
    prev_speaker = None
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if len(document_state.sentence_end) == 0 or document_state.sentence_end[-1] is True:
            sentence_start = True
        else:
            sentence_start = False
        if not sentence_end:
            assert len(row) >= 12
            word_idx += 1
            word = normalize_word(row[3], language)
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if sidx != 0 else (row + [len(subtokens)] + [sentence_start])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)
                document_state.pos_tags.append(row[4])
        else:
            document_state.sentence_end[-1] = True
    # split_into_segments(document_state, segment_len, document_state.token_end)
    # split_into_segments(document_state, segment_len, document_state.sentence_end)
    constraints1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, segment_len, constraints1, document_state.token_end)
    stats["max_sent_len_{}".format(language)] = max(max([len(s) for s in document_state.segments]),
                                                    stats["max_sent_len_{}".format(language)])
    document = document_state.finalize()
    document_state.subtokens = []
    document_state.sentence_end = []
    document_state.token_end = []
    speaker_mask = []
    document_state.segment_mask = []
    document_state.segment_len = []
    document_state.segments = []
    for line in document_lines[1]:
        row = line.split()
        sentence_end = len(row) == 0
        if not sentence_end:
            assert len(row) >= 12
            speaker = row[9]
            if prev_speaker is None or prev_speaker != speaker:
                speaker_subtoken = [SPEAKER_START] + tokenizer.tokenize(speaker) + [SPEAKER_END]
                document_state.token_end += ([False] * (len(speaker_subtoken) - 1)) + [True]
                document_state.subtokens += speaker_subtoken
                prev_speaker = speaker
                speaker_mask += [-2] * len(speaker_subtoken)
                document_state.sentence_end += len(speaker_subtoken) * [False]
            word = normalize_word(row[3], language)
            subtokens = tokenizer.tokenize(word)
            document_state.token_end += ([False] * (len(subtokens) - 1)) + [True]
            document_state.subtokens += subtokens
            speaker_mask += [1] * len(subtokens)
            document_state.sentence_end += len(subtokens) * [False]
        else:
            document_state.sentence_end[-1] = True
    document_state.speaker_mask = speaker_mask
    split_into_segments(document_state, segment_len, document_state.sentence_end, document_state.token_end)

    assert (len([i for mask in document_state.segment_mask for i in mask if i > 0])) == len(document['sentence_map']), \
        (len([i for mask in document_state.segment_mask for i in mask if i > 0]), len(document['sentence_map']))
    assert (sum(document_state.segment_len) == len(document['sentence_map'])), (sum(document_state.segment_len),
                                                                                len(document['sentence_map']))
    document['sentences'] = document_state.segments
    document['segment_mask'] = document_state.segment_mask
    document['segment_len'] = document_state.segment_len
    return document


def skip(doc_key):
    # if doc_key in ['nw/xinhua/00/chtb_0078_0', 'wb/eng/00/eng_0004_1']: #, 'nw/xinhua/01/chtb_0194_0', 'nw/xinhua/01/chtb_0157_0']:
    # return True
    return False


def minimize_partition(name, language, extension, labels, stats, tokenizer, seg_len, input_dir, output_dir):
    input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    output_path = "{}/{}.{}.{}.jsonlines".format(output_dir, name, language, seg_len)
    count = 0
    print("Minimizing {}".format(input_path))
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)
    with open(output_path, "w") as output_file:
        for document_lines in documents:
            if skip(document_lines[0]):
                continue
            document = get_document(document_lines, tokenizer, language, seg_len)
            output_file.write(json.dumps(document))
            output_file.write("\n")
            count += 1
    print("Wrote {} documents to {}".format(count, output_path))


def minimize_language(language, labels, stats, seg_len, input_dir, output_dir):
    # do_lower_case = True if 'chinese' in vocab_file else False
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    minimize_partition("dev", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("train", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)
    minimize_partition("test", language, "v4_gold_conll", labels, stats, tokenizer, seg_len, input_dir, output_dir)


if __name__ == "__main__":
    # vocab_file = sys.argv[1]
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # do_lower_case = sys.argv[4].lower() == 'true'
    # print(do_lower_case)
    labels = collections.defaultdict(set)
    stats = collections.defaultdict(int)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for seg_len in [384, 512]:
        minimize_language("english", labels, stats, seg_len, input_dir, output_dir)
        # minimize_language('chinese', labels, stats, seg_len, input_dir, output_dir)
        # minimize_language("chinese", labels, stats, vocab_file, seg_len)
        # minimize_language("es", labels, stats, vocab_file, seg_len)
        # minimize_language("arabic", labels, stats, vocab_file, seg_len)
    for k, v in labels.items():
        print("{} = [{}]".format(k, ", ".join("\"{}\"".format(label) for label in v)))
    for k, v in stats.items():
        print("{} = {}".format(k, v))
