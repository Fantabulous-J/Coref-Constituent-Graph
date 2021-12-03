import json
import os
import sys

import pyhocon
import errno
import re
import numpy as np
from collections import defaultdict
import tensorflow as tf

import logging

import torch

logger = logging.getLogger(__name__)


srl_labels = ["R-ARGM-COM", "C-ARGM-NEG", "C-ARGM-TMP", "R-ARGM-DIR", "ARGM-LOC", "R-ARG2", "ARGM-GOL", "ARG5",
              "ARGM-EXT", "R-ARGM-ADV", "C-ARGM-MNR", "ARGA", "C-ARG4", "C-ARG2", "C-ARG3", "C-ARG0", "C-ARG1",
              "ARGM-ADV", "ARGM-NEG", "R-ARGM-MNR", "C-ARGM-EXT", "R-ARGM-PRP", "C-ARGM-ADV", "R-ARGM-MOD",
              "C-ARGM-ADJ", "ARGM-LVB", "R-ARGM-PRD", "ARGM-MNR", "ARGM-ADJ", "C-ARGM-CAU", "ARGM-CAU", "C-ARGM-MOD",
              "R-ARGM-EXT", "C-ARGM-COM", "ARGM-COM", "R-ARGM-GOL", "R-ARGM-TMP", "R-ARG4", "ARGM-MOD", "R-ARG1",
              "R-ARG0", "R-ARG3", "V", "ARGM-REC", "C-ARGM-DSP", "R-ARG5", "ARGM-DIS", "ARGM-DIR", "R-ARGM-LOC",
              "C-ARGM-DIS", "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARGM-TMP", "C-ARGM-DIR", "ARGM-PRD", "R-ARGM-PNC",
              "ARGM-PRX", "ARGM-PRR", "R-ARGM-CAU", "C-ARGM-LOC", "ARGM-PNC", "ARGM-PRP", "C-ARGM-PRP", "ARGM-DSP"]


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_speaker_dict(speakers, max_num_speakers):
    speaker_dict = {'UNK': 0, '[SPL]': 1}
    for s in speakers:
        if s not in speaker_dict and len(speaker_dict) < max_num_speakers:
            speaker_dict[s] = len(speaker_dict)
    return speaker_dict


def initialize_from_env(eval_test=False):
    if "GPU" in os.environ:
        set_gpus(int(os.environ["GPU"]))

    name = sys.argv[1]
    print("Running experiment: {}".format(name))

    if eval_test:
        config = pyhocon.ConfigFactory.parse_file("test.experiments.conf")[name]
    else:
        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], name + "_" + str(config['max_segment_len'])))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def set_gpus(*gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def load_from_pretrained_coref_tf_checkpoint(model, tf_checkpoint_path):
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load bert weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if "bert" not in name:
            continue
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
                for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model.bert
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # load task weights from TF model
    task_names = []
    task_arrays = []
    for name, shape in init_vars:
        if "bert" in name:
            continue
        # logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        task_names.append(name)
        task_arrays.append(array)

    for name, array in zip(task_names, task_arrays):
        if name == "antecedent_distance_emb":
            model.antecedent_distance_embeddings.weight.data = torch.from_numpy(array)
        elif name == "span_width_embeddings":
            model.span_width_embeddings.weight.data = torch.from_numpy(array)
        elif name == "coref_layer/antecedent_distance_emb":
            model.antecedent_distance_embeddings_coref_layer.weight.data = torch.from_numpy(array)
        elif name == "coref_layer/f/output_bias":
            model.refined_gate_projection.output_layer.bias.data = torch.from_numpy(array)
        elif name == "coref_layer/f/output_weights":
            model.refined_gate_projection.output_layer.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "coref_layer/same_speaker_emb":
            model.same_speaker_embeddings.weight.data = torch.from_numpy(array)
        elif name == "coref_layer/segment_distance/segment_distance_embeddings":
            model.segment_distance_embeddings.weight.data = torch.from_numpy(array)
        elif name == "coref_layer/slow_antecedent_scores/hidden_bias_0":
            model.slow_antecedent_scores_layer.hidden_layer_0.bias.data = torch.from_numpy(array)
        elif name == "coref_layer/slow_antecedent_scores/hidden_weights_0":
            model.slow_antecedent_scores_layer.hidden_layer_0.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "coref_layer/slow_antecedent_scores/output_bias":
            model.slow_antecedent_scores_layer.output_layer.bias.data = torch.from_numpy(array)
        elif name == "coref_layer/slow_antecedent_scores/output_weights":
            model.slow_antecedent_scores_layer.output_layer.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "genre_embeddings":
            model.genre_embeddings.weight.data = torch.from_numpy(array)
        elif name == "mention_scores/hidden_bias_0":
            model.mention_score_layer.hidden_layer_0.bias.data = torch.from_numpy(array)
        elif name == "mention_scores/hidden_weights_0":
            model.mention_score_layer.hidden_layer_0.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "mention_scores/output_bias":
            model.mention_score_layer.output_layer.bias.data = torch.from_numpy(array)
        elif name == "mention_scores/output_weights":
            model.mention_score_layer.output_layer.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "mention_word_attn/output_bias":
            model.mention_word_attn_layer.output_layer.bias.data = torch.from_numpy(array)
        elif name == "mention_word_attn/output_weights":
            model.mention_word_attn_layer.output_layer.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "output_bias":
            model.antecedent_distance_scores_projection.output_layer.bias.data = torch.from_numpy(array)
        elif name == "output_weights":
            model.antecedent_distance_scores_projection.output_layer.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "span_width_prior_embeddings":
            model.span_width_prior_embeddings.weight.data = torch.from_numpy(array)
        elif name == "src_projection/output_bias":
            model.src_projection.output_layer.bias.data = torch.from_numpy(array)
        elif name == "src_projection/output_weights":
            model.src_projection.output_layer.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "width_scores/hidden_bias_0":
            model.width_scores_layer.hidden_layer_0.bias.data = torch.from_numpy(array)
        elif name == "width_scores/hidden_weights_0":
            model.width_scores_layer.hidden_layer_0.weight.data = torch.from_numpy(np.transpose(array))
        elif name == "width_scores/output_bias":
            model.width_scores_layer.output_layer.bias.data = torch.from_numpy(np.transpose(array))
        elif name == "width_scores/output_weights":
            model.width_scores_layer.output_layer.weight.data = torch.from_numpy(np.transpose(array))

    return model


def get_predicted_antecedents(antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
        if index < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents


def evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = get_predicted_clusters(top_span_starts, top_span_ends,
                                                                      predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters


def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            continue
        assert i > predicted_index, (i, predicted_index)
        # if i <= predicted_index:
        #     continue
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster

        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

    return predicted_clusters, mention_to_predicted


def get_dep_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split(" ")
            tag2id[tag] = int(idx)
    return tag2id


def get_pos_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split("\t")
            tag2id[tag] = int(idx)
    return tag2id


def get_cons_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split("\t")
            tag2id[tag] = int(idx)
    return tag2id


def read_depparse_features(data_path):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    dependencies = []

    for example in examples:
        depparse_features = example['depparse_features']
        dependency_features = []
        for sent_feature in depparse_features:
            temp_dependency = []
            for feature in sent_feature:
                word_id = feature['id']
                head_id = feature['head_id']
                deprel = feature['deprel']
                temp_dependency.append([deprel, head_id, word_id])

            dependency_features.append(temp_dependency)
        dependencies.append(dependency_features)

    return dependencies


def read_dep_matrix(examples, data_path, dep_tag2id):
    # directed dependency parsing tree
    dependencies, _, _ = read_depparse_features(data_path)
    dep_matrics = []
    dep_rel_matrics = []
    for dependency in dependencies:
        dependency = flatten(dependency)
        dep_dict = {}
        dep_rel_dict = defaultdict(defaultdict)
        for rel, head, word in dependency:
            dep_dict[word] = [word]
            dep_rel_dict[word][word] = dep_tag2id.get("cyclic", 0)
            if head == 0:
                continue
            dep_dict[word].append(head)
            dep_rel_dict[word][head] = dep_tag2id.get(rel, 0)
        dep_matrics.append(dep_dict)
        dep_rel_matrics.append(dep_rel_dict)

    subtoken_matrics = []
    for i, example in enumerate(examples):
        matrix = dep_matrics[i]
        subtoken_map = torch.tensor(example['subtoken_map'], dtype=torch.int64)
        lengh_range = torch.arange(0, subtoken_map.size()[0], dtype=torch.int64)
        subtoken_matrix_dict = {}
        for word, heads in matrix.items():
            word_subtoken = lengh_range[subtoken_map == (word - 1)].numpy().tolist()
            heads_subtoken = []
            for head in heads:
                head_subtoken = lengh_range[subtoken_map == (head - 1)].numpy().tolist()
                if type(head_subtoken) == list:
                    heads_subtoken.extend(head_subtoken)
                else:
                    heads_subtoken.append(head_subtoken)
            if type(word_subtoken) == list:
                for subtoken in word_subtoken:
                    subtoken_matrix_dict[subtoken] = heads_subtoken
            else:
                subtoken_matrix_dict[word_subtoken] = heads_subtoken

        subtoken_matrics.append(subtoken_matrix_dict)
    matrics = []
    rel_matrics = []
    for i, subtoken_matrix_dict in enumerate(subtoken_matrics):
        subtoken_map = examples[i]['subtoken_map']
        subtoken_len = len(subtoken_map)
        doc_matrix = torch.zeros([subtoken_len, subtoken_len], dtype=torch.int64)
        doc_rel_matrix = torch.zeros([subtoken_len, subtoken_len], dtype=torch.int64)
        for subtoken, heads_subtoken in subtoken_matrix_dict.items():
            for head in heads_subtoken:
                doc_matrix[subtoken, head] = 1
                subtoken_word = subtoken_map[subtoken]
                head_word = subtoken_map[head]
                doc_rel_matrix[subtoken, head] = dep_rel_matrics[i][subtoken_word + 1][head_word + 1]
        matrics.append(doc_matrix)
        rel_matrics.append(doc_rel_matrix)
    return matrics, rel_matrics


def read_srl_features(srl_features, srl_tag2id):
    srl_dict = {}
    args = set()
    for predicate, arg_start, arg_end, label in srl_features:
        args.add((arg_start, arg_end))
        label = srl_tag2id.get(label, 0)
        try:
            srl_dict[predicate].append((arg_start, arg_end, label))
        except KeyError:
            srl_dict[predicate] = [(arg_start, arg_end, label)]

    return srl_dict


def split_srl_labels(include_c_v=False):
    adjunct_role_labels = []
    core_role_labels = []
    for label in srl_labels:
        if "AM" in label or "ARGM" in label:
            adjunct_role_labels.append(label)
        elif label != "V" and (include_c_v or label != "C-V"):
            core_role_labels.append(label)
    return adjunct_role_labels, core_role_labels

# if __name__ == '__main__':
#     tag_vocab = []
#     _, tag_vocab = read_srl_features("brain/kgs/srl/train.english.srl.jsonlines", tag_vocab)
#     _, tag_vocab = read_srl_features("brain/kgs/srl/dev.english.srl.jsonlines", tag_vocab)
#     _, tag_vocab = read_srl_features("brain/kgs/srl/test.english.srl.jsonlines", tag_vocab)
#
#     counter = Counter()
#     counter.update(tag_vocab)
#
#     tag_vocab = sorted(counter.items(), key=lambda tup: tup[0])
#     tag_vocab.sort(key=lambda tup: tup[1], reverse=True)
#     tag2id = {tag: i for i, (tag, _) in enumerate(tag_vocab)}
#     with open("brain/kgs/srl/srl_vocab.txt", "w") as f:
#         for tag, id in tag2id.items():
#             f.write(tag + " " + str(id) + "\n")
