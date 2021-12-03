import json
import os
import sys

import pyhocon
import errno
import re
import numpy as np
import tensorflow as tf

import logging
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


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


def evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator,
                   top_span_mention_scores, singleton=False):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = get_predicted_clusters(top_span_starts, top_span_ends,
                                                                      predicted_antecedents,
                                                                      top_span_mention_scores,
                                                                      singleton=singleton)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters


def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents, top_span_mention_scores,
                           singleton=False):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            if singleton:
                if top_span_mention_scores[i] <= 0:
                    continue
                predicted_cluster = len(predicted_clusters)
                mention = (int(top_span_starts[i]), int(top_span_ends[i]))
                predicted_clusters.append([mention])
                mention_to_predicted[mention] = predicted_cluster
            continue

        assert i > predicted_index, (i, predicted_index)
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


def get_cons_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split("\t")
            tag2id[tag] = int(idx)
    return tag2id


def get_trace_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split("\t")
            tag2id[tag] = int(idx)
    return tag2id


def read_sdp_features(data_path):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    sdp_features = []

    for example in examples:
        sdp_features.append(example["semantic_dep_features"])

    return sdp_features


def read_depparse_features(data_path, num_orders=2):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    dependencies = []

    for example in tqdm(examples):
        depparse_features = example['depparse_features']
        dependency_features = []
        word_offset = 1
        for sent_feature in depparse_features:
            num_words = len(sent_feature)
            dep_edge = [[0] * num_words for _ in range(num_words)]
            temp_dependency = []
            for feature in sent_feature:
                # if feature['head_id'] == 0:
                #     continue
                word_id = feature['id']
                head_id = feature['head_id']
                deprel = feature['deprel']
                temp_dependency.append((deprel, head_id, word_id))
                if feature['head_id'] != 0:
                    dep_edge[word_id - word_offset][head_id - word_offset] = 1
                    dep_edge[head_id - word_offset][word_id - word_offset] = 1

            high_order_dependency = [temp_dependency]
            for i in range(1, num_orders):
                new_dependency = []
                for deprel, head_id, word_id in high_order_dependency[-1]:
                    if head_id == 0:
                        continue
                    if head_id - word_offset < 0 or head_id - word_offset >= len(temp_dependency):
                        print(head_id - word_offset, len(temp_dependency), word_id, head_id, word_offset)
                        print(temp_dependency)
                    head_node = temp_dependency[head_id - word_offset]
                    if head_node[1] == 0:
                        continue
                    new_dependency.append(("higher", head_node[1], word_id))
                    dep_edge[word_id - word_offset][head_node[1] - word_offset] = 1
                    dep_edge[head_node[1] - word_offset][word_id - word_offset] = 1
                high_order_dependency.append(new_dependency)
            word_offset += num_words
            dependency_features.append(high_order_dependency)

        dependencies.append(dependency_features)

    return dependencies


if __name__ == '__main__':
    read_depparse_features("conll_data/train.english.depparse.gold.jsonlines")