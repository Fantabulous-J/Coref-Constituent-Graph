import json
import os
import sys

import pyhocon
import errno
import re
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import tensorflow as tf

import logging

import torch

logger = logging.getLogger(__name__)


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


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


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
        # mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        # has_correct_antecedent = False
        # if mention in gold_mention_map:
        #     for j in range(0, antecedent_scores.shape[1] - 1):
        #         antecedent = antecedents[i, j]
        #         if antecedent >= i:
        #             continue
        #         antecedent_mention = (int(top_span_starts[antecedent]), int(top_span_ends[antecedent]))
        #         if antecedent_mention in gold_mention_map:
        #             if gold_cluster_ids[gold_mention_map[mention]] == gold_cluster_ids[gold_mention_map[antecedent_mention]]:
        #                 predicted_antecedents.append(antecedent)
        #                 has_correct_antecedent = True
        #                 break
        # if not has_correct_antecedent:
        if index < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents


def evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters,
                   evaluator, top_span_mention_scores, singleton=False):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = get_predicted_clusters(top_span_starts, top_span_ends,
                                                                      predicted_antecedents, top_span_mention_scores,
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


def read_dist_mat(data_path):
    dependencies = read_depparse_features(data_path)
    dist_mats = []
    for dependency in dependencies:
        word_offset = 1
        document_dist_mat = []
        for sent_dependency in dependency:
            sent_len = len(sent_dependency)
            adj_mat = np.zeros((sent_len, sent_len), dtype=np.float32)
            idx = []
            for deprel, head, modifier in sent_dependency:
                if head == 0:
                    continue
                adj_mat[modifier - word_offset, head - word_offset] = 1
                idx.append(modifier - word_offset)

            # undirected
            adj_mat = adj_mat + adj_mat.T

            # self-loop
            for i in idx:
                adj_mat[i, i] = 1

            dist_mat = shortest_path(csgraph=csr_matrix(adj_mat), directed=False)
            np.fill_diagonal(dist_mat, 1)
            document_dist_mat.append((word_offset - 1, dist_mat))
            word_offset += sent_len
        dist_mats.append(document_dist_mat)

    return dist_mats
