import os
import re

import torch
import tensorflow as tf
import numpy as np

import util
from model import CorefModel

import logging

logger = logging.getLogger(__name__)


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
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
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


def convert_tf_checkpoint_to_pytorch(model, tf_checkpoint_path, pytorch_dump_path):
    load_from_pretrained_coref_tf_checkpoint(model, tf_checkpoint_path)
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == '__main__':
    config = util.initialize_from_env()
    model = CorefModel(config)
    pytorch_dump_path = os.path.join(config['data_dir'], "pytorch_best_model.bin")
    convert_tf_checkpoint_to_pytorch(model, config['tf_checkpoint'], pytorch_dump_path)