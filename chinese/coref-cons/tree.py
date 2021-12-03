import json

from tqdm import tqdm


class Tree(object):
    def __init__(self, type):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.type = type
        self.is_leaf = False
        self.start = -1
        self.end = -1
        self.idx = -1

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()

        return count

    def __str__(self):
        return self.type

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def constituent_to_tree(constituent_string, word_offset, node_offset, num_orders=2):
    constituent_string = constituent_string.replace("\*", "*")
    constituents = []
    temp_str = ""
    for i, char in enumerate(constituent_string):
        if char == "(" or char == ")" or char == " ":
            if len(temp_str) != 0:
                constituents.append(temp_str)
                temp_str = ""
            if char != " ":
                constituents.append(char)
        else:
            temp_str += char
    stack = []
    for cons in constituents:
        if cons != ")":
            stack.append(cons)
        else:
            tail = stack.pop()
            temp_constituents = []
            while tail != "(":
                temp_constituents.append(tail)
                tail = stack.pop()

            parent = Tree(temp_constituents[-1])
            for i in range(len(temp_constituents) - 2, -1, -1):
                if isinstance(temp_constituents[i], Tree):
                    parent.add_child(temp_constituents[i])
                else:
                    child = Tree(temp_constituents[i])
                    parent.add_child(child)
            stack.append(parent)

    root = stack[-1]
    for node in root:
        if len(node.children) == 0:
            node.is_leaf = True

    for node in root:
        if node.is_leaf:
            node.start = word_offset
            node.end = word_offset
            word_offset += 1
    span_starts_ends(root)

    node_sequence = []
    internal_nodes = []
    for node in root:
        if not node.is_leaf:
            internal_nodes.append(node)
        node_sequence.append(node)

    node_offset_original = node_offset
    for node in root:
        if node.is_leaf:
            continue
        node.idx = node_offset
        node_offset += 1

    constituent_sequence = []
    num_internal_nodes = len(internal_nodes)
    constituent_edge = [[0] * num_internal_nodes for _ in range(num_internal_nodes)]
    for i, node in enumerate(internal_nodes):
        parent_idx = node.parent.idx if node.parent else -1
        constituent_sequence.append((node.idx, node.start, node.end, node.type, parent_idx))
        if parent_idx != -1:
            constituent_edge[node.idx - node_offset_original][parent_idx - node_offset_original] = 1
            constituent_edge[parent_idx - node_offset_original][node.idx - node_offset_original] = 1

    high_order_sequence = [constituent_sequence]
    for i in range(1, num_orders):
        new_constituent_sequence = []
        for idx, start, end, type, parent_idx in high_order_sequence[-1]:
            if parent_idx == -1:
                continue
            parent_node = constituent_sequence[parent_idx - node_offset_original]
            if parent_node[-1] == -1:
                continue
            new_constituent_sequence.append((idx, start, end, type, parent_node[-1]))
            constituent_edge[idx - node_offset_original][parent_node[-1] - node_offset_original] = 1
            constituent_edge[parent_node[-1] - node_offset_original][idx - node_offset_original] = 1
        high_order_sequence.append(new_constituent_sequence)

    return high_order_sequence, word_offset, node_offset


def read_constituents(data_path):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    constituents = []
    for example in tqdm(examples):
        constituent = []
        constituent_strings = example['constituents']
        word_offset, node_offset = 0, 0
        for string in constituent_strings:
            constituent_sequence, word_offset, node_offset = constituent_to_tree(string, word_offset, node_offset)
            constituent.append(constituent_sequence)
        constituents.append(constituent)
    return constituents


def span_starts_ends(node: Tree):
    if len(node.children) == 0:
        return
    for child in node.children:
        span_starts_ends(child)

    node.start = node.children[0].start
    node.end = node.children[-1].end


def build_constituent_tag():
    train_constituents = read_constituents(data_path="conll_data/train.chinese.constituency.bert.crf.jsonlines")
    dev_constituents = read_constituents(data_path="conll_data/dev.chinese.constituency.bert.crf.jsonlines")
    test_constituents = read_constituents(data_path="conll_data/test.chinese.constituency.bert.crf.jsonlines")
    constituents = train_constituents + dev_constituents + test_constituents

    labels = set()
    for constituent in constituents:
        for constituent_sequence in constituent:
            for seq in constituent_sequence:
                idx, start, end, label, parent_idx = seq
                labels.add(label)

    label2dict = {"<pad>": 0, "<unk>": 1, "cyclic": 2}
    for i, label in enumerate(labels):
        label2dict[label] = i + 3

    with open("conll_data/constituent_bert_crf_vocab.txt", "w") as f:
        for key, value in label2dict.items():
            f.write(key + "\t" + str(value) + "\n")


if __name__ == '__main__':
    build_constituent_tag()
