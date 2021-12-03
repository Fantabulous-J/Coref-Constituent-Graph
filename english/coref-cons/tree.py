import json

from nltk.tree import Tree as nltk_tree
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
    # print(constituent_string)
    constituents = []
    temp_str = ""
    for i, char in enumerate(constituent_string):
        # if (char == "(" and ((i == 0) or(i > 0 and constituent_string[i - 1] == " "))) \
        #         or (char == ")" and constituent_string[i - 1] != " ") or char == " ":
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

    word_offset_original = word_offset
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
            #  and node.type not in [":", "``", ".", ",", "XX", "X", "-LRB-", "-RRB-", "''", "HYPH"]
            internal_nodes.append(node)
        node_sequence.append(node)

    # # print("\n")
    # num_nodes = len(node_sequence)
    # head_ids = []
    # for head, internal_node in zip(heads, internal_nodes):
    #     node_type, node_word = head.split(" ")
    #     # print(node_type, node_word)
    #     for i, node in enumerate(node_sequence):
    #         if node.type == node_type and i < num_nodes - 1 and node_sequence[i + 1].type == node_word:
    #             head_ids.append(node_sequence[i].start)
    #             break
    # # print(head_ids)

    # if len(head_ids) != len(internal_nodes):
    #     print(constituent_string)
    #     print(heads)
    #     for node in root:
    #         print(node)
    #     for node, head in head_ids:
    #         print(node, head)
    # assert len(head_ids) == len(internal_nodes), (len(head_ids), len(internal_nodes))

    node_offset_original = node_offset
    for node in root:
        if node.is_leaf:
            #  or node.type in [":", "``", ".", ",", "XX", "X", "-LRB-", "-RRB-", "''", "HYPH"]
            continue
        node.idx = node_offset
        node_offset += 1

    # words = [node.type for node in root if node.is_leaf]
    constituent_sequence = []
    num_internal_nodes = len(internal_nodes)
    constituent_edge = [[0] * num_internal_nodes for _ in range(num_internal_nodes)]
    for i, node in enumerate(internal_nodes):
        # if node.type in [":", "``", ".", ",", "XX", "X", "-LRB-", "-RRB-", "''", "HYPH"]:
        #     continue
        parent_idx = node.parent.idx if node.parent else -1
        constituent_sequence.append((node.idx, node.start, node.end, node.type, parent_idx))
        if parent_idx != -1:
            constituent_edge[node.idx - node_offset_original][parent_idx - node_offset_original] = 1
            constituent_edge[parent_idx - node_offset_original][node.idx - node_offset_original] = 1
    # inter_sent_constituent_edges = []
    # for idx, start, end, type, _ in constituent_sequence:
    #     if type not in ["NP", "NML", "PRP", "PRP$", "WP", "WDT", "WRB", "NNP", "VB", "VBD", "VBN", "VBG", "VBZ", "VBP"]:
    #         continue
    #     tokens = words[start - word_offset_original: end - word_offset_original + 1]
    #     for idx1, start1, end1, type1, _, tokens1 in constituents_total:
    #         if tokens == tokens1 and type == type1:
    #             inter_sent_constituent_edges.append((idx, idx1))
    #         elif tokens1 in tokens or tokens in tokens1:
    #             inter_sent_constituent_edges.append((idx, idx1))
    #         elif len(set(tokens) & set(tokens1)) > 0.5 * min(len(tokens), len(tokens1)) and type == type1:
    #             inter_sent_constituent_edges.append((idx, idx1))
    # constituents_total += [(idx, start, end, type, parent_idx, words[start: end + 1])
    #                        for idx, start, end, type, parent_idx in constituent_sequence]
    # constituent_sequence = (constituent_sequence, inter_sent_constituent_edges)
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

    # head_sequence = []
    # for idx, start, end, type, parent_idx in constituent_sequence:
    #     for idx1, start1, end1, type1, parent_idx1 in constituent_sequence:
    #         if idx == idx1:
    #             continue
    #         head_id = head_ids[idx - node_offset_original] - word_offset_original
    #         head_id1 = head_ids[idx1 - node_offset_original] - word_offset_original
    #         if sent_dep[head_id][head_id1] == 1 and sent_dep[head_id1][head_id] == 1:
    #             nodeidx = idx - node_offset_original
    #             nodeidx1 = idx1 - node_offset_original
    #             if constituent_edge[nodeidx][nodeidx1] == 0 and constituent_edge[nodeidx1][nodeidx] == 0:
    #                 head_sequence.append((idx, start, end, type, idx1))
    #                 constituent_edge[nodeidx][nodeidx1] = 1
    #                 constituent_edge[nodeidx1][nodeidx] = 1
    # high_order_sequence.append(head_sequence)

    return high_order_sequence, word_offset, node_offset


def read_constituents(data_path):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    # dep_examples = util.read_depparse_features(dep_data_path)
    constituents = []
    for example in tqdm(examples):
        constituent = []
        # constituents_total = []
        constituent_strings = example['constituents']
        # constituent_heads = example['heads']
        word_offset, node_offset = 0, 0
        for string in constituent_strings:
            constituent_sequence, word_offset, node_offset = constituent_to_tree(string, word_offset, node_offset)
            constituent.append(constituent_sequence)
        constituents.append(constituent)
    return constituents


def extract_subtrees(data_path, output_path):
    with open(data_path) as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    with open(output_path, 'w') as f:
        for example in examples:
            doc_key = example['doc_key']
            constituent_strings = example['constituents']
            tree_example = []
            for string in constituent_strings:
                tree = nltk_tree.fromstring(string)
                subtrees = []
                for i in tree.subtrees():
                    subtree_str = i.__str__()
                    subtrees.append(subtree_str.replace("\n", "").replace("\r", ""))
                tree_example.append(subtrees)
            f.write(json.dumps({
                "doc_key": doc_key,
                "subtrees": tree_example
            }))
            f.write('\n')


def span_starts_ends(node: Tree):
    if len(node.children) == 0:
        return
    for child in node.children:
        span_starts_ends(child)

    node.start = node.children[0].start
    node.end = node.children[-1].end


def build_constituent_tag():
    train_constituents = read_constituents(data_path="conll_data/train.english.constituency.bert.crf.jsonlines")
    dev_constituents = read_constituents(data_path="conll_data/dev.english.constituency.bert.crf.jsonlines")
    test_constituents = read_constituents(data_path="conll_data/test.english.constituency.bert.crf.jsonlines")
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


def pformat(node, margin=70, indent=0, nodesep='', parens='()', quotes=False):
    s = '%s%s%s' % (parens[0], node.type, nodesep)
    for child in node.children:
        if len(child.children) != 0:
            s += (
                    '\n'
                    + ' ' * (indent + 2)
                    + pformat(child, margin, indent + 2, nodesep, parens, quotes)
            )
        else:
            s += ' ' + '%s' % child.type
    return s + parens[1]


def build_tree(constituent_string):
    constituents = []
    temp_str = ""
    for char in constituent_string:
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
        if node.type == "EDITED":
            node.parent.children.remove(node)
    return root


if __name__ == '__main__':
    # constituent_string = "(S (RB So) (NP (NP (DT the)) (IN if)) (HYPH -) (ADVP (RB then)) (NP (NN statement)) " \
    #                      "(VP (VBZ goes) (NP (NP (NP (NN something)) (PP (IN like) (NP (DT this)))) (: :) (NP (`` \") " \
    #                      "(IN if) (NP (JJ total) (JJ true) (NNP Kerry) (: -RRB-) (JJ total) (JJ true) (NNP Bush) (, ,) " \
    #                      "(NNP Bush) (SYM x) (CD 1.04x) (CD -LRB-) (NFP .04)) (VP (VP (VBZ is) (NP (DT a) (JJ random) " \
    #                      "(NN number) (: -RRB-) (NP (NFP -LRB-) (JJ total) (JJ true) (NNP Kerry) (: -RRB-)) (, ,) " \
    #                      "(JJ total) (JJ true) (NNP Bush))) ('' \"))))) (. .))"
    # constituent_string = "(TOP(S(PP(IN In)(NP(NP(DT the)(NN summer))(PP(IN of)(NP(CD 2005)))))(, ,)(NP(NP(DT a)" \
    #                      "(NN picture))(SBAR(WHNP(WDT that))(S(NP(NNS people))(VP(VBP have)(ADVP(RB long))(VP(VBN been)" \
    #                      "(VP(VBG looking)(ADVP(RB forward)(PP(IN to)))))))))(VP(VBD started)(S(VP(VBG emerging)" \
    #                      "(PP(IN with)(NP(NN frequency)))(PP(IN in)(NP(JJ various)(JJ major)(NML(NNP Hong)(NNP Kong))" \
    #                      "(NNS media))))))(. .)))"
    constituent_string = "(TOP(S(PP(IN In)(NP(NP(DT the)(NN summer))(PP(IN of)(NP(CD 2005)))))(, ,)(NP(NP(DT a)" \
                         "(NN picture))(SBAR(WHNP(WDT that))(S(NP(NNS people))(VP(VBP have)(ADVP(RB long))(VP (VBN been) " \
                         "(VP(VBG looking)(RB forward)(PP(IN to))))))))(VP(VBD started)(S(VP(VBG emerging)(PP(IN with)" \
                         "(NP(NN frequency)))(PP(IN in)(NP(JJ various)(JJ major)(NNP Hong)(NNP Kong)(NNS media))))))(. .)))"
    # constituent_heads = ["VBD started", "VBD started", "NN summer", "IN In", "NN summer", "NN summer", "DT the",
    #                      "NN summer", "CD 2005", "IN of", "CD 2005", "CD 2005", ", ,", "NN picture", "NN picture",
    #                      "DT a", "NN picture", "VBG looking", "WDT that", "WDT that", "VBG looking", "NNS people",
    #                      "NNS people", "VBG looking", "VBP have", "RB long", "RB long", "VBG looking", "VBN been",
    #                      "VBG looking", "VBG looking", "RB forward", "RB forward", "IN to", "IN to", "VBD started",
    #                      "VBD started", "VBG emerging", "VBG emerging", "VBG emerging", "NN frequency", "IN with",
    #                      "NN frequency", "NN frequency", "NNS media", "IN in", "NNS media", "JJ various", "JJ major",
    #                      "NNP Kong", "NNP Hong", "NNP Kong", "NNS media", ". ."]
    # constituent_sequence, word_offset, node_offset = constituent_to_tree(constituent_string, 0, 0, [])
    # for constituent in constituent_sequence:
    #     print(constituent)

    # train_constituents = read_constituents(data_path="conll_data/train.english.constituency.jsonlines")
    dev_constituents = read_constituents(data_path="conll_data/dev.english.constituency.head.jsonlines")
    for constituent in dev_constituents:
        for high_order_sent_cons in constituent:
            for sent_cons in high_order_sent_cons:
                print(sent_cons)
            exit()
    # test_constituents = read_constituents(data_path="conll_data/test.english.constituency.bert.crf.jsonlines")

    # t = nltk_tree.fromstring(constituent_string)

    # root = build_tree(constituent_string)
    # print(pformat(root))

    # for node in root:
    #     if node.is_leaf:
    #         tree_string += " " + node.type + ")"
    #     else:
    #         tree_string += "(" + node.type
    # print(tree_string)
    # print(constituent_to_tree(constituent_string, 0, 0))

    # build_constituent_tag()

    # input_path = "conll_data/test.english.constituency.jsonlines"
    # output_path = "conll_data/test.english.constituent.subtrees.jsonlines"
    # extract_subtrees(data_path=input_path, output_path=output_path)

    # input_path = "conll_data/test.english.constituent.subtrees.jsonlines"
    # with open(input_path) as f:
    #     examples = [json.loads(jsonline) for jsonline in f.readlines()]
    # output_path = "conll_data/test.english.constituent.heads.txt"
    # if path.isfile(output_path):
    #     remove(output_path)
    #     f = open(output_path, "w", encoding='utf-8')
    #     f.close()
    # for example in tqdm(examples):
    #     doc_key = example['doc_key']
    #     system('echo "{}" >> "{}"'.format(doc_key, output_path))
    #     subtrees = example['subtrees']
    #     subtree_strings = []
    #     for sent_subtrees in subtrees:
    #         subtree = "CLS".join(sent_subtrees)
    #         subtree_strings.append(subtree)
    #
    #     file = tempfile.NamedTemporaryFile()
    #     tmp = file.name
    #     # tmp = "C:\\Users\\jiang\\Dropbox\\Research Project\\codes\\coref-HGAT\\conll_data\\temp.txt"
    #     with open(tmp, 'w') as out:
    #         for subtree in subtree_strings:
    #             out.write(subtree + '\n')
    #
    #     command = "java -cp \"stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar\":\".\" " \
    #               "TreeFunctions \"{}\" >> \"{}\"".format(tmp, output_path)
    #     system(command)
    #
    # name = "test"
    # input_constituent_path = "conll_data/{}.english.constituency.jsonlines".format(name)
    # input_heads_path = "conll_data/{}.english.constituent.heads.txt".format(name)
    #
    # with open(input_constituent_path) as f:
    #     examples = [json.loads(jsonline) for jsonline in f.readlines()]
    #
    # sentence_head = []
    # document_heads = []
    # heads = []
    # doc_keys = []
    # with open(input_heads_path) as input_file:
    #     for i, line in enumerate(input_file.readlines()):
    #         line = line.strip()
    #         if len(line) == 0:
    #             heads.append(sentence_head)
    #             sentence_head = []
    #         elif not line[0] == "(":
    #             if i != 0:
    #                 document_heads.append(heads)
    #             heads = []
    #             doc_keys.append(line)
    #         else:
    #             sentence_head.append(line[1:-1])
    # document_heads.append(heads)
    #
    # for i, example in enumerate(examples):
    #     assert example['doc_key'] == doc_keys[i], (example['doc_key'], doc_keys[i])
    #     example['heads'] = document_heads[i]
    #
    # output_path = "conll_data/{}.english.constituency.head.jsonlines".format(name)
    # with open(output_path, "w") as output_file:
    #     for example in examples:
    #         output_file.write(json.dumps(example))
    #         output_file.write("\n")

    # # remove EDITED in PTB of Ontonotes
    # input_dir = "conll_data"
    # language = "english"
    # extension = "v4_gold_conll"
    # output_dir = "parse/"
    #
    # for name in ['train', 'dev', 'test']:
    #     input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    #     documents = []
    #     with open(input_path, "r") as input_file:
    #         for line in input_file.readlines():
    #             begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
    #             if begin_document_match:
    #                 doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
    #                 documents.append((doc_key, []))
    #             elif line.startswith("#end document"):
    #                 continue
    #             else:
    #                 documents[-1][1].append(line)
    #     doc_keys = {}
    #     for document in documents:
    #         doc_key = document[0].split("_")
    #         filename = "_".join(doc_key[:2])
    #         part = doc_key[-1]
    #         if filename not in doc_keys:
    #             doc_keys[filename] = ["0" * (3 - len(part)) + part]
    #         else:
    #             doc_keys[filename].append("0" * (3 - len(part)) + part)
    #
    #         parse_file = os.path.join(output_dir + name, filename + ".parse")
    #         parse_edited_file = os.path.join(output_dir + name, filename + ".edit.parse")
    #         # directory, file = os.path.split(output_path)
    #         parse_trees = []
    #         with open(parse_file) as input_file:
    #             tree_string = ""
    #             for line in input_file.readlines():
    #                 line = line.strip()
    #                 if len(line) != 0:
    #                     tree_string += line
    #                 else:
    #                     parse_trees.append(tree_string)
    #                     tree_string = ""
    #
    #         with open(parse_edited_file, "w") as output_file:
    #             for tree in parse_trees:
    #                 root = build_tree(tree)
    #                 output_file.write(pformat(root))
    #                 output_file.write("\n\n")
