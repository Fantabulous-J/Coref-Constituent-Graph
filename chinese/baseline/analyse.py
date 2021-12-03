from collections import defaultdict

import util

singular_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                     'it', 'its', 'itself', 'yourself']
plural_pronouns = ['they', 'them', 'their', 'theirs', 'themselves', 'we', 'us', 'our', 'ours', 'ourselves',
                   'yourselves']
ambiguous_pronouns = ['you', 'your', 'yours']
valid_pronouns = singular_pronouns + plural_pronouns + ambiguous_pronouns


def get_gold_to_cluster_ids(examples):
    gold_to_cluster_id = []  # 0 means not in cluster
    non_anaphoric = []  # Firstly appeared mention in a cluster
    for i, example in enumerate(examples):
        gold_to_cluster_id.append(defaultdict(int))
        non_anaphoric.append(set())

        clusters = example['clusters']
        clusters = [sorted(cluster) for cluster in clusters]  # Sort mention
        for c_i, c in enumerate(clusters):
            non_anaphoric[i].add(tuple(c[0]))
            for m in c:
                gold_to_cluster_id[i][tuple(m)] = c_i + 1
    return gold_to_cluster_id, non_anaphoric


def analyse(examples, predicted_clusters, predicted_spans, predicted_antecedents):
    gold_to_cluster_id, non_anaphoric = get_gold_to_cluster_ids(examples)
    cluster_list = []
    subtoken_list = []
    num_gold_pronouns = 0
    for i, example in enumerate(examples):
        subtokens = util.flatten(example['sentences'])
        subtoken_list.append(subtokens)
        cluster_list.append([[' '.join(subtokens[m[0]: m[1] + 1]) for m in c] for c in predicted_clusters[i]])
        num_gold_pronouns += len(example['pronouns'])
    num_non_gold, num_gold_span, num_total_span = 0, 0, 0
    correct = 0
    # not a gold span but link to other spans or is a non-anaphoric gold span but link to other spans
    false_non_gold_span_link, false_non_anaphoric_link = 0, 0
    # gold anaphoric span but falsely start a new entity
    false_new = 0
    # gold anaphoric span but link to a wrong cluster
    wrong_link = 0
    for i, antecedents in enumerate(predicted_antecedents):
        antecedents = [(-1, -1) if a == -1 else predicted_spans[i][a] for a in antecedents]
        for j, antecedent in enumerate(antecedents):
            span = predicted_spans[i][j]
            span_cluster_id = gold_to_cluster_id[i][span]

            span_text = ' '.join(subtoken_list[i][span[0]: span[1] + 1]).lower()
            # antecedent_text = ' '.join(subtoken_list[i][antecedent[0]: antecedent[1] + 1]).lower()

            if span_text not in valid_pronouns:
                continue
            num_total_span += 1
            # not a gold span
            if span_cluster_id == 0:
                num_non_gold += 1
                if antecedent == (-1, -1):
                    # correct += 1
                    pass
                else:
                    false_non_gold_span_link += 1

            # a gold span but non anaphoric, should link to (-1, -1)
            elif span in non_anaphoric[i]:
                num_gold_span += 1
                if antecedent == (-1, -1):
                    # correct += 1
                    pass
                else:
                    false_non_anaphoric_link += 1

            # is a gold span and anaphoric
            else:
                num_gold_span += 1
                # falsely start a new entity
                if antecedent == (-1, -1):
                    false_new += 1
                # link to a wrong cluster
                elif span_cluster_id != gold_to_cluster_id[i][antecedent]:
                    wrong_link += 1
                else:
                    correct += 1

    return num_gold_pronouns, num_non_gold, num_gold_span, num_total_span, correct, false_non_gold_span_link, \
           false_non_anaphoric_link, false_new, wrong_link
