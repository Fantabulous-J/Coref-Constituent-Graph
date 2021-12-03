import json
import re

import conll

input_dir = "conll_data"
language = "chinese"
extension = "v4_gold_conll"
output_dir = "conll_data"

count = 0


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


for name in ["train", "dev", "test"]:
    input_path = "{}/{}.{}.{}".format(input_dir, name, language, extension)
    output_path = "{}/{}.{}.constituency.jsonlines".format(output_dir, name, language)
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

    documents_constituent = {}
    with open(output_path, "w") as output_file:
        for document_lines in documents:
            document = []
            doc_key = document_lines[0]
            constituent_string = ""
            for line in document_lines[1]:
                row = line.split()
                sentence_end = len(row) == 0
                if not sentence_end:
                    assert len(row) >= 12
                    word = normalize_word(row[3], language)
                    pos = row[4]
                    constituent = row[5]
                    idx = constituent.find("*")
                    constituent = constituent[:idx] + "({} {})".format(pos, word) + constituent[idx + 1:]
                    constituent_string += constituent
                else:
                    document.append(constituent_string)
                    constituent_string = ""
            documents_constituent[doc_key] = document
            output_file.write(json.dumps({
                "doc_key": doc_key,
                "constituents": document
            }))
            output_file.write("\n")
