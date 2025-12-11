DATASET_PATH = "./data/dataset.json"
ID_DATASET_PATH = "./data/id_dataset.json"
DOCS_TOKENS_PATH = "./data/docs_tokens.json"
DATA_DIR = "./data/"
PLOTS_DIR = "./plots/"


def weighted_jaccard(doc1_tokens, doc2_tokens):
    min_sum = 0
    max_sum = 0
    for weight1, weight2 in zip(doc1_tokens, doc2_tokens):
        if weight1 <= 0 and weight2 <= 0:
            continue
        if weight1 >= weight2:
            min_sum += weight2
            max_sum += weight1
        else:
            min_sum += weight1
            max_sum += weight2
    if min_sum == 0 or max_sum == 0:
        return 0
    similarity = min_sum / max_sum
    return similarity