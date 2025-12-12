from index import LSHIndex
from forest import LSHForest
import random
import numpy as np
from utilities import DATASET_PATH, ID_DATASET_PATH, DOCS_TOKENS_PATH, weighted_jaccard
import json
from tqdm import tqdm


with open(DATASET_PATH, 'r') as fp:
    dataset = json.load(fp)
with open(ID_DATASET_PATH, 'r') as fp:
    id_dataset = json.load(fp)
with open(DOCS_TOKENS_PATH, 'r') as fp:
    docs_tokens = json.load(fp)


def query_lsh_index(index, c, num_queries, results_per_query, fill = False):
    queries = []
    queries_names = list(dataset.keys())
    random.shuffle(queries_names)
    results = []
    queries = queries_names[:num_queries]
    successful_queries = float(num_queries)
    avg_results_number = 0.0
    for i, query in enumerate(queries):
        queried_doc = docs_tokens[query]
        candidates = index.candidate_search(queried_doc, c)
        candidates = list(set(candidates)) #duplicate removal
        if fill and len(candidates) < results_per_query:
            candidates.extend(random.sample(list(id_dataset.keys()), k=int(results_per_query - len(candidates))))
        if queries[i] in candidates:
            candidates.remove(queries[i]) #removing original query
        if(not candidates and not fill):
            successful_queries -= 1
            continue
        scored_candidates = []
        for cand_id in candidates:
            score = weighted_jaccard(id_dataset[query].values(), id_dataset[cand_id].values())
            scored_candidates.append((cand_id, score))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        avg_results_number += len(scored_candidates[:results_per_query])
        results.append([score[1] for score in scored_candidates[:results_per_query]])
    mean = 0
    for result in results:
        if result:
            mean += np.mean(result)
    if successful_queries == 0:
        return -1, avg_results_number
    return mean / successful_queries, avg_results_number / num_queries

def tune_lsh_index(max_k, c, l, num_rep, num_queries, results_per_query, fill = False):
    nofill_similarity_results = []
    nofill_avg_results_numbers = []
    fill_similarity_results = []

    for k in tqdm(range(1, max_k)):
        print(f"k: {k}")
        nofill_avg_results_number = 0.0
        nofill_successful_reps = float(num_rep)
        fill_successful_reps = float(num_rep)
        nofill_mean = 0.0
        fill_mean = 0.0
        for _ in range(num_rep):
            index = LSHIndex(l, k)
            for doc_id in id_dataset.keys():
                index.insert(docs_tokens[doc_id], doc_id)
            nofill_rep_mean, nofill_results_number = query_lsh_index(index, c, num_queries, results_per_query)
            if nofill_rep_mean == -1:
                nofill_successful_reps -= 1
            else:
                nofill_mean += nofill_rep_mean
            nofill_avg_results_number += nofill_results_number
            fill_rep_mean, fill_results_number = query_lsh_index(index, c, num_queries, results_per_query, True)
            if fill_rep_mean == -1:
                fill_successful_reps -= 1
            else:
                fill_mean += fill_rep_mean

        nofill_mean /= nofill_successful_reps
        fill_mean /= fill_successful_reps
        nofill_similarity_results.append(nofill_mean)
        nofill_avg_results_numbers.append(nofill_avg_results_number / num_rep)
        fill_similarity_results.append(fill_mean)

    return nofill_similarity_results, nofill_avg_results_numbers, fill_similarity_results

def test_random_search(label_lengths, results_per_query, c, l, num_rep, num_queries):
    means = []
    for length in label_lengths:
        results = []
        for _ in range(num_rep * num_queries):
            queries_names = list(dataset.keys())
            random.shuffle(queries_names)
            
            query = queries_names[0]
            candidates = random.sample(queries_names, c * l)

            scored_candidates = []
            for cand_id in candidates:
                score = weighted_jaccard(id_dataset[query].values(), id_dataset[cand_id].values())
                scored_candidates.append((cand_id, score))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            results.append([score[1] for score in scored_candidates[:results_per_query]])
        
        mean = 0
        for result in results:
            mean += np.mean(result)
        means.append(mean / (num_rep * num_queries))
    return means

c = 5
l = 5
num_rep = 100
num_queries = 5
results_per_query = 5 #top m queries
max_k = 26

print("lsh no fill and fill")
similarities_nofill, num_results_nofill, similarities_fill = tune_lsh_index(max_k, c, l, num_rep, num_queries, results_per_query)
print(similarities_nofill, num_results_nofill)
print(similarities_fill)

print("random")
similarities_random = test_random_search(range(1, max_k), results_per_query, c, l, num_rep, num_queries)
print(similarities_random)

lsh_benchmark_dict = {}
lsh_benchmark_dict["lsfnofill"] = similarities_nofill
lsh_benchmark_dict["lsffill"] = similarities_fill
lsh_benchmark_dict["random"] = similarities_random
lsh_benchmark_dict["lenghts"] = list(range(1, max_k))
num_results_dict = {}
num_results_dict["results"] = num_results_nofill
num_results_dict["lengths"] = list(range(1, max_k))
with open("./data/tune_lsh.json", 'w') as fp:
    json.dump(lsh_benchmark_dict, fp)
with open("./data/results_sizes.json", 'w') as fp:
    json.dump(num_results_dict, fp)
