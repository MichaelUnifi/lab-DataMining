import json
from index import LSHIndex
from forest import LSHForest
import random
import numpy as np
from utilities import DATASET_PATH, ID_DATASET_PATH, DOCS_TOKENS_PATH, weighted_jaccard
import math
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

def query_lsh_forest(forest, c, num_queries, results_per_query, sync = False):
    queries = []
    queries_names = list(dataset.keys())
    random.shuffle(queries_names)
    results = []
    queries = queries_names[:num_queries]
    successful_queries = float(num_queries)
    avg_results_number = 0.0
    for i, query in enumerate(queries):
        queried_doc = docs_tokens[query]
        candidates = []
        if sync:
            candidates = forest.candidate_search_sync(queried_doc, c, results_per_query)
        else:
            candidates = forest.candidate_search(queried_doc, c, results_per_query)
        candidates = list(set(candidates)) #duplicate removal

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

def benchmark_lsh_index_on_m(m_range, k, l, num_rep, num_queries):
    
    means = [0] * len(m_range)
    for _ in tqdm(range(num_rep), desc="benchmark lsh on m"):
        index = LSHIndex(l, k)
        for doc_id in id_dataset.keys():
            index.insert(docs_tokens[doc_id], doc_id)
        rep_mean = 0
        for i, m in enumerate(m_range):
            candidate_size = 2 * m
            c = math.ceil(candidate_size / l)
            
            rep_mean, results_number = query_lsh_index(index, c, num_queries, m, True)
            means[i] += rep_mean / num_rep
    return means

def benchmark_lsh_index_on_M(candidate_size_range, k, l, num_rep, num_queries, m = 5):
    means = [0] * len(candidate_size_range)
    for _ in tqdm(range(num_rep), desc="benchmark lsh on M"):
        index = LSHIndex(l, k)
        for doc_id in id_dataset.keys():
            index.insert(docs_tokens[doc_id], doc_id)
        for i, candidate_size in enumerate(candidate_size_range):
            candidate_size = 2 * m
            c = math.ceil(candidate_size / l)
            
            rep_mean, results_number = query_lsh_index(index, c, num_queries, m, True)
            means[i] += rep_mean / num_rep
    return means

def benchmark_random_on_m(m_range, num_rep, num_queries, l = 5):
    means = []
    for m in m_range:
        results = []
        for _ in range(num_rep * num_queries):
            queries_names = list(dataset.keys())
            random.shuffle(queries_names)
            
            query = queries_names[0]
            candidates = random.sample(queries_names, 2 * m)

            scored_candidates = []
            for cand_id in candidates:
                score = weighted_jaccard(id_dataset[query].values(), id_dataset[cand_id].values())
                scored_candidates.append((cand_id, score))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            results.append([score[1] for score in scored_candidates[:m]])
        
        mean = 0
        for result in results:
            mean += np.mean(result)
        means.append(mean / (num_rep * num_queries))
    return means

def benchmark_random_on_M(candidate_size_range, num_rep, num_queries, m = 5):
    means = []
    for candidate_size in candidate_size_range:
        results = []
        for _ in range(num_rep * num_queries):
            queries_names = list(dataset.keys())
            random.shuffle(queries_names)
            
            query = queries_names[0]
            candidates = random.sample(queries_names, candidate_size)

            scored_candidates = []
            for cand_id in candidates:
                score = weighted_jaccard(id_dataset[query].values(), id_dataset[cand_id].values())
                scored_candidates.append((cand_id, score))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            results.append([score[1] for score in scored_candidates[:m]])
        
        mean = 0
        for result in results:
            mean += np.mean(result)
        means.append(mean / (num_rep * num_queries))
    return means

def benchmark_lsh_forest_on_m(m_range, k, l, num_rep, num_queries, sync = False):
    
    means = [0] * len(m_range)
    for _ in tqdm(range(num_rep), desc="benchmark forest on m"):
        forest = LSHForest(l, k)
        for doc_id in id_dataset.keys():
            forest.insert(docs_tokens[doc_id], doc_id)
        for i, m in enumerate(m_range):
            candidate_size = 2 * m
            c = math.ceil(candidate_size / l)    
            rep_mean, results_number = query_lsh_forest(forest, c, num_queries, m, sync)
            means[i] += rep_mean / num_rep
    return means

def benchmark_lsh_forest_on_M(candidate_size_range, k, l, num_rep, num_queries, m = 5, sync = False):
    
    means = [0] * len(candidate_size_range)
    for _ in tqdm(range(num_rep), desc="benchmark forest on M"):
        forest = LSHForest(l, k)
        for doc_id in id_dataset.keys():
            forest.insert(docs_tokens[doc_id], doc_id)
        for i, candidate_size in enumerate(candidate_size_range):
            candidate_size = 2 * m
            c = math.ceil(candidate_size / l)
            
            rep_mean, results_number = query_lsh_forest(forest, c, num_queries, m, sync)
            means[i] += rep_mean / num_rep

    return means

num_rep = 100
m_range = [2**i for i in range(1,8)]
M_range = range(5, 50, 5)
k_lsh = 11
k_tree = 26
l = 5
num_queries = 1


desired_size_dict = {}
candidate_size_dict = {}
print("desired results size benchmark")
desired_size_dict["lsh"] = benchmark_lsh_index_on_m(m_range, k_lsh, l, num_rep, num_queries)
desired_size_dict["random"] = benchmark_random_on_m(m_range, num_rep, num_queries, l = 5)
desired_size_dict["forestsync"] = benchmark_lsh_forest_on_m(m_range, k_tree, l, num_rep, num_queries, sync = True)
desired_size_dict["forest"] = benchmark_lsh_forest_on_m(m_range, k_tree, l, num_rep, num_queries, sync = False)
desired_size_dict["indexes"] = list(m_range)
print("candidate size benchmark")
candidate_size_dict["lsh"] = benchmark_lsh_index_on_M(M_range, k_lsh, l, num_rep, num_queries)
candidate_size_dict["random"] = benchmark_random_on_M(M_range, num_rep, num_queries, m = 5)
candidate_size_dict["forestsync"] = benchmark_lsh_forest_on_M(M_range, k_tree, l, num_rep, num_queries, m = 5, sync = True)
candidate_size_dict["forest"] = benchmark_lsh_forest_on_M(M_range, k_tree, l, num_rep, num_queries, m = 5, sync = False)
candidate_size_dict["indexes"] = list(M_range)


with open("./data/desired_size.json", 'w') as fp:
    json.dump(desired_size_dict, fp)
with open("./data/candidate_size.json", 'w') as fp:
    json.dump(candidate_size_dict, fp)

def find_level_stats(k, num_rep):
    total_nodes_per_depths = {}
    total_points_per_depths = {}
    for i in range(k):
        total_nodes_per_depths[i] = 0
        total_points_per_depths[i] = 0
    for _ in range(num_rep):

        forest = LSHForest(num_trees=1, num_hashes=k)
        for doc_id in id_dataset.keys():
            forest.insert(docs_tokens[doc_id], doc_id)
        points_stats, nodes_stats = forest.aggregate_level_counts()
        for i in range(k):
            total_nodes_per_depths[i] += nodes_stats[i]
            total_points_per_depths[i] += points_stats[i]
    return total_points_per_depths, total_nodes_per_depths


num_rep = 10
levels = list(range(k_tree))

total_points_per_depths, total_nodes_per_depths = find_level_stats(k_tree, num_rep)
print(total_nodes_per_depths)
print(total_points_per_depths)
points_stats = []
for i in range(k_tree):
    points_stats.append(total_points_per_depths[i]/total_nodes_per_depths[i])

tree_nodes_dict = {}
tree_nodes_dict["nodes"] = points_stats
tree_nodes_dict["levels"] = levels
with open("./data/tree_level_stats.json", 'w') as fp:
    json.dump(tree_nodes_dict, fp)