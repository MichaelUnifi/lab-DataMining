import matplotlib.pyplot as plt
import json
from utilities import PLOTS_DIR, DATA_DIR

with open(DATA_DIR + str("tune_lsh.json"), 'r') as fp:
    tune_lsh_results = json.load(fp)

x = tune_lsh_results["lenghts"]

plt.figure()
plt.clf()
plt.plot(x, tune_lsh_results["lsfnofill"],   label='LSH(K) no fill')
plt.plot(x, tune_lsh_results["lsffill"],     label='LSH(K) fill')
plt.plot(x, tune_lsh_results["random"],   label='random')
plt.xlabel('Length of label k')
plt.ylabel('Average similarity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR + "lsh_tuning.png")

with open(DATA_DIR + str("results_sizes.json"), 'r') as fp:
    returned_docs_results = json.load(fp)

plt.figure()
plt.clf()
plt.plot(returned_docs_results["lengths"], returned_docs_results["results"])
plt.xlabel('Length of label k')
plt.ylabel('Average results size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR + "returned_docs.png")

with open(DATA_DIR + str("tree_level_stats.json"), 'r') as fp:
    returned_docs_results = json.load(fp)

plt.figure()
plt.clf()
plt.plot(returned_docs_results["levels"], returned_docs_results["nodes"])
plt.yscale("log", base=2)
plt.xlabel('Level of the LSH tree')
plt.ylabel('Average results size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR + "tree_level_stats.png")

with open(DATA_DIR + str("desired_size.json"), 'r') as fp:
    desired_results_json = json.load(fp)

plt.figure()
plt.clf()
plt.plot(desired_results_json["indexes"], desired_results_json["lsh"],   label='LSH(K)')
plt.plot(desired_results_json["indexes"], desired_results_json["forest"],     label='LSH A')
plt.plot(desired_results_json["indexes"], desired_results_json["forestsync"],     label='LSH S')
plt.plot(desired_results_json["indexes"], desired_results_json["random"],   label='random')
plt.xlabel('# results desired')
plt.ylabel('Average similarity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR + "desired_size.png")

with open(DATA_DIR + str("candidate_size.json"), 'r') as fp:
    candidates_results_json = json.load(fp)

plt.figure()
plt.clf()
plt.plot(candidates_results_json["indexes"], candidates_results_json["lsh"],   label='LSH(K)')
plt.plot(candidates_results_json["indexes"], candidates_results_json["forest"],     label='LSH A')
plt.plot(candidates_results_json["indexes"], candidates_results_json["forestsync"],     label='LSH S')
plt.plot(candidates_results_json["indexes"], candidates_results_json["random"],   label='random')
plt.xlabel('Candidate set size')
plt.ylabel('Average similarity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR + "candidate_size.png")