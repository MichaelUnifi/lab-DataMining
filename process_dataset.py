import math
from collections import Counter
import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import random
import json
from utilities import DATASET_PATH, ID_DATASET_PATH, DOCS_TOKENS_PATH

nltk.download('reuters')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_tokens(tokens):
    processed = []
    mapping = []  # list of (raw, processed) pairs in order
    for t in tokens:
        if not t.isalpha():
            continue
        raw = t
        t1 = raw.lower()
        if t1 in stop_words:
            continue
        t2 = stemmer.stem(t1)
        processed.append(t2)
        mapping.append((raw, t2))
    return processed, mapping

#build a dict for all documents (bag of words)
dataset = {}
for doc_id in reuters.fileids():
    raw_words = reuters.words(doc_id)
    processed, mapping = preprocess_tokens(raw_words)
    dataset[doc_id] = processed
print(f"Total number of documents: {len(dataset)}")
dataset = dataset
term_to_id = {}
current_id = 0
for document in dataset.values():
    for term in document:
        if term not in term_to_id:
            term_to_id[term] = current_id
            current_id += 1
print(f"Total unique terms in corpus: {len(term_to_id)}")
print(f"Sample term to ID mapping (first 20 terms): {list(term_to_id.items())[:20]}")

#map each document's term to its unique id, keep track of the weights (tf format, we'll apply log afterwards)
id_dataset = {}
document_weights = {}
all_terms = list(term_to_id.values())

for name, document in zip(list(dataset.keys()), list(dataset.values())):
    indexed_terms = {}
    for term in all_terms:
        indexed_terms[term] = 0
    for term in document:
        indexed_terms[term_to_id[term]] += 1
    id_dataset[name] = indexed_terms

for document in id_dataset.values():
    for term_id in document.keys():
        count = document[term_id]
        if count > 0:
            document[term_id]= 1 + math.log(count)


doc_ids = list(id_dataset.keys())
docs_tokens = {}
docs_counts = id_dataset.values()


for doc_id, doc in zip(doc_ids, docs_counts):
    doc_tokens = []
    for token, count in zip(doc.keys(), doc.values()):
        if count > 0:
            doc_tokens.append(token)
    docs_tokens[doc_id] = doc_tokens


with open(DATASET_PATH, 'w') as fp:
    json.dump(dataset, fp)
with open(ID_DATASET_PATH, 'w') as fp:
    json.dump(id_dataset, fp)
with open(DOCS_TOKENS_PATH, 'w') as fp:
    json.dump(docs_tokens, fp)
    