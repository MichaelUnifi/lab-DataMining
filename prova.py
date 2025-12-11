import json
from utilities import ID_DATASET_PATH



with open(ID_DATASET_PATH, 'rb') as fp:
    data = json.load(fp)

print(data)