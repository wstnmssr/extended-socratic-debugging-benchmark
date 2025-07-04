from datasets import Dataset
import itertools
import pandas as pd
from pandas import DataFrame
import os

def generate_dataset(df: DataFrame, data_dict, multiple=False):
    for _, row in df.iterrows():
        context = row['context']
        # only need prediction list if generating multiple responses
        prediction = row['prediction_list'][1:-1].split('\', \'') if multiple else row['prediction']
        reference = row['reference_list'][1:-1].split('\', \'')
        user = {"role": "user", "content": context}
        if multiple:
            for ref, pred in itertools.product(reference, prediction):
                assistant = {"role": "assistant", "content": pred}
                data_dict['rejected'].append([user, assistant])
                
                assistant = {"role": "assistant", "content": ref}
                data_dict['chosen'].append([user, assistant])
        else:
            for ref in reference:
                assistant = {"role": "assistant", "content": prediction}
                data_dict['rejected'].append([user, assistant])
                
                assistant = {"role": "assistant", "content": ref}
                data_dict['chosen'].append([user, assistant])
                
paths = []
for root, dirs, files in os.walk("results/cache/good"):
    for file in files:
        if file.endswith(".csv"):
            paths.append(os.path.join(root, file))

for path in paths:             
    data_dict = { 'chosen': [], 'rejected': [] }
    df = pd.read_csv(path)
    generate_dataset(df, data_dict, multiple=True)
    dataset = Dataset.from_dict(data_dict)
    new_path = path.replace('.csv', '_dataset')
    print('saving to: ', new_path)
    dataset.save_to_disk(new_path)
