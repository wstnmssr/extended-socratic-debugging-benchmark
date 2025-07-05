import argparse
import traceback
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from datasets import Dataset
                
# 
def evaluate(model, tokenizer, dataset, to_path:str):
    rewards = {
        'prediction': []
    }
    template = "{role0}: {content0} {role1}: {content1}"


    for data in tqdm(dataset):
        prediction = template.format(role0=data['rejected'][0]['role'], content0=data['rejected'][0]['content'], role1=data['rejected'][1]['role'], content1=data['rejected'][1]['content']).replace('\n', ' ').replace('\\/', '/')

        with torch.no_grad():
            rewards_pred = None
            try:
                rewards_pred = model(**tokenizer(prediction, return_tensors='pt'))
            except Exception as e:
                print(e)
                traceback.print_exc()
                continue
            
            rewards['prediction'].append(rewards_pred.logits.item())
    df = DataFrame(rewards)
    df.to_json(to_path, orient='index')
    desc = df.describe()
    desc.to_json(to_path.replace('.json', '_desc.json'), orient='index')
    print(desc)

def main(path:str):
    dataset = Dataset.load_from_disk(path)
    to_path = path + '.json'
    tokenizer = AutoTokenizer.from_pretrained("/Users/wstnmssr/school/socratic-debugging-benchmark/reward_model/tokenizer/last_checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained(
        '/Users/wstnmssr/school/socratic-debugging-benchmark/reward_model/model/last_checkpoint', 
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    evaluate(model, tokenizer, dataset, to_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        type=str)
    args = parser.parse_args()
    main(args.path)