import argparse
import pandas as pd
from metrics.metric_computer import MetricComputer
import json

def main(path:str):
    df = pd.read_excel(path)
    df['reference_list'] = df['reference_list'].apply(lambda x: eval(x))
    df['prediction_list'] = df['prediction_list'].apply(lambda x: eval(x))

    computer = MetricComputer()
    scores = computer.compute_thoroughness(df['prediction_list'], df['reference_list'], contexts=df['context'])
    print(scores)
    with open(path.replace('.xlsx', '_scores.json'), 'w') as f:
        json.dump(scores, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
    )
    args = parser.parse_args()
    main(args.path)
