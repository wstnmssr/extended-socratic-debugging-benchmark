import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from datasets import Dataset
paths = [
    'llama/comprehensive_prompt_multiple_1_responses_java_dataset.json',
    'llama/comprehensive_prompt_multiple_1_responses_python_dataset.json',
    'llama/comprehensive_prompt_cot_1_responses_java_dataset.json',
    'llama/comprehensive_prompt_cot_1_responses_python_dataset.json',
    'gpt4/comprehensive_prompt_multiple_1_responses_java_dataset.json',
    'gpt4/comprehensive_prompt_multiple_1_responses_python_dataset.json',
    'gpt4/comprehensive_prompt_cot_1_responses_java_dataset.json',
    'gpt4/comprehensive_prompt_cot_1_responses_python_dataset.json',
    'gpt3.5/comprehensive_prompt_multiple_1_responses_java_dataset.json',
    'gpt3.5/comprehensive_prompt_multiple_1_responses_python_dataset.json',
    'gpt3.5/comprehensive_prompt_cot_1_responses_java_dataset.json',
    'gpt3.5/comprehensive_prompt_cot_1_responses_python_dataset.json'
]
df_all = DataFrame()
for path in paths:
    dataset = Dataset.from_json(path)
    df = dataset.to_pandas()
    df = df.transpose()
    df = df.map(lambda x: x.get('prediction'))
    col_name = path.replace('/', '_').replace('_dataset.json', '').replace('comprehensive_prompt_multiple_1_responses_', '').replace('comprehensive_prompt_cot_1_responses', 'cot')
    df.columns = [col_name]
    # print(df.head())
    df_all = pd.concat([df_all, df], axis=1)
    
print(df_all.head())
df_all.boxplot(figsize=(20, 6))
plt.ylabel('Reward Model Score')
plt.show()