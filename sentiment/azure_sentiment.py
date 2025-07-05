from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import os
import pandas as pd
from statistics import mean, median, mode, stdev

gpt4_pred_scores = {
    'neg': [],
    'neu': [],
    'pos': []
}

gpt4_ref_scores = {
    'neg': [],
    'neu': [],
    'pos': []
}

chatgpt_pred_scores = {
    'neg': [],
    'neu': [],
    'pos': []
}

chatgpt_ref_scores = {
    'neg': [],
    'neu': [],
    'pos': []
}

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def analyze_sentiment(documents, scores) -> None:
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    key = os.environ["AZURE_LANGUAGE_KEY"]

    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    result = text_analytics_client.analyze_sentiment(documents, show_opinion_mining=True)
    docs = [doc for doc in result if not doc.is_error]

    for _, doc in enumerate(docs):
        scores['neg'].append(doc.confidence_scores['negative'])
        scores['neu'].append(doc.confidence_scores['neutral'])
        scores['pos'].append(doc.confidence_scores['positive'])

df = pd.read_csv('/Users/wstnmssr/school/socratic-debugging-benchmark/results/chatgpt/comprehensive_prompt_multiple_1_responses.csv')
df = df[['prediction', 'reference']]
for batch_df in batch(df['prediction'].to_list(), 10):
    analyze_sentiment(batch_df, chatgpt_pred_scores)

for batch_df in batch(df['reference'].to_list(), 10):
    analyze_sentiment(batch_df, chatgpt_ref_scores)

df = pd.read_csv('/Users/wstnmssr/school/socratic-debugging-benchmark/results/gpt4/comprehensive_prompt_multiple_1_responses.csv')
df = df[['prediction', 'reference']]
for batch_df in batch(df['prediction'].to_list(), 10):
    analyze_sentiment(batch_df, gpt4_pred_scores)

for batch_df in batch(df['reference'].to_list(), 10):
    analyze_sentiment(batch_df, gpt4_ref_scores)

print('---ChatGPT prediction---')
print(f'chatgpt pos mean {mean(chatgpt_pred_scores['pos'])}')
print(f'chatgpt pos median {median(chatgpt_pred_scores['pos'])}')
print(f'chatgpt pos mode {mode(chatgpt_pred_scores['pos'])}')
print(f'chatgpt pos stddev {stdev(chatgpt_pred_scores['pos'])}')
print(f'chatgpt neg mean {mean(chatgpt_pred_scores['neg'])}')
print(f'chatgpt neg median {median(chatgpt_pred_scores['neg'])}')
print(f'chatgpt neg mode {mode(chatgpt_pred_scores['neg'])}')
print(f'chatgpt neg stddev {stdev(chatgpt_pred_scores['neg'])}')
print(f'chatgpt neu mean {mean(chatgpt_pred_scores['neu'])}')
print(f'chatgpt neu median {median(chatgpt_pred_scores['neu'])}')
print(f'chatgpt neu mode {mode(chatgpt_pred_scores['neu'])}')
print(f'chatgpt neu stddev {stdev(chatgpt_pred_scores['neu'])}')

print('---ChatGPT reference---')
print(f'chatgpt pos mean {mean(chatgpt_ref_scores['pos'])}')
print(f'chatgpt pos median {median(chatgpt_ref_scores['pos'])}')
print(f'chatgpt pos mode {mode(chatgpt_ref_scores['pos'])}')
print(f'chatgpt pos stddev {stdev(chatgpt_ref_scores['pos'])}')
print(f'chatgpt neg mean {mean(chatgpt_ref_scores['neg'])}')
print(f'chatgpt neg median {median(chatgpt_ref_scores['neg'])}')
print(f'chatgpt neg mode {mode(chatgpt_ref_scores['neg'])}')
print(f'chatgpt neg stddev {stdev(chatgpt_ref_scores['neg'])}')
print(f'chatgpt neu mean {mean(chatgpt_ref_scores['neu'])}')
print(f'chatgpt neu median {median(chatgpt_ref_scores['neu'])}')
print(f'chatgpt neu mode {mode(chatgpt_ref_scores['neu'])}')
print(f'chatgpt neu stddev {stdev(chatgpt_ref_scores['neu'])}')

print('---GPT4 prediction---')
print(f'gpt4 pos mean {mean(gpt4_pred_scores['pos'])}')
print(f'gpt4 pos median {median(gpt4_pred_scores['pos'])}')
print(f'gpt4 pos mode {mode(gpt4_pred_scores['pos'])}')
print(f'gpt4 pos stddev {stdev(gpt4_pred_scores['pos'])}')
print(f'gpt4 neg mean {mean(gpt4_pred_scores['neg'])}')
print(f'gpt4 neg median {median(gpt4_pred_scores['neg'])}')
print(f'gpt4 neg mode {mode(gpt4_pred_scores['neg'])}')
print(f'gpt4 neg stddev {stdev(gpt4_pred_scores['neg'])}')
print(f'gpt4 neu mean {mean(gpt4_pred_scores['neu'])}')
print(f'gpt4 neu median {median(gpt4_pred_scores['neu'])}')
print(f'gpt4 neu mode {mode(gpt4_pred_scores['neu'])}')
print(f'gpt4 neu stddev {stdev(gpt4_pred_scores['neu'])}')

print('---GPT4 reference---')
print(f'gpt4 pos mean {mean(gpt4_ref_scores['pos'])}')
print(f'gpt4 pos median {median(gpt4_ref_scores['pos'])}')
print(f'gpt4 pos mode {mode(gpt4_ref_scores['pos'])}')
print(f'gpt4 pos stddev {stdev(gpt4_ref_scores['pos'])}')
print(f'gpt4 neg mean {mean(gpt4_ref_scores['neg'])}')
print(f'gpt4 neg median {median(gpt4_ref_scores['neg'])}')
print(f'gpt4 neg mode {mode(gpt4_ref_scores['neg'])}')
print(f'gpt4 neg stddev {stdev(gpt4_ref_scores['neg'])}')   
print(f'gpt4 neu mean {mean(gpt4_ref_scores['neu'])}')
print(f'gpt4 neu median {median(gpt4_ref_scores['neu'])}')
print(f'gpt4 neu mode {mode(gpt4_ref_scores['neu'])}')
print(f'gpt4 neu stddev {stdev(gpt4_ref_scores['neu'])}')
