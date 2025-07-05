from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from statistics import mean, median, mode, stdev

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

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

def collect_scores(text, scores):
    sentiment = analyzer.polarity_scores(text)
    scores['compound'].append(sentiment['compound'])
    scores['neg'].append(sentiment['neg'])
    scores['neu'].append(sentiment['neu'])
    scores['pos'].append(sentiment['pos'])

df = pd.read_csv('/Users/wstnmssr/school/socratic-debugging-benchmark/results/chatgpt/comprehensive_prompt_multiple_1_responses.csv')
df = df[['prediction', 'reference']]
df['pred_sentiment'] = df['prediction'].apply(collect_scores, scores=chatgpt_pred_scores)
df['ref_sentiment'] = df['reference'].apply(collect_scores, scores=chatgpt_ref_scores)

df = pd.read_csv('/Users/wstnmssr/school/socratic-debugging-benchmark/results/gpt4/comprehensive_prompt_multiple_1_responses.csv')
df = df[['prediction', 'reference']]
df['pred_sentiment'] = df['prediction'].apply(collect_scores, scores=gpt4_pred_scores)
df['ref_sentiment'] = df['reference'].apply(collect_scores,  scores=gpt4_ref_scores)
df.to_csv('vader_data.csv', index=False) 

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
