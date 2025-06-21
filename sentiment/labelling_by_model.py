import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from preprocessing import get_cleaned_dataframe

df = get_cleaned_dataframe()
# Carica pipeline di sentiment analysis (modello BERT fine-tuned)
sentiment_analyzer = pipeline("sentiment-analysis")

# Applica modello ai testi puliti (in batch se il dataset è grande)
results = sentiment_analyzer(df['clean_text'].tolist())

# Estrai etichette numeriche: POSITIVE -> 1, NEGATIVE -> 0
df['label'] = [1 if r['label'] == 'POSITIVE' else 0 for r in results]

# Stampa qualche esempio
#print(df[['clean_text', 'label']].head())

y_bert = df['label']

