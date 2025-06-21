# preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Zàèéìòùç ]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

def get_cleaned_dataframe(json_path="comments.json"):
    df = pd.read_json(json_path, lines=True)
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(clean_text)
    return df
