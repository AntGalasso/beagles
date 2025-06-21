import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Download risorse nltk necessarie
nltk.download('punkt')
nltk.download('stopwords')

# Carica il file JSON (assumendo formato JSON Lines)
df = pd.read_json("comments.json", lines=True)

# Preprocessing testo
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # rimuove link
    text = re.sub(r"[^a-zA-Zàèéìòùç ]", "", text)  # lascia solo lettere e spazi
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]  # stopwords inglesi
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Carica pipeline di sentiment analysis (modello BERT fine-tuned)
sentiment_analyzer = pipeline("sentiment-analysis")

# Applica modello ai testi puliti (in batch se il dataset è grande)
results = sentiment_analyzer(df['clean_text'].tolist())

# Estrai etichette numeriche: POSITIVE -> 1, NEGATIVE -> 0
df['label'] = [1 if r['label'] == 'POSITIVE' else 0 for r in results]

# Stampa qualche esempio
print(df[['clean_text', 'label']].head())


#print(df['clean_text'].head(10))

