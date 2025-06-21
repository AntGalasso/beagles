import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')         
nltk.download('punkt_tab')
nltk.download('stopwords')
df = pd.read_json("comments.json", lines = True)
texts = df['text'].astype(str).tolist()

# Preprocessing
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Zàèéìòùç ]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

print(df['clean_text'].head(10))

