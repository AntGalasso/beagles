from textblob import TextBlob
from preprocessing import get_cleaned_dataframe
from pathlib import Path

# Risali di una cartella e vai in "data/comments.json"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "comments.json"

def lexicon_label(comment):
    polarity = TextBlob(comment).sentiment.polarity
    if polarity > 0.1:
        return "positivo"
    elif polarity < -0.1:
        return "negativo"
    else:
        return "neutro"

# Carica i commenti preprocessati dal path corretto
df = get_cleaned_dataframe(str(DATA_PATH))

# Applica il labelling
df['label'] = df['clean_text'].apply(lexicon_label)

# Stampa i primi 20 commenti con label
"""for i, row in df.head(20).iterrows():
    print(f"\n🗨️  Commento originale: {row['text']}")
    print(f"🧼 Commento pulito:     {row['clean_text']}")
    print(f"🏷️  Etichetta:           {row['label']}")"""


y_dict = df['label']