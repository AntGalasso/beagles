from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import get_cleaned_dataframe
from pathlib import Path

# Percorso file dati (modifica se serve)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "comments.json"

def extract_features(df):
    """
    Dataframe con colonna 'clean_text'
    Ritorna matrice TF-IDF e il vettore delle feature names
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # max_features limita dimensione feature space
    X = vectorizer.fit_transform(df['clean_text'])
    return X, vectorizer.get_feature_names_out()

if __name__ == "__main__":
    # Carica dataframe preprocessato
    df = get_cleaned_dataframe(str(DATA_PATH))

    # Estrai features TF-IDF
    X, features = extract_features(df)

    print(f"Numero di commenti: {X.shape[0]}")
    print(f"Dimensione feature matrix: {X.shape}")
    print(f"Esempio feature names: {features[:20]}")
