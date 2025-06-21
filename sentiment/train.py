from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from feature_extraction import extract_features
from preprocessing import get_cleaned_dataframe
from labelling_by_dict import y_dict
from labelling_by_model import y_bert

# Caricamento dati e feature extraction
df = get_cleaned_dataframe(json_path="../data/comments.json")
X, feature_names = extract_features(df)

def train_model(X, y, name=""):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Output
    print(f"\n🔍 Risultati con etichettatura {name}")
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Training con BERT
train_model(X, y_bert, name="da modello pre-addestrato")

# Training con dizionario
train_model(X, y_dict, name="da dizionario (TextBlob o simili)")
