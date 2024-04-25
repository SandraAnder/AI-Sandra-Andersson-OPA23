import pandas as pd
import joblib

# Laddar in test_samples med errorhandling
try:
    test_samples = pd.read_csv('labs/Lab1/data/test_samples.csv')
    print("Filen lästes in korrekt.")
except FileNotFoundError as e:
    print("Filen kunde inte hittas:", e)

# Laddar in min modell från .pkl-filen
try:
    model = joblib.load('labs/Lab1/data/svm_model.pkl')
except FileNotFoundError as e:
    print("Filen kunde inte hittas:", e)

# Gör prediktioner på de 100 datapunkterna
predictions = model.predict(test_samples)

# Beräknar sannolikheterna för varje klass
probabilities = model.predict_proba(test_samples)
probability_class_0 = probabilities[:, 0]
probability_class_1 = probabilities[:, 1]

# Skapar en ny DataFrame med prediktioner och sannolikheter
prediction_df = pd.DataFrame({
    'probability class 0': probability_class_0,
    'probability class 1': probability_class_1,
    'prediction': predictions
})

# Exporterar DataFramen till en CSV-fil
prediction_df.to_csv('data/prediction.csv', index=False)
