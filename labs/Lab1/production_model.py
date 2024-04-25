import pandas as pd
import joblib

# Ladda in test_samples.csv
test_samples = pd.read_csv('data/test_samples.csv')

# Ladda in modellen från .pkl-filen
model = joblib.load('din_modell.pkl')

# Gör förutsägelser på de 100 datapunkterna
predictions = model.predict(test_samples)

# Beräkna sannolikheterna för varje klass
probabilities = model.predict_proba(test_samples)
probability_class_0 = probabilities[:, 0]
probability_class_1 = probabilities[:, 1]

# Skapa en DataFrame med förutsägelser och sannolikheter
prediction_df = pd.DataFrame({
    'probability class 0': probability_class_0,
    'probability class 1': probability_class_1,
    'prediction': predictions
})

# Exportera DataFrame till CSV
prediction_df.to_csv('data/prediction.csv', index=False)
