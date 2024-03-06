import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Téléchargement des données
df = yf.download(
    "BTC-USD",
    start="2019-01-01",
    end="2024-03-06",
    interval="1d"
)

# Ingénierie des caractéristiques
df['Volume'] = df['Volume'].fillna(0)  # Remplacer les NaN par 0 pour le volume
btc_close = df[['Close', 'Volume']]
btc_close = btc_close.reset_index(drop=False)
btc_close.columns = ['ds', 'y', 'Volume']

# Séparation des données d'entraînement et de test
btc_train, btc_test = train_test_split(
    btc_close,
    test_size=0.2,
    shuffle=False,
    random_state=42
)

# Initialisation du modèle avec hyperparamètres ajustés
model_prophet = Prophet(
    changepoint_range=0.95,  # Plage de changement ajustée
    changepoint_prior_scale=0.15,
    seasonality_prior_scale=1.5,  # Échelle de saisonnalité ajustée
)
model_prophet.add_country_holidays(country_name='US')
model_prophet.add_seasonality(name="annual", period=365, fourier_order=8)

# Ajout de la régresseur supplémentaire (Volume)
model_prophet.add_regressor('Volume')

# Entraînement du modèle
model_prophet.fit(btc_train)

# Création du DataFrame Futur
btc_future = model_prophet.make_future_dataframe(
    periods=len(btc_test),
    freq="B"
)
btc_future['Volume'] = btc_close['Volume']  # Étendre le DataFrame futur avec les données de volume

# Prédictions
btc_pred = model_prophet.predict(btc_future)

# Enregistrement des prédictions dans un fichier texte
with open('bitcoin_prediction.txt', 'w') as file:
    for index, row in btc_pred.iterrows():
        file.write(f"{row['ds']}, {row['yhat']}, {row['yhat_lower']}, {row['yhat_upper']}\n")

# Affichage des prédictions
plt.figure(figsize=(10, 6))
plt.plot(btc_pred['ds'], btc_pred['yhat'], label='Prédiction', color='blue')
plt.fill_between(btc_pred['ds'], btc_pred['yhat_lower'], btc_pred['yhat_upper'], color='skyblue', alpha=0.4)
plt.xlabel('Date')
plt.ylabel('Prix BTC')
plt.title('Prédiction du Prix du Bitcoin')
plt.legend()
plt.grid(True)

# Sauvegarde de la figure en tant qu'image
plt.savefig('bitcoin_prediction_plot.png')
plt.show()

from prophet.plot import add_changepoints_to_plot

fig = model_prophet.plot(btc_pred)
add_changepoints_to_plot(fig.gca(), model_prophet, btc_pred)

plt.savefig('figure_with_changepoints.png')

# Sauvegarde du modèle
joblib.dump(model_prophet, 'btc_predic_model.pkl')
