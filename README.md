# Bitcoin-Predicter

## Prophet Settings

<br>
Hyperparameter modifié de Prophet

```py
# Initialisation du modèle avec hyperparamètres ajustés
model_prophet = Prophet(
    changepoint_range=0.95,  # Plage de changement ajustée
    changepoint_prior_scale=0.15,
    seasonality_prior_scale=1.5,  # Échelle de saisonnalité ajustée
)
```

<br>
Option des paramètres saisonier

```py
model_prophet.add_country_holidays(country_name='US')
model_prophet.add_seasonality(name="annual", period=365, fourier_order=8)
```

## Model Prediction

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/bitcoin_prediction_plot.png)

## Model Changepoints

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/figure_with_changepoints.png)

## Taux d'échecs

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/taux_derreurs_ia.png)
