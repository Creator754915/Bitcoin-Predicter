# Bitcoin-Predicter

## Prophet Settings

<br>
Télécharger les données depuis internet

```py
df = yf.download(
    "BTC-USD",
    start="2019-01-01",
    end="2024-03-06",
    interval="1d"
)
```

<br>
Hyperparameter modifié de Prophet

```py
# Initialisation du modèle avec hyperparamètres ajustés
model_prophet = Prophet(
    changepoint_range=0.95,  # Plage de changement ajustée
    changepoint_prior_scale=0.15, # Ajouter de nouveaux points de changement
    seasonality_prior_scale=1.5,  # Échelle de saisonnalité ajustée
)
```

<br>
Option des paramètres saisonier

```py
model_prophet.add_country_holidays(country_name='US')
model_prophet.add_seasonality(name="annual", period=365, fourier_order=8)
```

## Model Prediction Text

```txt
	    ds	           yhat          yhat_lower	 yhat_upper
0	2019-01-01	1825.660589	-938.724612	4615.102245
1	2019-01-02	2452.785371	-173.765656	5269.470747
2	2019-01-03	2704.465776	-240.673647	5517.625968
3	2019-01-04	3027.981416	321.280947	5851.345484
4	2019-01-05	3316.344694	633.547490	6301.613011
...	...	...	...	...
1886	2024-07-29	67430.916368	-55403.097488	194796.751060
1887	2024-07-30	67919.745045	-56951.865388	194743.450099
1888	2024-07-31	68422.504826	-56550.890794	197425.350436
1889	2024-08-01	68724.121673	-57338.220283	197346.131072
1890	2024-08-02	69112.080500	-58477.960423	198117.128924
1891 rows × 4 columns
```
ds => Date

yhat => Price Prediction

yhat_lower => Lower Price Prediction

yhat_upper => Upper Price Prediction

<br>
<br>

## Model Prediction

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/bitcoin_prediction_plot.png)

## Model Changepoints

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/figure_with_changepoints.png)

## Taux d'échecs

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/taux_derreurs_ia.png)
