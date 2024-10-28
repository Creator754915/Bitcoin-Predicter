# Bitcoin-Predicter

<br>

## Prophet Settings

<br>
Download the data from internet

```py
df = yf.download(
    "BTC-USD",
    start="2019-01-01",
    end="2024-03-06",
    interval="1d"
)
```

<br>
Hyperparameter of Prophet modified

```py
# Initialisation du modèle avec hyperparamètres ajustés
model_prophet = Prophet(
    changepoint_range=0.95,  # Plage de changement ajustée
    changepoint_prior_scale=0.15, # Ajouter de nouveaux points de changement
    seasonality_prior_scale=1.5,  # Échelle de saisonnalité ajustée
)
```

<br>
Sesonality settings

```py
model_prophet.add_country_holidays(country_name='US')                     # Hollidays of the USA
model_prophet.add_seasonality(name="annual", period=365, fourier_order=8) # Model of a basic 365 days year
```

<br>

## Model Prediction Table

| Row       | ds         | yhat           | yhat_lower   | yhat_upper    |
| :-------- | :--------- | :------------- | ------------ | ------------- |
| 0	        | 2019-01-01 | 1825.660589    | -938.724612  | 4615.102245   |
| 1	        | 2019-01-02 | 2452.785371    | -173.765656	 | 5269.470747   |
| 2	        | 2019-01-03 | 2704.465776    | -240.673647  | 5517.625968   |
| 3	        | 2019-01-04 | 3027.981416    | 321.280947	 | 5851.345484   |
| 4	        | 2019-01-05 | 3316.344694    | 633.547490	 | 6301.613011   |
| ...	    | ...	     | ...	          | ...	         | ...           |
| 1886	    | 2024-07-29 | 67430.916368   | -55403.097488| 194796.751060 |
| 1887	    | 2024-07-30 | 67919.745045   | -56951.865388| 194743.450099 |
| 1888	    | 2024-07-31 | 68422.504826   | -56550.890794| 197425.350436 |
| 1889	    | 2024-08-01 | 68724.121673   | -57338.220283| 197346.131072 |
| 1890	    | 2024-08-02 | 69112.080500   | -58477.960423| 198117.128924 |
| 1891 rows × 4 columns |

Legend:

ds => **Date**

yhat => **Price Prediction**

yhat_lower => **Lower Price Prediction**

yhat_upper => **Upper Price Prediction**

## Import Point

We can see the model predit with lots of *accuracy* the **price** of Bitcoin when we saw the **70000$ up**.

<br>
<br>

## Model Prediction

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/bitcoin_prediction_plot.png)

## Model Changepoints

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/figure_with_changepoints.png)

## Model Fail Rate

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/taux_derreurs_ia.png)

<br>

# Bitcoin Predicter Beta V2

## Model Prediciton

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/Test%20Beta%20V2/bitcoin_prediction_beta_v2.png)

## Model Changepoints

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/Test%20Beta%20V2/changepoints_beta_v2.png)

## Model Fail Rate

![no image](https://github.com/Creator754915/Bitcoin-Predicter/blob/main/Test%20Beta%20V2/taux_derreurs_moyenne_MAE_RMSE_beta_v2.png)

<br>

## PRO ZONE

In statistics, mean absolute error (MAE) is a measure of errors between paired observations expressing the same phenomenon. Examples of Y versus X include comparisons of predicted versus observed, subsequent time versus initial time, and one technique of measurement versus an alternative technique of measurement. MAE is calculated as the sum of absolute errors divided by the sample size.

![MAE FORMULE](https://wikimedia.org/api/rest_v1/media/math/render/svg/3ef87b78a9af65e308cf4aa9acf6f203efbdeded)

The root mean square error (RMSE) or root mean square deviation (RMSE) is a frequently used measure of the differences between the values ​​(sample or population values) predicted by a model or estimator and the observed values ​​(or true values). The REQM represents the square root of the second sampling time of the differences between the predicted values ​​and the observed values. These deviations are called residuals when the calculations are performed on the data sample that was used for the estimation or they are called errors (or prediction errors) when they are calculated on data outside the sample. estimate. REQM aggregates prediction errors from different data points into a single measure of increased predictive power. REQM is a measure of accuracy, which is used to compare the errors of different predictive models for a particular dataset and not between different datasets, as it is scale dependent1.

![RMSE FORMULE](https://wikimedia.org/api/rest_v1/media/math/render/svg/b343f69f0e089cecedace789c161c540265f97ae)
