# Stock Market Analysis for Bank of America.

Took data for the last 10 years of BAC i.e starting from 1st April 2023 (1st day of Finacial Year) to current day-5. You can change the starting date by giving year as input, and make changes in data.
In case you don't download the BAC data as it contain details till Nov 2023, the code will automatically download the data using _yfinace_ library till 5 days prior to today's date.
Outliers are checked on every input column. As Volume column contains huge number of outliers, it has been treated with IQR. Lear more about IQ here : https://www.analyticsvidhya.com/blog/2022/09/dealing-with-outliers-using-the-iqr-method/

### Simple Model Using Random Forset

1. Created a Random Forest Model to predict accuracy and precision of the model. The precison and accuracy comes around 47 and 51 % respectiverly.
2. Back testing is impelmented like last day data, last week data, last quarter data, last 6 months data. Using back-testing the precision gets improved by around 6%.

### Model created using ARIMA

1. we need to calculate p,d,q value need to pass in order in ARIMA model.
2. Firstly ADF test is run on the data. We plot graph to check the no of difference. We can check from plot only 1 difference is fine, and we can double confirm by _ndiffs_ class. So d value come as1.
3. We plot pacf to caluclate the p value in order.
4. We check the q value using acf plot.
5. Mean Absolute Error, Root mean squared Error using ARIMA model gives under 5.

We then calculate the order using AutoArima and then build model.


