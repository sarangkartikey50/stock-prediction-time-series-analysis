import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
color = sns.color_palette()
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np


def parser(x):
    return datetime.strptime(x, '%Y-%m')


print("please wait. importing...")
data = pd.read_csv("dataset.csv")

print("importing finished.")

data['Total_Price'] = data['Quantity']*data['UnitPrice']

data = data[data.Quantity > 0]
data = data[data.UnitPrice > 0]
data = data[data.iloc[:, :] != '']


data['date'] = data['InvoiceDate'].map(lambda x: str(x)[:7])

#print(data)
Quantity_date=data[['date', 'Quantity']].drop_duplicates()
Quantity_date_count=Quantity_date.groupby(['date'])['Quantity'].aggregate('sum').reset_index().sort_values(by='date', ascending=0)

date=list(Quantity_date_count['date'])
quantity=list(Quantity_date_count['Quantity'])
date_quantity = pd.DataFrame({'dates': date, 'quantity':quantity})
#date_quantity['quantity'] = date_quantity['quantity'].map(lambda x: str(x))


date_quantity.index = date_quantity['dates'].map(lambda x: parser(x))
date_quantity['quantity'] = date_quantity['quantity'].map(lambda x: float(x))



date_quantity = date_quantity.fillna(date_quantity.bfill())
date_quantity = date_quantity['quantity'].resample('MS').mean()
print(date_quantity)
date_quantity.plot()
plt.show()

autocorrelation_plot(date_quantity)
plt.show()

quantity = date_quantity.values

size = int(len(quantity) * 0.66)
train, test = quantity[0:size], quantity[size:len(quantity)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2 ,2 ,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat[0])
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

pred = np.array(predictions)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()