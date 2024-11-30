#include necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

#import data from local device
data=pd.read_csv('C:\\Users\\wwwza\\OneDrive\\Desktop\\project\\coffee_sale_index.csv')

#print few rows to inspect
print(data.head())

#convert date to datetime
data['date'] = pd.to_datetime(data['date'])
print(data.dtypes)
data['Month'] = data['date'].dt.month
data['Year'] = data['date'].dt.year
print(data)
data.drop(columns=['date', 'datetime'], inplace=True)
print(data)

#check for null values and fill the appropriate replacement
print(data.isnull().sum())
data['card'] = data['card'].fillna('cash')

#display to confirm null data has replaced
print(data.isnull().sum())

#display sales by type
revenue=data.groupby(['coffee_name']).sum(['money']).reset_index().sort_values(by='money',ascending=False)
plt.figure(figsize=(10,5))
plt.title('Sales by type')
ax=sns.barplot(data=revenue, x='money', y='coffee_name')
ax.bar_label(ax.containers[0], fontsize=6)
plt.show()

time=data.groupby(['Month']).sum(['money']).reset_index().sort_values(by='Month',ascending=False)
plt.figure(figsize=(10,5))
ax=sns.barplot(data=time, x='Month', y='money')
ax.bar_label(ax.containers[0], fontsize=6)
plt.title('Sales every month')
plt.show()

#maschine learning
from sklearn.model_selection import train_test_split
# Define variable
X = data.drop(columns=['money'])
y = data['Month']

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


#visualize scatter plot for test and predicted data
plt.scatter(y_test,y_pred, color='green')
plt.xlabel('True Values (y_test)')
plt.ylabel('Predictions (y_pred)')
plt.title('Linear Regression: True vs Predicted Values')
plt.show()
