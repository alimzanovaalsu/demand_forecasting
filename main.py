import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from scipy.stats import norm, skew 
from scipy import stats 
import statsmodels.api as sm 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
from pylab import rcParams
import statsmodels.api as sm
import itertools

data=pd.read_csv(r'C:\Users\alimz\OneDrive\Desktop\FINAL PROJECT\Historical Product Demand.csv',parse_dates=['Date'])
df=data.copy()
print (df.isna().sum())
print ('Total Ratio of null values: ',df.isnull().sum()[3]/df.shape[0]*100) #  it is so small so we can easy drop them
df.dropna(axis=0, inplace=True) #remove all rows with na's.
df.reset_index(drop=True)
df.sort_values('Date')
print(df)

df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")
df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'])

if (df['Order_Demand'] < 0).any():
    print("There are negative values in the column.")
    negative_values = df[df['Order_Demand'] < 0]['Order_Demand']
    print("Negative values:")
    print(negative_values)
else:
    print("There are no negative values in the column.")

duplicate_rows = df[df.duplicated()]

print("Data Shape",df.shape)
# Check if there are duplicate rows
if not duplicate_rows.empty:
    print("Duplicate rows found, deleting...")
    # Drop duplicate rows and keep the first occurrence
    df.drop_duplicates(inplace=True)
    print("Duplicate rows deleted.")
else:
    print("No duplicate rows found.")

print("Data Shape",df.shape)

print("Data Types")
print(df.dtypes)
print(df)
print("Data Shape",df.shape)
print("The number of products is",len(df['Product_Code'].value_counts().index))
print("The number of categories is",len(df['Product_Category'].value_counts().index))
print("Period range is from",df['Date'].min(),"to", df['Date'].max())
print("Order Quantity is from",df['Order_Demand'].min(),"to", df['Order_Demand'].max())
print('Number of missing values by column',[sum(df[i].isnull()) for i in df.columns])

print(df['Date'].min() , df['Date'].max())

#lets get a period of 2012 and 2016, to have data of whole 4 years
df = df[(df['Date']>='2012-01-01') & (df['Date']<='2016-12-31')].sort_values('Date', ascending=True)
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['weekday_or_weekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
def get_part_of_month(date):
    day = date.day
    if day <= 10:
        return 0  # Beginning of the month
    elif day <= 20:
        return 1  # Middle of the month
    else:
        return 2  # End of the month
    
# Apply the function to the 'Date' column and create a new column 'Part_of_Month'
df['Part_of_Month'] = df['Date'].apply(get_part_of_month)

def get_season(month):
    if month in [12, 1, 2]:  # Winter: December, January, February
        return 1
    elif month in [3, 4, 5]:  # Spring: March, April, May
        return 2
    elif month in [6, 7, 8]:  # Summer: June, July, August
        return 3
    else:                      # Autumn: September, October, November
        return 4

# Apply the function to the 'Month' column (assuming 'Month' contains month numbers)
df['Seasonal_Indicator_Winter'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
df['Seasonal_Indicator_Spring'] = df['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
df['Seasonal_Indicator_Summer'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
df['Seasonal_Indicator_Fall'] = df['Month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)

# Calculate the frequency counts of each category
category_counts = df['Product_Category'].value_counts()
warehouse_counts=df['Warehouse'].value_counts()
product_counts=df['Product_Code'].value_counts()
# Calculate the total number of observations
total_observations_cat = len(df)
total_observations_war=len(df)
total_observations_pr=len(df)

# Calculate the percentage of each category
category_percentages = (category_counts / total_observations_cat) * 100
warehouse_percentage=(warehouse_counts/total_observations_war)*100
product_percentage=(product_counts/total_observations_pr)*100

# Create a dictionary with category percentages
category_percentage_dict = category_percentages.to_dict()
warehouse_percentage_dict = warehouse_percentage.to_dict()
product_percentage_dict = product_percentage.to_dict()
# Map values to a new column using the dictionary
df['Category_Percentage'] = df['Product_Category'].map(category_percentage_dict)
df['Warehouse_Percentage'] = df['Warehouse'].map(warehouse_percentage_dict)
df['Product_Percentage'] = df['Product_Code'].map(product_percentage_dict)

# Display the DataFrame with the new column
df.drop(['Warehouse','Product_Category'], axis=1, inplace=True)

print(df.describe)
print(df)
print(df.describe)
# Select only numeric columns
numeric_columns = df.select_dtypes(include=['int', 'float']).columns
string_columns = df.select_dtypes(include=['object'])

# Calculate the correlation matrix for numeric columns only
correlation_matrix = df[numeric_columns].corr()

# Extract the correlation of each numeric column with the target column ('Order_Demand')
correlation_with_target = correlation_matrix['Order_Demand']

# Display the correlation of each numeric column with the target column
print(correlation_with_target)

columns_to_drop = correlation_with_target[correlation_with_target < 0].index

# Drop the columns with correlation less than 0
df.drop(columns=columns_to_drop, inplace=True)
print(df)
print(df.columns)

print(df['weekday_or_weekend'].unique())

window_size = 30  # Window size for moving average (e.g., 30 days)
df['Order_Demand_Moving_Avg'] = df['Order_Demand'].rolling(window=window_size).mean()

numeric_columns = df.select_dtypes(include=['int', 'float']).columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

df.to_csv("filtereddata.csv")

print(df)

# Plot the count of each product category
plt.figure(figsize=(50, 14))
sns.countplot(data=df, x='Product_Code', order=df['Product_Code'].value_counts().index.sort_values(ascending=True))
plt.show()


selected_category = 'Product_0001'
filtered_data = df[df['Product_Code'] == selected_category]
# Group filtered data by Date and Product_Category and sum Order_Demand
grouped_data = filtered_data.groupby(['Date', 'Product_Code'], as_index=False)['Order_Demand'].sum()
# Prepare data
y2 = grouped_data.set_index('Date')['Order_Demand'].resample('MS').sum()

# Plot time series for the selected product
plt.plot(y2, label=selected_category)
plt.ylabel('Order Demand')
plt.title(f'Order Demand for Product Code: {selected_category}')
plt.legend()
# Set common x-axis label
plt.xlabel('Date')
# Show plot
plt.show()

decomposition = sm.tsa.seasonal_decompose(y2, model='additive')
fig = decomposition.plot()
plt.show()


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
            mod = sm.tsa.statespace.SARIMAX(y2,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))


mod = sm.tsa.statespace.SARIMAX(y2,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


pred = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False) #false is when using the entire history.
#Confidence interval.
pred_ci = pred.conf_int()

#Plotting real and forecasted values.
ax = y2['2012':].plot(label='observed') #=======================
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()

y2_forecasted = pred.predicted_mean
y2_truth = y2['2016-01-01':]
mse = ((y2_forecasted - y2_truth) ** 2).mean()
print('MSE {}'.format(round(mse, 2)))
print('RMSE: {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()
ax = y2.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()
plt.show()
print(y2)

residuals = y2['2016-01-01':] - pred.predicted_mean

# Calculate MSE
mse = (residuals ** 2).mean()

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAE
mae = np.abs(residuals).mean()

# Calculate MAPE
absolute_percentage_errors = np.abs(residuals / y2_truth)
mape = np.mean(absolute_percentage_errors) * 100

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
