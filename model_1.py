import numpy as np
import json
import math
from keras.src.models import Sequential
from keras.src.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(5)

# Load and prepare the data
with open('./crash_history.json', 'r') as file:
    data = json.load(file)
data = np.array(data)

threshold = int(input("Threshold: "))

# Convert data to pandas Series
data = np.array(data)
filtered_data = data[data < threshold]
print(len(filtered_data) / len(data) * 100)
data = filtered_data
dataset=data.reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets, 50% test data, 50% training data
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1, timestep 240
look_back = 200
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))
# 2nd LSTM layer
# * units = add 50 neurons is the dimensionality of the output space
# * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
model.add(LSTM(units=50, return_sequences=True))
# 20% of the layers will be dropped
model.add(Dropout(0.2))
# 3rd LSTM layer
# * units = add 50 neurons is the dimensionality of the output space
# * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
model.add(LSTM(units=50, return_sequences=True))
# 50% of the layers will be dropped
model.add(Dropout(0.5))
# 4th LSTM layer
# * units = add 50 neurons is the dimensionality of the output space
model.add(LSTM(units=50))
# 50% of the layers will be dropped
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=100, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Prepare the arrays for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

# Create the Plotly figure
fig = go.Figure()

# Plot baseline dataset
fig.add_trace(go.Scatter(y=scaler.inverse_transform(dataset).flatten(), mode='lines', name='Actual Values', ))

# Plot train predictions
fig.add_trace(go.Scatter(y=trainPredictPlot.flatten(), mode='lines', name='Train Predictions', ))

# Plot test predictions
fig.add_trace(go.Scatter(y=testPredictPlot.flatten(), mode='lines', name='Test Predictions', line=dict(color='red')))

# Update layout
fig.update_layout(
    title='BC Crash Values Prediction',
    xaxis_title='Time',
    yaxis_title='Value',
    legend_title='Legend',
    xaxis={'rangeslider': {'visible': True}, 'type': 'linear'},
    margin=dict(l=20, r=20, t=40, b=20)
)

# Display the figure
fig.show()

# Output test prices and predictions for additional clarity
print('testPrices:')
if scaler:
    testPrices = scaler.inverse_transform(dataset[test_size + look_back:])
    print(testPrices.flatten())

print('testPredictions:')
print(testPredict.flatten())

# Calculate accuracy for predicting if values are greater than 10
actual = testY.flatten()
predicted = testPredict.flatten()

# Threshold for comparison
threshold_value = 10

# Count correct predictions
correct_predictions = np.sum((actual > threshold_value) == (predicted > threshold_value))
total_predictions = len(actual)

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f'Accuracy for predicting if values are greater than {threshold_value}: {accuracy:.2f}%')