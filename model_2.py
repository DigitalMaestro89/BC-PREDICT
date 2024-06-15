import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import ADASYN

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Fix random seed for reproducibility
np.random.seed(5)

print('Loading JSON...')

# Load and prepare the data
with open('./crash_history.json', 'r') as file:
    data = json.load(file)
data = np.array(data)

print('Loading Completed!')

# Convert data to numpy array
data = np.array(data)
dataset = data.reshape(-1, 1)

print('Transforming Dataset...')

# Normalize the dataset
scaler = QuantileTransformer(output_distribution='normal')
dataset = scaler.fit_transform(dataset)

print('Transformation Completed!')

# Split into train and test sets, 80% training data, 20% test data
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size,:], dataset[train_size:,:]

print('Generating Sequence Dataset...')

# Reshape into X=t and Y=t+1, timestep 15
look_back = 15
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print('Generating Sequence Dataset Completed!')

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1]))

# Threshold for comparison
threshold_value = 10

# Convert the original (unscaled) trainY to binary labels for SMOTE
original_data = data.reshape(-1, 1)
train_originalY = original_data[:train_size][look_back + 1: len(trainX) + look_back + 1].flatten()
trainY_binary = (train_originalY > threshold_value).astype(int)

print('Resampling...')

# Apply ADASYN
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
trainX_resampled, trainY_resampled = adasyn.fit_resample(trainX, trainY_binary)

print('Resampling Completed!...')

# Reshape resampled data back to [samples, time steps, features]
trainX_resampled = np.reshape(trainX_resampled, (trainX_resampled.shape[0], 1, trainX_resampled.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=50))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Change activation to sigmoid for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Change loss to binary_crossentropy

# Train the model
model.fit(trainX_resampled, trainY_resampled, epochs=220, batch_size=700, verbose=1)

# Make predictions
trainPredict = model.predict(trainX_resampled)
testPredict = model.predict(np.reshape(testX, (testX.shape[0], 1, testX.shape[1])))

testY = testY.reshape(-1, 1)

# Invert predictions
testY = scaler.inverse_transform(testY)

limit_value = 0.7

# Calculate accuracy for predicting if values are greater than 10
actual = (testY.flatten() >= threshold_value).astype(int)
predicted = testPredict.flatten()

print(f'Actual Length: {len(actual)}, Prediction Length: {len(predicted)}')

# Count correct predictions
correct_predictions = np.sum((actual == 1) & (predicted > limit_value))
total_predictions = np.sum(actual == 1)

print(f'Correct Predictions: {correct_predictions}')
print(f'Total Predictions: {total_predictions}')

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f'Accuracy for predicting if values are greater than {threshold_value}: {accuracy:.2f}%')
