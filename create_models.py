import numpy as np
import json
import math
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dropout, Dense
from keras.src.models import Sequential
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

# get several train data
def get_train_data():
    # load history datas
    histories = []
    for i in range(0, 10, 1):
        path = f"./history/histories/history{i}.json"
        with open(path, 'r') as file:
            history = json.load(file)
        history = np.array(history)
        print(history[0])
        histories.append(history)
    return histories

if __name__ == "__main__":
    np.random.seed(5)
    # set sequence length of the train data 
    look_back = 200
    # load the train data
    histories = get_train_data() 
    # make the threshold value array
    thresholds = []
    for i in range(0, 5, 1):
        threshold = 30 + i * 10
        thresholds.append(threshold)
    
    data_num = 0
    for history in histories:
        for threshold in thresholds:
            data = np.array(history)
            # handle the special values
            data[data > threshold] = threshold
            dataset = data.reshape(-1, 1)
            # normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            # splite the dataset into the test set and train set
            train_size = int(len(dataset) * 0.8)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            # reshape into X=t and Y=t+1
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)
            # reshape input data to be [samples, time, steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(units=50))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam')

            model.fit(trainX, trainY, epochs=1000, batch_size=100, verbose=1, validation_data=(testX, testY))
            model_id = f"LSTM_100k_{threshold}_0.h5"
            model.save(model_id)
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

            # Test model case 
            result = []
            for threshold_value in range(2, threshold):
                print(f"=======Performance of the prediction for greater than: {threshold_value} ========")
                correct_predictions = np.sum((actual >= threshold_value) & (predicted >= threshold_value))
                wrong_predictions = np.sum((actual >= threshold_value) & (predicted < threshold_value))
                total_greater_predictions = np.sum(predicted >= threshold_value)
                total_greater_chance = np.sum(actual >= threshold_value)
                print(f"Correct predictoin for greater than {threshold_value}: ", correct_predictions)
                print(f"Wrong prediction for less than {threshold_value}: ", wrong_predictions)
                print(f"Total prediction for greater than {threshold_value}: ", total_greater_predictions)
                # Calculate accuracy
                accuracy = (correct_predictions / total_greater_predictions) * 100
                print(f'Accuracy for predicting if values are greater than {threshold_value}: {accuracy:.2f}%')
                profit = correct_predictions * threshold_value - total_greater_predictions
                print(f"Profit from beting with {profit}")

                result_threshold = {
                        "Index" : threshold_value,
                        "Correct predictoin for greater than " + str(threshold_value) + ": " : str(correct_predictions),
                        "Wrong prediction for less than " + str(threshold_value) + ": " : str(wrong_predictions),
                        "Total prediction for greater than " + str(threshold_value) + ": " : str(total_greater_predictions),
                        "Accuracy for predicting if values are greater than " + str(threshold_value) + ": " : str(accuracy),
                        "Profit from beting on " + str(threshold_value) + ": " : str(profit)
                }

                result.append(result_threshold)

            path = f"result_{threshold}_{data_num}.json"
            with open(path, "w") as file:
                json.dump(result, file, indent=2)
            data_num += 1
