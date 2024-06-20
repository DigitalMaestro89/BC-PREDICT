import numpy as np
import json
import math

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == "__main__":
    np.random.seed(5)
    # load history datas
    histories = []
    for i in range(0, 10, 1):
        path = f"./history/histories/history{i}.json"
        with open(path, 'r') as file:
            history = json.load(file)
        history = np.array(history)
        print(history[0])
        histories.append(history)
    