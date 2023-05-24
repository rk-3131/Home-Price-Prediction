import json
import pickle
import numpy as np
import warnings


__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("Loading saved artifacts")
    global __data_columns
    global __locations

    with open ("D:/Home Price Prediction/Code/BHP/server/artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open ("D:/Home Price Prediction/Code/BHP/server/artifacts/House_Price_Prediction.pickle", 'rb') as f:
            __model = pickle.load(f)
    print("Loading artifacts is done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if  __name__ == '__main__':
    # warnings.filterwarnings("ignore", category=UserWarning)
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st block jayanagar', 1000, 3, 3))
    print(get_estimated_price('aecs layout', 1000, 2, 3))