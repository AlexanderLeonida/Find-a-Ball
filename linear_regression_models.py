from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_position_model(df):
    if df.empty:
        print("No data to train")
        return None
    
    # time per change in coordinates
    df["time_step"] = np.arange(len(df))
    # change in x
    # fill NA coordinates w/a 0
    df["x_velocity"] = df["x_coordinate"].diff().fillna(0)
    # change in y
    df["y_velocity"] = df["y_coordinate"].diff().fillna(0)
 
    X = df[["x_coordinate", "y_coordinate", "time_step"]]
    # predict new postiion
    y = df[["x_coordinate", "y_coordinate"]]
 
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # entire method is pretty much this link
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_velocity_model(df):
    if df.empty:
        print("No data to train")
        return None
 
    # time per change in coordinates
    df["time_step"] = np.arange(len(df))
    # make sure there are no nulls in data, replace with 0s
    df["x_velocity"] = df["x_coordinate"].diff().fillna(0)
    df["y_velocity"] = df["y_coordinate"].diff().fillna(0)
 
 
    X = df[["x_coordinate", "y_coordinate", "time_step"]]
    y_x = df["x_velocity"]
    y_y = df["y_velocity"]
 
 
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # entire method is pretty much this link but doubled
    velocity_x = LinearRegression()
    velocity_y = LinearRegression()
    velocity_x.fit(X, y_x)
    velocity_y.fit(X, y_y)
 
 
    return velocity_x, velocity_y