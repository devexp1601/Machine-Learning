import pickle
import pandas as pd
with open('regression_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)



data = {"highway-mpg" : 10,"city-mpg" : 25,"peak-rpm" : 5000,"horsepower" : 100,"compression-ratio" : 10,"stroke" : 3.5,'bore' : 3.0,"fuel-system" : "mpfi","engine-size" : 130,"num-of-cylinders" : 2,"engine-type" : "ohc","curb-weight" : 2000,"height" : 50,"width" : 70,"length" : 180,"wheel-base" : 100,"engine-location" : "front","drive-wheels" : "fwd","body-style" : "convertible","num-of-doors" : 2,"aspiration" : "std","fuel-type" : "gas","make" : "toyota","normalized-losses" : 150}


test_df =  pd.DataFrame(data, index=[0])

predictions = loaded_pipeline.predict(test_df)

print(predictions)