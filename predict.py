import numpy as np
from keras.models import load_model

def predict_sample(model_path, sample):
    model = load_model(model_path)
    prob = model.predict(np.array(sample).reshape(1, -1))
    return prob[0][0]