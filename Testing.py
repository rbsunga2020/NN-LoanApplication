import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from Neural_Network import forward_propagation, visualize_confusion_matrix, visualize_data_distribution

# Load saved data of model
scaler = StandardScaler()
loaded_data = np.load('final_model.npz', allow_pickle=True)
loaded_parameters = {key: loaded_data[key].item() if key != 'lr' else loaded_data[key] for key in loaded_data.keys()}

# Inference with new data

# Load validation data from CSV
validation_data = pd.read_csv('Validation.csv')
X = validation_data.drop(' loan_status', axis=1).values
Y = validation_data[' loan_status'].values

# Assuming the columns in your CSV file are in the same order as your original input features
validation_input = scaler.fit_transform(X)

# Inference with validation data
_, _, A3_validation = forward_propagation(validation_input, loaded_parameters)
predicted_classes = np.argmax(A3_validation, axis=1)

score = accuracy_score(Y, predicted_classes)
print("Accuracy Score:", score)
print(Y.shape)
visualize_data_distribution(Y)
visualize_confusion_matrix(Y, predicted_classes)

