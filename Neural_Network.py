import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Neural Network
def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters:
    - x (numpy array): Input values.

    Returns:
    - numpy array: Output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def initialize_parameters(input_size, hidden1_size, hidden2_size, output_size):
    """
    Initialize the parameters of a neural network.

    Parameters:
    - input_size (int): Number of input features.
    - hidden1_size (int): Number of units in the first hidden layer.
    - hidden2_size (int): Number of units in the second hidden layer.
    - output_size (int): Number of output classes.

    Returns:
    - dict: Dictionary containing the initialized parameters.
    """
    parameters = {
        'W1': {'value': np.random.randn(input_size, hidden1_size), 'change': np.random.randn(input_size, hidden1_size), 'ave_deriv': np.random.randn(input_size, hidden1_size), 'learning_rate': np.random.randn(input_size, hidden1_size)},
        'b1': {'value': np.zeros((1, hidden1_size)), 'change': np.zeros((1, hidden1_size)), 'ave_deriv': np.zeros((1, hidden1_size)), 'learning_rate': np.zeros((1, hidden1_size))},
        'W2': {'value': np.random.randn(hidden1_size, hidden2_size), 'change': np.random.randn(hidden1_size, hidden2_size), 'ave_deriv': np.random.randn(hidden1_size, hidden2_size), 'learning_rate': np.random.randn(hidden1_size, hidden2_size)},
        'b2': {'value': np.zeros((1, hidden2_size)), 'change': np.zeros((1, hidden2_size)), 'ave_deriv': np.zeros((1, hidden2_size)), 'learning_rate': np.zeros((1, hidden2_size))},
        'W3': {'value': np.random.randn(hidden2_size, output_size), 'change': np.random.randn(hidden2_size, output_size), 'ave_deriv': np.random.randn(hidden2_size, output_size), 'learning_rate': np.random.randn(hidden2_size, output_size)},
        'b3': {'value': np.zeros((1, output_size)), 'change': np.zeros((1, output_size)), 'ave_deriv': np.zeros((1, output_size)), 'learning_rate': np.zeros((1, output_size))}
    }
    return parameters

def forward_propagation(X, parameters):
    """
    Perform forward propagation through the neural network.

    Parameters:
    - X (numpy array): Input data.
    - parameters (dict): Dictionary containing the network parameters.

    Returns:
    - Tuple: Activations for each layer (A1, A2, A3).
    """
    Z1 = np.dot(X, parameters['W1']['value']) + parameters['b1']['value']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, parameters['W2']['value']) + parameters['b2']['value']
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, parameters['W3']['value']) + parameters['b3']['value']
    A3 = sigmoid(Z3)
    return A1, A2, A3

def compute_loss(A3, y):
    """
    Compute the cross-entropy loss.

    Parameters:
    - A3 (numpy array): Output of the final layer.
    - y (numpy array): True labels.

    Returns:
    - float: Cross-entropy loss.
    """
    m = len(y)
    log_probs = -np.log(A3[range(m), y])
    loss = np.sum(log_probs) / m
    return loss

def backward_propagation(X, y, parameters, A1, A2, A3):
    """
    Perform backward propagation to compute gradients.

    Parameters:
    - X (numpy array): Input data.
    - y (numpy array): True labels.
    - parameters (dict): Dictionary containing the network parameters.
    - A1, A2, A3 (numpy arrays): Activations from forward propagation.

    Returns:
    - dict: Gradients for each parameter.
    """
    m = len(y)
    
    dZ3 = A3.copy()
    dZ3[range(m), y] -= 1
    dZ3 /= m
    
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    
    dA2 = np.dot(dZ3, parameters['W3']['value'].T)
    dZ2 = dA2 * (A2 * (1 - A2))
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = np.dot(dZ2, parameters['W2']['value'].T)
    dZ1 = dA1 * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
        'dW3': dW3,
        'db3': db3
    }
    
    return gradients


def update_parameters(parameters, gradients, phi, theta, mu, kappa):
    """
    Update the parameters using gradient descent.

    Parameters:
    - parameters (dict): Dictionary containing the network parameters.
    - gradients (dict): Gradients for each parameter.
    - phi (float): Phi constant.
    - theta (float): Theta constant.
    - mu (float): Mu constant.

    Returns:
    - dict: Updated parameters.
    """
    for key in parameters.keys():
        parameters[key]['value'], gradients["d"+key], parameters[key]['ave_deriv'], parameters[key]['learning_rate'], parameters[key]['change'] = change_parameter(
            parameters[key]['value'], gradients["d"+key], parameters[key]['learning_rate'], phi, theta, mu,
            parameters[key]['ave_deriv'], parameters[key]['change'], kappa
        )
    
    return parameters

def change_parameter(weight, gradient, learning_rate, phi, theta, mu, ave_deriv, change_in_weight, kappa):
    """
    A helper function that update the value of each parameter.

    Parameters:
    - weight (numpy array): Weight for for each parameter.
    - gradient (numpy array): Gradients for each parameter.
    - learning_rate (numpy array): Prior learning rates for each parameter.
    - phi (float): Phi constant.
    - theta (float): Theta constant.
    - mu (float): Mu constant.
    - ave_deriv (numpy array): Average of derivative for each parameter.
    - change_in_weight (numpy array): Prior change for each parameter.

    Returns:
    - dict: Updated parameters.
    """
    i, j = weight.shape
    for k in range(0, i):
        for l in range(0, j):
            if gradient[k,l] * ave_deriv[k,l] > 0:
                learning_rate[k,l] += kappa
            else:
                learning_rate[k,l] *= phi
            
            ave_deriv[k,l] = (1 - theta) * gradient[k,l] + theta * ave_deriv[k,l]
            change_in_weight[k,l] = (1 - mu) * -1 * learning_rate[k,l] * gradient[k,l] + mu * change_in_weight[k,l]
            weight[k,l] += change_in_weight[k,l]
    
    return weight, gradient, ave_deriv, learning_rate, change_in_weight

def main():

    # Load data
    df = pd.read_csv('Loan_App.csv')
    X = df.drop('Loan Status', axis=1).values
    y = df['Loan Status'].values

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_size = X_train.shape[1]
    hidden1_size = 40
    hidden2_size = 40
    output_size = 2

    parameters = initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)
    kappa = 0.3
    phi = 0.5
    theta = 0.7
    mu = 0.5
    epochs = 1000
    loss_history = np.zeros((epochs))
    # Training
    print(X_train.shape)
    for epoch in range(epochs):
        A1, A2, A3 = forward_propagation(X_train, parameters)
        loss_history[epoch] = compute_loss(A3, Y_train)
        if epoch % 100 == 0:
            print(f"Epoch number: {epoch} and the loss: {loss_history[epoch]}")
        gradients = backward_propagation(X_train, Y_train, parameters, A1, A2, A3)
        parameters = update_parameters(parameters, gradients, phi, theta, mu, kappa)

    # Prediction
    _,_,A3_test = forward_propagation(X_test, parameters)
    print(A3_test.shape)
    predictions = np.argmax(A3_test, axis=1)

    # Evaluation
    print("Predictions:", predictions)
    print("Actual Values:", Y_test)

    cm = confusion_matrix(Y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    score = accuracy_score(Y_test, predictions)
    print("Accuracy Score:", score)

    print(predictions.shape)
    # Save the model
    np.savez('model.npz', **parameters)

    # # Load the model
    # loaded_data = np.load('model.npz', allow_pickle=True)
    # loaded_parameters = {key: loaded_data[key].item() if key != 'lr' else loaded_data[key] for key in loaded_data.keys()}

    # # Inference with new data
    # lst1 = [0, 0, 1, 4100000, 12200000, 8, 417, 2700000, 2200000, 8800000, 3300000]
    # new_data = np.array([lst1])
    # new_data = scaler.transform(new_data)

    # _, _, A3_new_data = forward_propagation(new_data, loaded_parameters)
    # predicted_class = np.argmax(A3_new_data)
    # print("Inference with new data:", predicted_class)

    visualize_data_distribution(Y_train)
    visualize_correlation_matrix(df)
    visualize_learning_curve(loss_history)
    visualize_confusion_matrix(Y_test, predictions)

def visualize_data_distribution(y):
    """
    Visualize the distribution of loan statuses.

    Parameters:
    - y (numpy array): True labels.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.show()

def visualize_correlation_matrix(df):
    """
    Visualize the correlation matrix of features.

    Parameters:
    - df (pandas DataFrame): Input data.
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def visualize_learning_curve(loss_history):
    """
    Visualize the learning curve during training.

    Parameters:
    - loss_history (list): List of loss values during training.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def visualize_confusion_matrix(y_true, y_pred):
    """
    Visualize the confusion matrix.

    Parameters:
    - y_true (numpy array): True labels.
    - y_pred (numpy array): Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    main()

    