import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train = X_train[:, 0:2]
X_test = X_test[:, 0:2]

n_qubits = 2
dev = qml.device("default.mixed", wires=n_qubits) # using a noisy simulator

@qml.qnode(dev)
def qnode(inputs, weights):
    AngleEmbedding(inputs, wires=range(n_qubits))
    StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

opt = qml.AdamOptimizer(0.01)
params = qml.init.strong_ent_layers_normal(n_layers=n_layers, n_wires=n_qubits)

for i in range(60):
    params, prev_cost = opt.step_and_cost(lambda params: -np.mean([qnode(x, params) for x in X_train]), params)
    if i % 10 == 0:
        print(f"Cost at step {i}:", prev_cost)

predictions = np.sign(np.array([qnode(x, params) for x in X_test]).flatten())

y_predictions = []
for val in predictions:
    if val == -1:
        y_predictions.append(0)
    elif val == 1:
        y_predictions.append(1)
    else:
        y_predictions.append(2)

accuracy = (y_predictions == y_test).sum() / y_test.size
print(f"Accuracy on test set: {accuracy * 100}%")
