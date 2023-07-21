# Q-AID
Quantum AI for Iris classification

## Quantum Machine Learning for Iris Data Classification

Introduction

This script aims to classify the Iris dataset using a hybrid quantum-classical machine learning model known as a Variational Quantum Classifier (VQC). We leverage the power of quantum computers to process and identify patterns in the dataset that may not be easily discernable with classical computing methods.

The main use case of this script is to serve as a starting point for the application of Quantum Machine Learning in the classification of data, particularly in scenarios where the data may contain complex patterns or relationships that can potentially be better recognized by Quantum Computing techniques.

Although the script uses the Iris dataset, it can be adapted for other classification tasks in different domains such as image recognition, natural language processing, medical diagnosis, etc., given that the data can be encoded into the quantum device.

## Dependencies

To run the script, the following Python libraries are required:

	•	PennyLane: A cross-platform Python library for quantum machine learning.
	•	scikit-learn: A free software machine learning library for the Python programming language.
	•	NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## Script Explanation

The script follows these steps:

	1.	Data Preprocessing: We load the Iris dataset using the scikit-learn load_iris function. We then standardize the features to have zero mean and unit variance using the StandardScaler. The dataset is then split into a training set and a test set using train_test_split function.
	2.	Quantum Device Setup: We define a quantum device with two qubits using PennyLane’s default.mixed device. This simulates a noise model with depolarizing noise, helping us simulate the behavior of real quantum hardware.
	3.	Quantum Node (QNode) Definition: A QNode represents a quantum function. In our script, this quantum function is used to encode the data inputs into the quantum device and then apply the StronglyEntanglingLayers. The expectation values of the Pauli-Z operator on each qubit are returned.
	4.	Model Training: We initialize the weights of the StronglyEntanglingLayers and optimize them using the Adam optimizer over a set number of steps. In each step, the optimizer minimizes the cost function, which is defined as the negative mean of the QNode outputs over the training data.
	5.	Prediction and Evaluation: With the optimized weights, we can now make predictions on the test set. The sign of the output from the quantum node is taken as the prediction, which is then mapped to class labels. The accuracy of the model on the test set is calculated and printed.

## Running the Script

After installing the necessary libraries, the script can be run from a Python environment. Note that due to the stochastic nature of the training process, the results may slightly vary each time the script is run.

## Limitations and Further Improvements

While quantum machine learning presents intriguing possibilities, it’s important to keep in mind that the field is still in its early days. Quantum hardware is continually improving, but current noisy intermediate-scale quantum (NISQ) devices may not provide a performance advantage over classical computers for many tasks. In the script, we have simulated a noise model to mimic a real quantum device, which might not necessarily lead to better performance.

Possible improvements include tuning hyperparameters, increasing the number of layers or qubits, applying error correction techniques, or trying different types of quantum-classical hybrid models. It’s also beneficial to stay updated with the latest quantum machine learning research, as the field is rapidly advancing.

## Conclusion

This script presents a simple example of applying Quantum Machine Learning to a classical machine learning task. It demonstrates how to build and train a quantum-classical hybrid model using PennyLane and how to make predictions with the trained model. As the field of quantum computing and quantum machine learning continues to grow, such hybrid models may become an essential tool for tackling complex problems.

## Credits

Development & research by: 

Adam Rivers - https://abtzpro.github.io
