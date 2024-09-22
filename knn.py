import numpy as np
from collections import Counter
import csv

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calcula a distância entre x e todos os pontos de treino
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Obtem os k vizinhos mais próximos
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Retorna a classe mais comum entre os vizinhos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(y_true)
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_index[true], label_to_index[pred]] += 1
    
    return matrix

def precision_recall_f1(conf_matrix):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP

    # Precisão
    precision = np.mean(np.divide(TP, (TP + FP), where=(TP + FP) != 0))
    # Recall
    recall = np.mean(np.divide(TP, (TP + FN), where=(TP + FN) != 0))
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f1

def load_iris_dataset(filename='Iris.csv'):
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pular o cabeçalho
        for row in reader:
            X.append([float(val) for val in row[:-1]])
            y.append(row[-1])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Carregar dataset Iris
    X, y = load_iris_dataset()

    # Embaralhar o dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Dividir em treino e teste (80% treino, 20% teste)
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Instanciar o modelo
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Fazer previsões
    y_pred = knn.predict(X_test)

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(conf_matrix)

    # Calcular acurácia
    acc = accuracy(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")

    # Calcular precisão, recall e F1-score
    precision, recall, f1 = precision_recall_f1(conf_matrix)
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")