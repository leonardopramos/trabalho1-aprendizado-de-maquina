import numpy as np
import pandas as pd
from collections import defaultdict

class NaiveBayes:
    def fit(self, X, y):
        # Separar por classe
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._gaussian(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

# Funções para cálculo de métricas
def confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(y_true)
    matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for i, true_label in enumerate(y_true):
        matrix[true_label][y_pred[i]] += 1
    return matrix

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    precisions = []
    for i in range(len(conf_mat)):
        precisions.append(conf_mat[i, i] / np.sum(conf_mat[:, i]))
    return np.mean(precisions)

def recall(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    recalls = []
    for i in range(len(conf_mat)):
        recalls.append(conf_mat[i, i] / np.sum(conf_mat[i, :]))
    return np.mean(recalls)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)

# Carregar o dataset Iris
def load_iris_dataset():
    # Carregue o arquivo Iris.csv da pasta raiz
    df = pd.read_csv('Iris.csv')
    # Remover a coluna 'Id' se existir
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    # Mapear classes para números
    df['Species'] = df['Species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    # Separar features e labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

# Dividir o dataset em treino e teste
def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]
    return X_train, X_test, y_train, y_test

# Função principal
if __name__ == '__main__':
    # Carregar dataset
    X, y = load_iris_dataset()

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Treinar o modelo
    model = NaiveBayes()
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Calcular métricas
    conf_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Exibir resultados
    print("Matriz de Confusão:")
    print(conf_mat)
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")