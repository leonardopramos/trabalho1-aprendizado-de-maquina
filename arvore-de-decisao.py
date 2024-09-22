import numpy as np
import pandas as pd
from collections import Counter

# Função para carregar o dataset Iris
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features (4 primeiras colunas)
    y = data.iloc[:, -1].values   # Classe (última coluna)
    return X, y

# Função para calcular a entropia
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Função para dividir o dataset com base em um valor de feature
def split_dataset(X, y, feature_index, threshold):
    left_idx = np.where(X[:, feature_index] <= threshold)
    right_idx = np.where(X[:, feature_index] > threshold)
    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

# Função para calcular o ganho de informação
def information_gain(X, y, feature_index, threshold):
    parent_entropy = entropy(y)
    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)

    if len(y_left) == 0 or len(y_right) == 0:
        return 0

    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    weighted_avg_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    
    return parent_entropy - weighted_avg_entropy

# Nó da árvore de decisão
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Implementação da árvore de decisão
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Condições de parada
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        # Encontrar o melhor split
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = threshold

        if best_gain == 0:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        # Dividir o dataset
        X_left, X_right, y_left, y_right = split_dataset(X, y, split_idx, split_threshold)

        # Crescer as sub-árvores
        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return DecisionNode(split_idx, split_threshold, left_child, right_child)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Funções para calcular as métricas
def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return matrix

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis=0)

def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.diag(cm) / np.sum(cm, axis=1)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)

# Carregar o dataset
X, y = load_data('Iris.csv')

# Codificar as classes para inteiros
class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
y = np.array([class_mapping[label] for label in y])

# Embaralhar os dados
np.random.seed(42)
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]

# Dividir em treino e teste (80% treino, 20% teste)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Treinar a árvore de decisão
tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)

# Fazer previsões
y_pred = tree.predict(X_test)

# Calcular métricas
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nAcurácia:", accuracy(y_test, y_pred))
print("Recall:", recall(y_test, y_pred))
print("Precisão:", precision(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))