import numpy as np
import pandas as pd

# Carregar o arquivo CSV
data = pd.read_csv('penguins_lter.csv')

# Limpar dados removendo entradas com valores ausentes e mapear a coluna Species para números
data_cleaned = data[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Species']].dropna()

# Converter espécies para valores numéricos
species_mapping = {species: idx for idx, species in enumerate(data_cleaned['Species'].unique())}
data_cleaned['Species'] = data_cleaned['Species'].map(species_mapping)

# Separar features e alvo
X = data_cleaned[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']].values
y = data_cleaned['Species'].values

# Função para dividir os dados em treino e teste aleatoriamente
def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    
    return X_train, X_test, y_train, y_test

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return {"leaf": True, "value": leaf_value}

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return {"leaf": True, "value": leaf_value}

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_subtree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X[:, feature], threshold)

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                gini = self._gini_index(y, left_idxs, right_idxs)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split(self, X_column, threshold):
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]
        return left_idxs, right_idxs

    def _gini_index(self, y, left_idxs, right_idxs):
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        gini_left, gini_right = 0, 0

        if n_left > 0:
            gini_left = self._gini(y[left_idxs])

        if n_right > 0:
            gini_right = self._gini(y[right_idxs])

        return (n_left / n) * gini_left + (n_right / n) * gini_right

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node["leaf"]:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])

# Funções para calcular métricas de desempenho

def confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
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
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r)

# Instanciar e treinar o modelo
tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = tree.predict(X_test)

# Calcular e exibir as métricas
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("Acurácia:", accuracy(y_test, y_pred))
print("Precisão:", precision(y_test, y_pred))
print("Recall:", recall(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
