import numpy as np
import pandas as pd

# Carregar o dataset
penguins_df = pd.read_csv('penguins_lter.csv')

# Limpar o dataset e selecionar colunas relevantes
penguins_cleaned_df = penguins_df[['Species', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']].dropna()

# Codificar as espécies em valores numéricos
penguins_cleaned_df['Species'] = penguins_cleaned_df['Species'].astype('category').cat.codes

# Função para embaralhar os dados
def shuffle_data(X, y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return X[indices], y[indices]

# Divisão dos dados em treino e teste sem usar sklearn
def train_test_split(X, y, test_size=0.3, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Preparar os dados
X = penguins_cleaned_df[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']].values
y = penguins_cleaned_df['Species'].values

# Embaralhar os dados
X, y = shuffle_data(X, y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementação do classificador Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Inicializar médias, variâncias e priors para cada classe
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c, :] = X_c.mean(axis=0)
            self.var[c, :] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / float(n_samples)

    def _gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._gaussian_pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

# Funções para calcular as métricas
def confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(y_true)
    cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

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

# Função para calcular a média ponderada
def weighted_average(metrics, y_true):
    classes, counts = np.unique(y_true, return_counts=True)
    total = counts.sum()
    weighted_metrics = np.sum(metrics * counts[:, None], axis=0) / total
    return weighted_metrics

# Treinar o modelo
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Fazer previsões
y_pred = nb.predict(X_test)

# Calcular as métricas
cm = confusion_matrix(y_test, y_pred)
acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calcular médias ponderadas
prec_avg = np.mean(prec)
rec_avg = np.mean(rec)
f1_avg = np.mean(f1)

# Exibir os resultados
print("Matriz de Confusão:")
print(cm)
print(f"Acurácia: {acc * 100:.2f}%")
print(f"Precisão (média ponderada): {prec_avg:.4f}")
print(f"Recall (média ponderada): {rec_avg:.4f}")
print(f"F1 Score (média ponderada): {f1_avg:.4f}")
