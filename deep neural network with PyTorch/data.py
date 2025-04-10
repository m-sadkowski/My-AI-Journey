from sklearn.datasets import load_iris
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

features_names = iris.feature_names # nazwy parametrów wejściowych (na wejściu mamy 4 liczby, wymiary płatków irysów)
# print(features_names)

iris_types = iris.target_names # nazwy parametrów wyjściowych (gatunki irysów)
# print(iris_types)

df_X = DataFrame(X)
df_X_labeled = df_X.set_axis([features_names[0], features_names[1], features_names[2], features_names[3]], axis=1)

df_y = DataFrame(iris_types[y])

df_data_pairs = df_X_labeled
df_data_pairs['iris type'] = df_y
print(df_data_pairs)

# Plot before standarization
sns.catplot(data=df_X).set_xticklabels(rotation=30, labels=features_names).set_ylabels('[cm]')
plt.show()

# One hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Standarization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

