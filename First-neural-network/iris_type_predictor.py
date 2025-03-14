import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
from perceptron import Perceptron

iris_dataset = datasets.load_iris()

"""
Plotting visualization of iris_dataset
"""
_, ax = plt.subplots()
scatter = ax.scatter(iris_dataset.data[:, 0], iris_dataset.data[:, 1], c=iris_dataset.target)
ax.set(xlabel=iris_dataset.feature_names[0], ylabel=iris_dataset.feature_names[1])
_ = ax.legend(scatter.legend_elements()[0], iris_dataset.target_names, loc="lower right", title="Classes")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size = 0.2, random_state = 5)

perceptron = Perceptron()

perceptron.train(X_train, y_train, n_iter = 400, learning_rate = 0.03)
y_predicted = perceptron.predict(X_test)
y_predicted = [math.ceil(y) for y in y_predicted] # Approximating all predicted y's
accuracy = accuracy_score(y_test, y_predicted)

print(f'Accuracy of our perceptron: {round(accuracy, 4) * 100}%')

"""
Plotting number of appearances
"""
count_test = Counter(y_test)
count_predicted = Counter(y_predicted)
values = [0, 1, 2]
test_counts = [count_test.get(value, 0) for value in values]
predicted_counts = [count_predicted.get(value, 0) for value in values]
bar_width = 0.35
index = range(len(values))
plt.figure(figsize=(8, 6), dpi=300)
plt.bar(index, test_counts, bar_width, label='Expected values (y_test)', color='b')
plt.bar([i + bar_width for i in index], predicted_counts, bar_width, label='Predicted values (y_predicted)', color='r')
plt.xlabel('Values')
plt.ylabel('Number of appearances')
plt.title('Comparison of appearances values 0, 1, 2')
plt.xticks([i + bar_width / 2 for i in index], values)
plt.legend()
plt.show()