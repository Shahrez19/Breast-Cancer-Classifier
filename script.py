#Programmer::= Shahrez Jan
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Steps 1-7
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
# First data point is tagged as malignant


training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,breast_cancer_data.target,train_size=0.8,random_state=100)
print(len(training_data))
print(len(training_labels))


# Steps 8-11
classifer = KNeighborsClassifier(n_neighbors=3)
classifer.fit(training_data,training_labels)

print(classifer.score(validation_data,validation_labels))

# Steps 12-18
accuracies = []
for k in range(1,101):
    classifer = KNeighborsClassifier(n_neighbors=k)
    classifer.fit(training_data,training_labels)
    accuracies.append(classifer.score(validation_data,validation_labels))
max_value = max(accuracies)
max_index = accuracies.index(max_value)
print(max_value,max_index)
# k = 22
# Gives the most accuracy in the validation data set


k_list = list(range(1,101))

plt.plot(k_list,accuracies)
plt.xlabel(k)
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()