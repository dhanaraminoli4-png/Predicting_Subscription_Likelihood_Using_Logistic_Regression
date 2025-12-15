import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# 1. Load and inspect data
data = pd.read_csv("logistic_regression_project.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data["Subscribed"].value_counts())

# 2. Handle missing values and duplicates
data = data.drop_duplicates()

num_features = ["Age", "Hours_Studied_Per_Week", "Completed_Free_Courses", "Visited_Forum"]
imputer = SimpleImputer(strategy="mean")
data[num_features] = imputer.fit_transform(data[num_features])

# Normalize numerical features
scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

# 3. Apply PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data[num_features])
data[["PC1", "PC2"]] = pca_components

# 4. Exploratory plots (optional)
num_col = ["PC1", "PC2"]
for col in num_col:
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
    
    sns.boxplot(x='Subscribed', y=col, data=data)
    plt.show()

sns.heatmap(data[num_col + ["Subscribed"]].corr(), annot=True)
plt.show()

# 5. Function to sample train/test
def sample_train_test(data, samp_frac=0.6, train_frac=0.8):
    sample_data = data.sample(frac=samp_frac, random_state=None)
    train = sample_data.sample(frac=train_frac, random_state=None)
    test = sample_data.drop(train.index)
    
    x_train = train[["PC1", "PC2"]]
    y_train = train["Subscribed"]
    x_test = test[["PC1", "PC2"]]
    y_test = test["Subscribed"]
    
    return x_train, y_train, x_test, y_test

# 6. Train logistic regression and calculate accuracy
rounds = 3
accuracies = []

for i in range(rounds):
    x_train, y_train, x_test, y_test = sample_train_test(data)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("Average accuracy score:", round(sum(accuracies)/len(accuracies), 4))

# 7. Visualize Logistic Regression Decision Boundary
x_min, x_max = x_train["PC1"].min() - 1, x_train["PC1"].max() + 1
y_min, y_max = x_train["PC2"].min() - 1, x_train["PC2"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(x_train["PC1"], x_train["PC2"], c=y_train, edgecolor='k', cmap=plt.cm.RdYlBu)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Logistic Regression Decision Boundary (PCA)')
plt.show()
