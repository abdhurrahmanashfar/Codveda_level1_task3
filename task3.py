

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("D:/Codveda Technologies/Projects/CSV/drive/iris.csv")

print("Summary Statistics:")
print(df.describe())

print("\nMedian of each column:")
print(df.median(numeric_only=True))

print("\nVariance of each column:")
print(df.var(numeric_only=True))

df.hist(figsize=(10, 6), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

sns.pairplot(df, hue='species')  
plt.suptitle("Pairwise Scatter Plots", y=1.02)
plt.show()

plt.figure(figsize=(12, 6))
for idx, col in enumerate(df.select_dtypes(include='number').columns):
    plt.subplot(1, 4, idx + 1)
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


print("\n Report Summary:")
print("- Sepal length and petal length have a strong positive correlation.")
print("- Petal width varies significantly between species (seen in boxplot).")
print("- There are no missing values in the dataset.")
print("- Histograms show petal length and width are skewed, sepal features are more normally distributed.")
