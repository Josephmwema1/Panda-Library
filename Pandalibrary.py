# ===========================
# Final Assignment: Iris Dataset Analysis
# ===========================

# Step 0: Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ===========================
# Task 1: Load and Explore the Dataset
# ===========================
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("Dataset loaded successfully!\n")
    
except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head(), "\n")

# Explore structure: data types and missing values
print("Dataset info:")
print(df.info(), "\n")
print("Missing values in each column:")
print(df.isnull().sum(), "\n")

# Clean dataset: drop or fill missing values if any
# (Iris dataset has no missing values, but this is a template)
# df.dropna(inplace=True)
# df.fillna(df.mean(), inplace=True)

# ===========================
# Task 2: Basic Data Analysis
# ===========================

# Basic statistics of numerical columns
print("Statistical summary of numerical columns:")
print(df.describe(), "\n")

# Grouping by species and computing mean sepal length
species_group = df.groupby('species')['sepal length (cm)'].mean()
print("Average Sepal Length per Species:")
print(species_group, "\n")

# Additional pattern: mean petal width per species
petal_width_group = df.groupby('species')['petal width (cm)'].mean()
print("Average Petal Width per Species:")
print(petal_width_group, "\n")

# ===========================
# Task 3: Data Visualization
# ===========================

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line Chart: Cumulative Sepal Length per Species
plt.figure(figsize=(8,5))
for sp in df['species'].unique():
    subset = df[df['species'] == sp]
    plt.plot(subset.index, subset['sepal length (cm)'].cumsum(), label=sp)

plt.title('Cumulative Sepal Length by Species')
plt.xlabel('Index')
plt.ylabel('Cumulative Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar Chart: Average Petal Length per Species
avg_petal_length = df.groupby('species')['petal length (cm)'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(data=avg_petal_length, x='species', y='petal length (cm)', palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(6,4))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)',
                hue='species', style='species', s=100)
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

# ===========================
# Optional: Save cleaned dataset
# ===========================
# df.to_csv('iris_cleaned.csv', index=False)
