Step 0: Installation of Libraries
pip install pandas matplotlib seaborn scikit-learn

Step 1: Load and Explore the Dataset
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows
print(df.head())

# Check dataset info (data types, missing values)
print(df.info())
print(df.isnull().sum())

# Clean data (drop or fill missing values if any)
# In this dataset, there are no missing values, but here's how you could handle them:
# df.dropna(inplace=True)        # Drop missing rows
# df.fillna(df.mean(), inplace=True)  # Fill numerical missing values with mean


Explanation:

We created a pandas DataFrame from the Iris dataset.

.head() lets you inspect the first few rows.

.info() shows data types.

.isnull().sum() checks for missing values.

Step 2: Basic Data Analysis
# Compute basic statistics
print(df.describe())

# Grouping by species and computing mean sepal length
species_group = df.groupby('species')['sepal length (cm)'].mean()
print(species_group)

# Patterns / Insights
# For example, setosa has the smallest sepal length, virginica has the largest


Explanation:

.describe() gives mean, median (50% percentile), std, min, max, etc.

.groupby() allows aggregation based on categorical columns.

Step 3: Data Visualization
1. Line Chart (for illustration, using cumulative sum of sepal length)
plt.figure(figsize=(8,5))
for sp in df['species'].unique():
    subset = df[df['species'] == sp]
    plt.plot(subset.index, subset['sepal length (cm)'].cumsum(), label=sp)

plt.title('Cumulative Sepal Length by Species')
plt.xlabel('Index')
plt.ylabel('Cumulative Sepal Length (cm)')
plt.legend()
plt.show()

2. Bar Chart (Average petal length per species)
avg_petal_length = df.groupby('species')['petal length (cm)'].mean().reset_index()

plt.figure(figsize=(6,4))
sns.barplot(data=avg_petal_length, x='species', y='petal length (cm)', palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

3. Histogram (Distribution of sepal width)
plt.figure(figsize=(6,4))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

4. Scatter Plot (Sepal length vs. Petal length)
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', s=100)
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.show()

Step 4: Exception Handling for File Loading
try:
    df_custom = pd.read_csv('your_dataset.csv')
except FileNotFoundError:
    print("Error: CSV file not found.")
except pd.errors.ParserError:
    print("Error: Could not parse CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

✅ Summary

Dataset loaded using pandas, inspected, and cleaned.

Basic statistics calculated with .describe().

Grouped by species to compute averages.

Four plots created with proper titles, labels, and legends:

Line chart

Bar chart

Histogram

Scatter plot

Exception handling added for robust CSV loading.
