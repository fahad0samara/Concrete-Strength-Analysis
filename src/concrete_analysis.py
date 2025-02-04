import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Read the dataset
df = pd.read_csv('concrete.csv')

# Basic statistical analysis
print("\n=== Basic Dataset Information ===")
print(df.info())
print("\n=== Statistical Summary ===")
print(df.describe())

# Correlation analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Concrete Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Distribution of target variable (strength)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='strength', bins=30, kde=True)
plt.title('Distribution of Concrete Compressive Strength')
plt.xlabel('Strength (MPa)')
plt.ylabel('Count')
plt.savefig('strength_distribution.png')
plt.close()

# Relationship between age and strength
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='strength')
plt.title('Age vs Concrete Strength')
plt.xlabel('Age (days)')
plt.ylabel('Strength (MPa)')
plt.savefig('age_vs_strength.png')
plt.close()

# Feature importance based on correlation with strength
correlations = df.corr()['strength'].sort_values(ascending=False)
print("\n=== Feature Correlations with Strength ===")
print(correlations)

# Box plots for key features
features = ['cement', 'water', 'age']
plt.figure(figsize=(15, 5))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=df, y=feature)
    plt.title(f'{feature.capitalize()} Distribution')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Calculate some interesting statistics
print("\n=== Interesting Statistics ===")
print(f"Average strength: {df['strength'].mean():.2f} MPa")
print(f"Maximum strength: {df['strength'].max():.2f} MPa")
print(f"Minimum strength: {df['strength'].min():.2f} MPa")
print(f"Most common concrete age: {df['age'].mode().values[0]} days")
print(f"Average cement content: {df['cement'].mean():.2f} kg/m³")
print(f"Water-to-cement ratio statistics:")
df['water_cement_ratio'] = df['water'] / df['cement']
print(df['water_cement_ratio'].describe())
