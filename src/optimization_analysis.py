import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read data
df = pd.read_csv('concrete.csv')

# 1. Strength Classification and Guidelines
def analyze_strength_classes():
    # Define strength classes
    df['strength_class'] = pd.cut(df['strength'], 
                                bins=[0, 20, 30, 40, 50, float('inf')],
                                labels=['Low', 'Moderate', 'High', 'Very High', 'Ultra High'])
    
    # Calculate average mixture proportions for each strength class
    class_proportions = df.groupby('strength_class').agg({
        'cement': 'mean',
        'slag': 'mean',
        'ash': 'mean',
        'water': 'mean',
        'superplastic': 'mean'
    }).round(2)
    
    # Add water-cement ratio
    class_proportions['water_cement_ratio'] = (class_proportions['water'] / class_proportions['cement']).round(3)
    
    # Visualization of strength classes
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='strength_class', y='strength')
    plt.title('Concrete Strength Classes Distribution')
    plt.xlabel('Strength Class')
    plt.ylabel('Strength (MPa)')
    plt.savefig('strength_classes.png')
    plt.close()
    
    return class_proportions

# 2. Ingredient Interaction Analysis
def analyze_interactions():
    interactions = []
    
    # Analyze cement-water interaction
    plt.figure(figsize=(10, 6))
    plt.scatter(df['cement'], df['water'], c=df['strength'], cmap='viridis')
    plt.colorbar(label='Strength (MPa)')
    plt.xlabel('Cement Content (kg/m³)')
    plt.ylabel('Water Content (kg/m³)')
    plt.title('Cement-Water Interaction Effect on Strength')
    plt.savefig('cement_water_interaction.png')
    plt.close()
    
    # Analyze superplasticizer-water interaction
    plt.figure(figsize=(10, 6))
    plt.scatter(df['superplastic'], df['water'], c=df['strength'], cmap='viridis')
    plt.colorbar(label='Strength (MPa)')
    plt.xlabel('Superplasticizer Content (kg/m³)')
    plt.ylabel('Water Content (kg/m³)')
    plt.title('Superplasticizer-Water Interaction Effect on Strength')
    plt.savefig('superplastic_water_interaction.png')
    plt.close()
    
    return interactions

# 3. Cost-Effectiveness Analysis
def analyze_cost_effectiveness():
    # Approximate costs in USD per kg (these are example prices)
    costs = {
        'cement': 0.10,
        'slag': 0.05,
        'ash': 0.03,
        'water': 0.001,
        'superplastic': 2.0
    }
    
    # Calculate material cost per m³
    for material, cost in costs.items():
        df[f'{material}_cost'] = df[material] * cost
    
    df['total_cost'] = sum(df[f'{material}_cost'] for material in costs.keys())
    df['cost_effectiveness'] = df['strength'] / df['total_cost']
    
    # Find most cost-effective mixtures
    cost_effective_mixtures = df.nlargest(10, 'cost_effectiveness')
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_cost'], df['strength'], alpha=0.5)
    plt.xlabel('Total Material Cost (USD/m³)')
    plt.ylabel('Strength (MPa)')
    plt.title('Cost vs Strength Relationship')
    plt.savefig('cost_strength_relationship.png')
    plt.close()
    
    return cost_effective_mixtures[['strength', 'total_cost', 'cost_effectiveness'] + 
                                 list(costs.keys())].round(2)

# 4. Optimization Guidelines for Different Ages
def create_age_based_guidelines():
    age_groups = [1, 7, 28, 90, 365]
    df['age_group'] = pd.cut(df['age'], bins=age_groups, labels=['1-7', '7-28', '28-90', '90+'])
    
    # Calculate optimal mixtures for each age group
    age_guidelines = df.groupby('age_group').apply(
        lambda x: x.nlargest(5, 'strength')[
            ['strength', 'cement', 'slag', 'ash', 'water', 'superplastic']
        ].mean()
    ).round(2)
    
    return age_guidelines

# Run analyses
print("=== Strength Class Analysis ===")
class_proportions = analyze_strength_classes()
print("\nTypical mixture proportions for each strength class:")
print(class_proportions)

print("\n=== Cost-Effectiveness Analysis ===")
cost_effective_mixtures = analyze_cost_effectiveness()
print("\nTop 10 most cost-effective mixtures:")
print(cost_effective_mixtures)

print("\n=== Age-Based Optimization Guidelines ===")
age_guidelines = create_age_based_guidelines()
print("\nOptimal mixtures for different ages:")
print(age_guidelines)

# 5. Additional Insights
print("\n=== Additional Insights ===")
# Calculate efficiency metrics
df['cement_efficiency'] = df['strength'] / df['cement']
top_efficient = df.nlargest(5, 'cement_efficiency')
print("\nMost efficient cement utilization (Strength/Cement ratio):")
print(top_efficient[['strength', 'cement', 'cement_efficiency', 'water', 'superplastic', 'age']].round(2))

# Calculate supplementary material effectiveness
df['supplementary_ratio'] = (df['slag'] + df['ash']) / (df['cement'] + df['slag'] + df['ash'])
high_strength_sustainable = df[df['strength'] > df['strength'].mean()]
print("\nAverage supplementary material ratio for high-strength concrete:")
print(f"{high_strength_sustainable['supplementary_ratio'].mean():.2%}")

# Optimal water-cement ratios for different strength levels
print("\nOptimal water-cement ratios by strength class:")
df['w_c_ratio'] = df['water'] / df['cement']
w_c_ratios = df.groupby('strength_class')['w_c_ratio'].agg(['mean', 'std']).round(3)
print(w_c_ratios)
