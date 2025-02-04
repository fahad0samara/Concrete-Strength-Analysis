import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Read and prepare the data
df = pd.read_csv('concrete.csv')
X = df.drop('strength', axis=1)
y = df['strength']

# 1. Advanced Feature Analysis
def analyze_feature_importance():
    # Prepare data for modeling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Concrete Strength Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance

# 2. Age-based Strength Development Analysis
def analyze_strength_development():
    age_groups = [1, 7, 14, 28, 56, 90, 180, 365]
    df['age_group'] = pd.cut(df['age'], bins=age_groups, labels=age_groups[:-1])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='age_group', y='strength')
    plt.title('Concrete Strength Development Over Time')
    plt.xlabel('Age (days)')
    plt.ylabel('Strength (MPa)')
    plt.savefig('strength_development.png')
    plt.close()
    
    age_stats = df.groupby('age_group')['strength'].agg(['mean', 'std', 'count'])
    return age_stats

# 3. Water-Cement Ratio Analysis
def analyze_water_cement_ratio():
    df['water_cement_ratio'] = df['water'] / df['cement']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='water_cement_ratio', y='strength')
    plt.title('Water-Cement Ratio vs Strength')
    plt.xlabel('Water-Cement Ratio')
    plt.ylabel('Strength (MPa)')
    plt.savefig('water_cement_ratio.png')
    plt.close()
    
    # Calculate correlation
    correlation = df['water_cement_ratio'].corr(df['strength'])
    return correlation

# 4. Predictive Modeling
def build_predictive_model():
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title('Actual vs Predicted Concrete Strength')
    plt.tight_layout()
    plt.savefig('prediction_performance.png')
    plt.close()
    
    return {
        'R2 Score': r2,
        'RMSE': rmse,
        'CV Scores Mean': cv_scores.mean(),
        'CV Scores Std': cv_scores.std()
    }

# Run all analyses
print("=== Feature Importance Analysis ===")
importance = analyze_feature_importance()
print(importance)

print("\n=== Strength Development Analysis ===")
age_stats = analyze_strength_development()
print(age_stats)

print("\n=== Water-Cement Ratio Analysis ===")
wc_correlation = analyze_water_cement_ratio()
print(f"Correlation between Water-Cement Ratio and Strength: {wc_correlation:.4f}")

print("\n=== Predictive Model Performance ===")
model_metrics = build_predictive_model()
for metric, value in model_metrics.items():
    print(f"{metric}: {value:.4f}")

# Additional insights about optimal mixture proportions
print("\n=== Optimal Mixture Analysis ===")
top_strength = df.nlargest(10, 'strength')
print("\nAverage proportions for top 10 strongest concrete mixtures:")
for column in ['cement', 'water', 'slag', 'ash', 'superplastic', 'coarseagg', 'fineagg', 'age']:
    print(f"{column}: {top_strength[column].mean():.2f}")
