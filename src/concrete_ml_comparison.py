import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the data"""
    # Load data
    df = pd.read_csv('concrete.csv')
    print("\n=== Data Overview ===")
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Separate features and target
    X = df.drop('strength', axis=1)
    y = df['strength']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y

def train_and_evaluate_model(model, X, y, model_name):
    """Train and evaluate a single model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Print results
    print(f"\n=== {model_name} Results ===")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"CV Scores Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.savefig(f'{model_name.lower()}_predictions.png')
    plt.close()
    
    # Feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_importance.png')
        plt.close()
        
        print("\nFeature Importance:")
        print(importance)
    
    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Lasso': LassoCV(cv=5, random_state=42),
        'Ridge': RidgeCV(cv=5)
    }
    
    # Train and evaluate all models
    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate_model(model, X, y, name)
    
    # Compare models
    print("\n=== Model Comparison ===")
    comparison = pd.DataFrame({
        name: {
            'R²': res['r2'],
            'RMSE': res['rmse'],
            'MAE': res['mae'],
            'CV R² (mean)': res['cv_mean'],
            'CV R² (std)': res['cv_std']
        }
        for name, res in results.items()
    }).round(4)
    
    print("\nModel Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv('model_comparison.csv')
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\nBest Model: {best_model_name}")
    
    # Example prediction with best model
    example_mixture = pd.DataFrame([[380, 120, 0, 180, 8, 1000, 800, 28]], 
                                 columns=['cement', 'slag', 'ash', 'water', 
                                        'superplastic', 'coarseagg', 'fineagg', 'age'])
    
    best_model = results[best_model_name]['model']
    predicted_strength = best_model.predict(example_mixture)[0]
    
    print(f"\nExample Prediction:")
    print(f"Mixture: {example_mixture.iloc[0].to_dict()}")
    print(f"Predicted Strength: {predicted_strength:.2f} MPa")

if __name__ == "__main__":
    main()
