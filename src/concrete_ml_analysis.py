import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

def load_and_check_data(file_path):
    """Load data and check for issues"""
    # Load the data
    df = pd.read_csv(file_path)
    
    print("=== Initial Data Check ===")
    print("\nShape of data:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    
    return df

def detect_outliers(df):
    """Detect outliers using IQR method"""
    outliers_dict = {}
    
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        outliers_dict[column] = len(outliers)
    
    print("\n=== Outliers Detection ===")
    print("Number of outliers in each column:")
    for col, count in outliers_dict.items():
        print(f"{col}: {count}")
    
    return outliers_dict

def handle_outliers(df):
    """Handle outliers using RobustScaler"""
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def plot_distributions(df, output_file='distributions.png'):
    """Plot distributions of all features"""
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'{column} Distribution')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def train_model(X, y):
    """Train and evaluate Random Forest model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    
    print("\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Average CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return rf_model, feature_importance

def main():
    # Load and check data
    df = load_and_check_data('concrete.csv')
    
    # Plot initial distributions
    plot_distributions(df, 'initial_distributions.png')
    
    # Detect outliers
    outliers = detect_outliers(df)
    
    # Separate features and target
    X = df.drop('strength', axis=1)
    y = df['strength']
    
    # Handle outliers
    X_cleaned = handle_outliers(X)
    
    # Plot cleaned distributions
    plot_distributions(X_cleaned, 'cleaned_distributions.png')
    
    # Train and evaluate model
    model, importance = train_model(X_cleaned, y)
    
    # Save feature importance
    importance.to_csv('feature_importance.csv', index=False)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- initial_distributions.png")
    print("- cleaned_distributions.png")
    print("- feature_importance.png")
    print("- feature_importance.csv")

if __name__ == "__main__":
    main()
