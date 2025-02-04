import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ConcreteStrengthPredictor:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'Lasso': LassoCV(cv=5),
            'Ridge': RidgeCV(cv=5),
            'SVR': SVR(kernel='rbf')
        }
        
        self.best_model = None
        self.feature_selector = None
        self.scaler = RobustScaler()
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the data"""
        # Load data
        df = pd.read_csv(file_path)
        print("\n=== Data Overview ===")
        print("Shape:", df.shape)
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Plot initial distributions
        self.plot_distributions(df, 'initial_distributions.png')
        
        # Handle outliers
        X = df.drop('strength', axis=1)
        y = df['strength']
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Plot cleaned distributions
        self.plot_distributions(X_scaled, 'cleaned_distributions.png')
        
        return X_scaled, y
    
    def plot_distributions(self, df, output_file):
        """Plot feature distributions"""
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(df.columns, 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[column], kde=True)
            plt.title(f'{column} Distribution')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def select_features(self, X, y):
        """Perform feature selection"""
        # Initialize selector with RandomForest
        self.feature_selector = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            prefit=False
        )
        
        # Fit and transform
        self.feature_selector.fit(X, y)
        X_selected = self.feature_selector.transform(X)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print("\n=== Selected Features ===")
        print("Features kept:", selected_features)
        
        return X_selected, selected_features
    
    def tune_hyperparameters(self, X, y):
        """Perform hyperparameter tuning"""
        # Define parameter grids for each model
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7]
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        }
        
        best_models = {}
        best_score = 0
        
        print("\n=== Hyperparameter Tuning ===")
        for name, model in self.models.items():
            if name in param_grids:
                print(f"\nTuning {name}...")
                grid_search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X, y)
                best_models[name] = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best score: {grid_search.best_score_:.4f}")
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
            else:
                model.fit(X, y)
                best_models[name] = model
        
        return best_models
    
    def evaluate_models(self, models, X, y):
        """Evaluate all models"""
        results = {}
        print("\n=== Model Evaluation ===")
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Train-test split evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'R2': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'CV_mean': cv_scores.mean(),
                'CV_std': cv_scores.std()
            }
            
            print(f"\n{name} Results:")
            for metric, value in results[name].items():
                print(f"{metric}: {value:.4f}")
            
            # Plot actual vs predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Strength (MPa)')
            plt.ylabel('Predicted Strength (MPa)')
            plt.title(f'{name}: Actual vs Predicted')
            plt.savefig(f'{name}_predictions.png')
            plt.close()
        
        return results
    
    def save_model(self, filename='concrete_strength_model.joblib'):
        """Save the best model"""
        if self.best_model is not None:
            joblib.dump(self.best_model, filename)
            print(f"\nBest model saved as {filename}")
    
    def predict_strength(self, mixture_data):
        """Predict concrete strength for new mixture"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale the input data
        scaled_data = self.scaler.transform(mixture_data)
        
        # Make prediction
        prediction = self.best_model.predict(scaled_data)
        return prediction[0]

def main():
    # Initialize predictor
    predictor = ConcreteStrengthPredictor()
    
    # Load and preprocess data
    X, y = predictor.load_and_preprocess_data('concrete.csv')
    
    # Select features
    X_selected, selected_features = predictor.select_features(X, y)
    
    # Tune and train models
    best_models = predictor.tune_hyperparameters(X_selected, y)
    
    # Evaluate models
    results = predictor.evaluate_models(best_models, X_selected, y)
    
    # Save the best model
    predictor.save_model()
    
    # Example prediction
    example_mixture = pd.DataFrame([[380, 120, 0, 180, 8, 1000, 800, 28]], 
                                 columns=['cement', 'slag', 'ash', 'water', 
                                        'superplastic', 'coarseagg', 'fineagg', 'age'])
    predicted_strength = predictor.predict_strength(example_mixture)
    print(f"\nExample Prediction:")
    print(f"Mixture: {example_mixture.iloc[0].to_dict()}")
    print(f"Predicted Strength: {predicted_strength:.2f} MPa")

if __name__ == "__main__":
    main()
