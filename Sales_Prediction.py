# Sales Prediction Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class SalesPredictionAnalyzer:
   
    def __init__(self, csv_path='sales_prediction.csv'):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            print(" Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(" File not found. Please ensure 'sales_prediction.csv' is in the current directory.")
            return None
    
    def explore_data(self):
        if self.df is None:
            print(" Please load data first using load_data()")
            return
        
        print("\n" + "="*50)
        print(" EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information 
        print("\n1. Dataset Overview:")
        print(self.df.info())
        
        print("\n2. Statistical Summary:")
        print(self.df.describe())
        
        print("\n3. Missing Values:")
        print(self.df.isnull().sum())
        
        # visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sales Prediction - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # Distribution plots
        for i, col in enumerate(['TV', 'Radio', 'Newspaper', 'Sales']):
            if i < 4:
                row, col_idx = i // 2, i % 2
                if i < 2:
                    axes[row, col_idx].hist(self.df[col], bins=20, alpha=0.7, edgecolor='black')
                    axes[row, col_idx].set_title(f'{col} Distribution')
                    axes[row, col_idx].set_xlabel(col)
                    axes[row, col_idx].set_ylabel('Frequency')
        
        # Correlation map
        corr_matrix = self.df.corr()
        im = axes[0, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[0, 2].set_xticks(range(len(corr_matrix.columns)))
        axes[0, 2].set_yticks(range(len(corr_matrix.columns)))
        axes[0, 2].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[0, 2].set_yticklabels(corr_matrix.columns)
        axes[0, 2].set_title('Correlation Heatmap')
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[0, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        # Scatter plot for TV vs Sales
        axes[1, 2].scatter(self.df['TV'], self.df['Sales'], alpha=0.6, color='blue')
        axes[1, 2].set_xlabel('TV Advertising Spend')
        axes[1, 2].set_ylabel('Sales')
        axes[1, 2].set_title('TV Spend vs Sales')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        print("\n4. Correlation with Sales:")
        correlations = self.df.corr()['Sales'].sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != 'Sales':
                print(f"   {feature}: {corr:.3f}")
    
    def prepare_features(self):
        if self.df is None:
            print(" Please load data first")
            return
        
        print("\n" + "="*50)
        print(" FEATURE ENGINEERING")
        print("="*50)
        
        self.df['Total_Spend'] = self.df['TV'] + self.df['Radio'] + self.df['Newspaper']
        self.df['TV_Radio_Interaction'] = self.df['TV'] * self.df['Radio']
        self.df['Spend_per_Channel'] = self.df['Total_Spend'] / 3
        
        self.df['TV_log'] = np.log1p(self.df['TV'])
        self.df['Radio_log'] = np.log1p(self.df['Radio'])
        
        print("New features created:")
        print("   - Total_Spend: Sum of all advertising channels")
        print("   - TV_Radio_Interaction: Interaction term between TV and Radio")
        print("   - Spend_per_Channel: Average spend per channel")
        print("   - TV_log, Radio_log: Log-transformed features")
        
        self.basic_features = ['TV', 'Radio', 'Newspaper']
        self.advanced_features = ['TV', 'Radio', 'Newspaper', 'Total_Spend', 
                                'TV_Radio_Interaction', 'TV_log', 'Radio_log']
        
        return self.df
    
    def split_data(self, feature_set='basic', test_size=0.2, random_state=42):
        if self.df is None:
            print("Please load and prepare data first")
            return
        
        features = self.basic_features if feature_set == 'basic' else self.advanced_features
        
        X = self.df[features]
        y = self.df['Sales']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split completed:")
        print(f"   Training set: {self.X_train.shape[0]} samples")
        print(f"   Testing set: {self.X_test.shape[0]} samples")
        print(f"   Features used: {features}")
    
    def train_models(self):
        if self.X_train is None:
            print("Please split data first using split_data()")
            return
        
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Training models and making predictions
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            model.fit(self.X_train, self.y_train)
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            self.predictions[name] = {
                'train': y_pred_train,
                'test': y_pred_test
            }
            
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            
            self.metrics[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   {name} completed")
    
    def evaluate_models(self):
        if not self.metrics:
            print("Please train models first using train_models()")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # evaluation DataFrame
        eval_df = pd.DataFrame(self.metrics).T
        eval_df = eval_df.round(4)
        
        print("\nModel Performance Summary:")
        print(eval_df)
        
        # best model
        best_model = eval_df['test_r2'].idxmax()
        print(f"\nBest Model: {best_model}")
        print(f"   Test R²: {eval_df.loc[best_model, 'test_r2']:.4f}")
        print(f"   Test RMSE: {eval_df.loc[best_model, 'test_rmse']:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # R² comparison
        models = list(self.metrics.keys())
        train_r2 = [self.metrics[m]['train_r2'] for m in models]
        test_r2 = [self.metrics[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        axes[0, 0].bar(x - 0.2, train_r2, 0.4, label='Train R²', alpha=0.8)
        axes[0, 0].bar(x + 0.2, test_r2, 0.4, label='Test R²', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse = [self.metrics[m]['test_rmse'] for m in models]
        axes[0, 1].bar(models, test_rmse, alpha=0.8, color='coral')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Test RMSE Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predc vs Actl for best model
        best_pred = self.predictions[best_model]['test']
        axes[1, 0].scatter(self.y_test, best_pred, alpha=0.6)
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Sales')
        axes[1, 0].set_ylabel('Predicted Sales')
        axes[1, 0].set_title(f'{best_model}: Predicted vs Actual')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals plot for best model
        residuals = self.y_test - best_pred
        axes[1, 1].scatter(best_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Sales')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'{best_model}: Residuals Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return eval_df
    
    def feature_importance(self):
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Regression Coefficients
        lr_model = self.models['Linear Regression']
        feature_names = self.X_train.columns
        
        print("\nLinear Regression Coefficients:")
        for feature, coef in zip(feature_names, lr_model.coef_):
            print(f"   {feature}: {coef:.4f}")
        print(f"   Intercept: {lr_model.intercept_:.4f}")
        
        # Random Forest 
        rf_model = self.models['Random Forest']
        rf_importance = rf_model.feature_importances_
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # LR coefficients
        abs_coefs = np.abs(lr_model.coef_)
        axes[0].barh(feature_names, abs_coefs)
        axes[0].set_xlabel('Absolute Coefficient Value')
        axes[0].set_title('Linear Regression - Feature Coefficients')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].barh(feature_names, rf_importance)
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('Random Forest - Feature Importance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def business_insights(self):
        print("\n" + "="*50)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*50)
        
        if not self.metrics:
            print("Please train models first")
            return
        
        # Best model analysis
        best_model_name = max(self.metrics.keys(), key=lambda x: self.metrics[x]['test_r2'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest Performing Model: {best_model_name}")
        print(f"   Accuracy (R²): {self.metrics[best_model_name]['test_r2']:.1%}")
        print(f"   Average Error (RMSE): ₹{self.metrics[best_model_name]['test_rmse']:.2f}K")
        
        # Channel effectiveness (based on Linear Regression)
        lr_model = self.models['Linear Regression']
        feature_names = self.X_train.columns
        coefficients = dict(zip(feature_names, lr_model.coef_))
        
        print(f"\nChannel Effectiveness Analysis:")
        for channel in ['TV', 'Radio', 'Newspaper']:
            if channel in coefficients:
                roi = coefficients[channel]
                print(f"   {channel}: ₹{roi:.3f} sales increase per ₹1 spent")
        
        print(f"\nBudget Optimization Recommendations:")
        print(f"   1. Prioritize channels with highest coefficients")
        print(f"   2. Current model explains {self.metrics[best_model_name]['test_r2']:.1%} of sales variance")
        print(f"   3. Consider interaction effects between channels")
        
        print(f"\nPrediction Capability:")
        print(f"   Model can predict sales within ±₹{self.metrics[best_model_name]['test_rmse']:.2f}K")
        print(f"   Suitable for budget planning and ROI forecasting")
        
        print(f"\nNext Steps for Improvement:")
        print(f"   • Collect more data (seasonality, competition, promotions)")
        print(f"   • Experiment with advanced models (XGBoost, Neural Networks)")
        print(f"   • Implement real-time prediction system")
        print(f"   • A/B test different budget allocations")
    
    def predict_sales(self, tv_spend, radio_spend, newspaper_spend, model_name=None):
        if not self.models:
            print("Please train models first")
            return None
        
        if model_name is None:
            model_name = max(self.metrics.keys(), key=lambda x: self.metrics[x]['test_r2'])
        
        model = self.models[model_name]
        
        # input data
        input_data = np.array([[tv_spend, radio_spend, newspaper_spend]])
        
        if len(self.X_train.columns) > 3:
            total_spend = tv_spend + radio_spend + newspaper_spend
            tv_radio_interaction = tv_spend * radio_spend
            tv_log = np.log1p(tv_spend)
            radio_log = np.log1p(radio_spend)
            
            input_data = np.array([[tv_spend, radio_spend, newspaper_spend, 
                                  total_spend, tv_radio_interaction, tv_log, radio_log]])
        
        prediction = model.predict(input_data)[0]
        
        print(f"\nSales Prediction using {model_name}:")
        print(f"   TV Spend: ₹{tv_spend}K")
        print(f"   Radio Spend: ₹{radio_spend}K") 
        print(f"   Newspaper Spend: ₹{newspaper_spend}K")
        print(f"   Predicted Sales: ₹{prediction:.2f}K")
        
        return prediction

# Example usage and complete analysis pipeline
def run_complete_analysis():
    print("Starting Sales Prediction Analysis")
    print("="*60)
    
    analyzer = SalesPredictionAnalyzer()
    
    analyzer.load_data()
    analyzer.explore_data()
    
    analyzer.prepare_features()
    
    analyzer.split_data(feature_set='advanced')
    analyzer.train_models()
    
    results = analyzer.evaluate_models()
    
    analyzer.feature_importance()
    
    analyzer.business_insights()
    
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    analyzer.predict_sales(tv_spend=100, radio_spend=50, newspaper_spend=25)
    
    return analyzer

if __name__ == "__main__":
    analyzer = SalesPredictionAnalyzer()

    analyzer.load_data()
    
    analyzer.explore_data()
    
    analyzer.prepare_features()
    
    analyzer.split_data(feature_set='advanced')
    
    analyzer.train_models()
    
    analyzer.evaluate_models()
    
    analyzer.feature_importance()
    
    analyzer.business_insights()
    
    analyzer.predict_sales(tv_spend=150, radio_spend=40, newspaper_spend=30)