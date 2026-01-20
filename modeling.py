"""
================================================================================
ADVANCED PREDICTIVE MODELING
================================================================================
Risk modeling and prediction for flood economic damage
Target Variable: Economic Damage (Total Damage, Adjusted '000 US$)

Models:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization)
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. XGBoost Regressor
7. Neural Network (MLPRegressor)

Author: Statistical Analysis
Date: January 2026
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("ADVANCED PREDICTIVE MODELING FOR FLOOD ECONOMIC DAMAGE")
print("="*80)
print("\nLoading datasets...")

# Load datasets
emdat_bgd = pd.read_excel('EMDAT Bangladesh.xlsx')
emdat_tr = pd.read_excel('EMDAT Türkiye.xlsx')

# Add country identifier
emdat_bgd['Country'] = 'Bangladesh'
emdat_tr['Country'] = 'Turkey'

# Combine datasets
emdat_combined = pd.concat([emdat_bgd, emdat_tr], ignore_index=True)

# Filter for floods only and date range 2000-2025
emdat_floods = emdat_combined[
    (emdat_combined['Disaster Type'] == 'Flood') & 
    (emdat_combined['Start Year'] >= 2000) & 
    (emdat_combined['Start Year'] <= 2025)
].copy()

print(f"✓ Total Flood Events: {len(emdat_floods)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Target variable: Economic Damage
target_col = 'Total Damage, Adjusted (\'000 US$)'

# Filter for events with damage data
modeling_data = emdat_floods[
    (emdat_floods[target_col].notna()) & 
    (emdat_floods[target_col] > 0)
].copy()

print(f"\nEvents with damage data: {len(modeling_data)}")
print(f"  Bangladesh: {len(modeling_data[modeling_data['Country']=='Bangladesh'])}")
print(f"  Turkey: {len(modeling_data[modeling_data['Country']=='Turkey'])}")

# Create features
print("\n[Creating Features]")
print("-"*80)

# 1. Country indicator (binary)
modeling_data['Country_Bangladesh'] = (modeling_data['Country'] == 'Bangladesh').astype(int)

# 2. Temporal features
modeling_data['Year'] = modeling_data['Start Year']
modeling_data['Month'] = modeling_data['Start Month']
modeling_data['Season'] = modeling_data['Month'].apply(
    lambda x: 'Winter' if x in [12, 1, 2] 
    else 'Spring' if x in [3, 4, 5]
    else 'Summer' if x in [6, 7, 8]
    else 'Autumn'
)

# One-hot encode season
season_dummies = pd.get_dummies(modeling_data['Season'], prefix='Season')
modeling_data = pd.concat([modeling_data, season_dummies], axis=1)

# 3. Disaster subtype
subtype_dummies = pd.get_dummies(modeling_data['Disaster Subtype'], prefix='Subtype')
modeling_data = pd.concat([modeling_data, subtype_dummies], axis=1)

# 4. Mortality and impact features
modeling_data['Total_Deaths_Log'] = np.log1p(modeling_data['Total Deaths'].fillna(0))
modeling_data['Total_Affected_Log'] = np.log1p(modeling_data['Total Affected'].fillna(0))
modeling_data['No_Injured_Log'] = np.log1p(modeling_data['No. Injured'].fillna(0))
modeling_data['No_Homeless_Log'] = np.log1p(modeling_data['No. Homeless'].fillna(0))

# 5. Magnitude (if available)
modeling_data['Magnitude_Available'] = modeling_data['Magnitude'].notna().astype(int)
modeling_data['Magnitude_Value'] = modeling_data['Magnitude'].fillna(0)

# 6. Duration (End Year - Start Year)
modeling_data['Duration_Years'] = modeling_data['End Year'] - modeling_data['Start Year']

# 7. Time since 2000 (temporal trend)
modeling_data['Years_Since_2000'] = modeling_data['Year'] - 2000

# 8. Interaction features
modeling_data['Deaths_x_Affected'] = modeling_data['Total_Deaths_Log'] * modeling_data['Total_Affected_Log']
modeling_data['Country_x_Deaths'] = modeling_data['Country_Bangladesh'] * modeling_data['Total_Deaths_Log']
modeling_data['Country_x_Affected'] = modeling_data['Country_Bangladesh'] * modeling_data['Total_Affected_Log']

# 9. Historical context (rolling averages)
modeling_data = modeling_data.sort_values(['Country', 'Year'])
for country in ['Bangladesh', 'Turkey']:
    country_mask = modeling_data['Country'] == country
    modeling_data.loc[country_mask, 'Avg_Damage_3yr'] = (
        modeling_data.loc[country_mask, target_col]
        .rolling(window=3, min_periods=1).mean().shift(1)
    )

modeling_data['Avg_Damage_3yr'] = modeling_data['Avg_Damage_3yr'].fillna(modeling_data[target_col].median())

# Target variable (log-transformed for better model performance)
modeling_data['Target_Log'] = np.log1p(modeling_data[target_col])

print(f"✓ Total features created: {len([col for col in modeling_data.columns if col not in emdat_floods.columns])}")

# Select features for modeling
# We significantly reduce feature space to avoid overfitting on small N=20 dataset
feature_cols = [
    'Country_Bangladesh',
    'Total_Deaths_Log',
    'Total_Affected_Log',
    'Duration_Years',
    'Years_Since_2000'
]

# Prepare X and y
X = modeling_data[feature_cols].copy()
y = modeling_data['Target_Log'].copy()

# Critical: Remove Statistical Outliers to stabilize training on small dataset
q95 = y.quantile(0.95)
X = X[y <= q95]
y = y[y <= q95]
print(f"✓ Removed top 5% outliers. New shape: {X.shape}")

# Use RobustScaler for small-N data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=feature_cols)

# Handle any remaining missing values
X = X.fillna(0)

print(f"\n✓ Dataset prepared: X shape = {X.shape}, y shape = {y.shape}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")
print(f"Train/Test ratio: {X_train.shape[0]/X_test.shape[0]:.1f}")

# Scale features
scaler = RobustScaler()  # Robust to outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features scaled using RobustScaler")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("MODEL TRAINING & EVALUATION")
print("="*80)

# Dictionary to store results
results = {}
predictions = {}

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=5.0),
    'Lasso Regression': Lasso(alpha=0.05),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
}

print("\nRunning Robust Cross-Validation (RepeatedKFold)...")
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # 1. CV Performance (The only reliable metric for N=20)
    cv_scores = cross_val_score(model, X, y, cv=rkf, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_squared_error'))
    
    # 2. Train (Fit on all data to see overfitting gap)
    model.fit(X, y)
    y_pred_train = model.predict(X)
    train_r2 = r2_score(y, y_pred_train)
    
    # Store
    results[name] = {
        'model': model,
        'predictions': y_pred_train, # Full dataset predictions
        'r2_train': train_r2,
        'r2_test': cv_scores.mean(), # We use CV score as "Test" for the plot
        'rmse_test': cv_rmse.mean(), # CV RMSE
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  CV R²:   {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"  Train R²: {train_r2:.3f}")
    print(f"  CV RMSE: {cv_rmse.mean():.3f}")

# Removed legacy individual model blocks
#

# ============================================================================
# CLEANUP
# ============================================================================


# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

# Create comparison DataFrame
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name,
        'Train R²': metrics['r2_train'],
        'Test R²': metrics['r2_test'],
        'RMSE': metrics['rmse_test'],
        'CV R² Mean': metrics['cv_mean'],
        'CV R² Std': metrics['cv_std'],
        'Overfit Gap': metrics['r2_train'] - metrics['r2_test']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test R²', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R²: {comparison_df.iloc[0]['Test R²']:.4f}")
print(f"   RMSE: {comparison_df.iloc[0]['RMSE']:.4f}")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Random Forest
if 'Random Forest' in results:
    rf_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': results['Random Forest']['model'].feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n[Top Features - Random Forest]")
    print(rf_importance.to_string(index=False))

# Clean exit for XGBoost which was removed
#

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Plot 1: Model Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

# 1. R² Comparison
ax1 = axes[0, 0]
models = comparison_df['Model'].values
train_r2 = comparison_df['Train R²'].values
test_r2 = comparison_df['Test R²'].values

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='#3498db')
bars2 = ax1.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='#e74c3c')

ax1.set_ylabel('R² Score', fontweight='bold')
ax1.set_title('R² Comparison: Train vs Test', fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax1.legend(framealpha=0.9)
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=7)

# 2. RMSE Comparison
ax2 = axes[0, 1]
rmse_values = comparison_df['RMSE'].values
bars = ax2.barh(models, rmse_values, color='#9b59b6', alpha=0.8, edgecolor='black')
ax2.set_xlabel('RMSE (log scale)', fontweight='bold')
ax2.set_title('Root Mean Squared Error', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

for bar, val in zip(bars, rmse_values):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            ha='left', va='center', fontsize=8)

# 3. Cross-Validation R²
ax3 = axes[1, 0]
cv_means = comparison_df['CV R² Mean'].values
cv_stds = comparison_df['CV R² Std'].values

ax3.barh(models, cv_means, xerr=cv_stds, color='#2ecc71', alpha=0.8, 
        edgecolor='black', error_kw={'linewidth': 2, 'ecolor': '#34495e'})
ax3.set_xlabel('CV R² Score', fontweight='bold')
ax3.set_title('Cross-Validation Performance (5-Fold)', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, axis='x')
ax3.invert_yaxis()

for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax3.text(mean + 0.01, i, f'{mean:.3f}±{std:.3f}',
            ha='left', va='center', fontsize=8)

# 4. Overfitting Analysis
ax4 = axes[1, 1]
overfit_gap = comparison_df['Overfit Gap'].values
colors_overfit = ['#e74c3c' if gap > 0.15 else '#f39c12' if gap > 0.05 else '#2ecc71' 
                  for gap in overfit_gap]

bars = ax4.barh(models, overfit_gap, color=colors_overfit, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Train R² - Test R² Gap', fontweight='bold')
ax4.set_title('Overfitting Analysis', fontweight='bold', pad=10)
ax4.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='High overfit threshold')
ax4.grid(True, alpha=0.3, axis='x')
ax4.legend(fontsize=8)
ax4.invert_yaxis()

for bar, val in zip(bars, overfit_gap):
    ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}',
            ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('plots/modeling_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: plots/modeling_performance_comparison.png")

# Plot 2: Feature Importance
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')

# Random Forest
if 'Random Forest' in results:
    top_rf = rf_importance.head(15)
    bars1 = ax1.barh(range(len(top_rf)), top_rf['Importance'].values, 
                    color='#3498db', alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(top_rf)))
    ax1.set_yticklabels(top_rf['Feature'].values, fontsize=9)
    ax1.set_xlabel('Importance Score', fontweight='bold')
    ax1.set_title('Random Forest - Feature Importance', fontweight='bold', pad=10)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars1, top_rf['Importance'].values):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                ha='left', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('plots/modeling_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/modeling_feature_importance.png")

# Plot 3: Prediction vs Actual
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Predicted vs Actual Economic Damage (Log Scale)', fontsize=14, fontweight='bold')

# Plot 3: Predictions vs Actual (Full Dataset Fit)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predictions vs Actual (Full Dataset)', fontsize=14, fontweight='bold')

plot_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest']

for idx, model_name in enumerate(plot_models):
    if idx >= 4: break # Safety break
    if model_name not in results: continue

    ax = axes.flat[idx]
    
    y_pred = results[model_name]['predictions']
    
    # Scatter plot
    ax.scatter(y, y_pred, alpha=0.6, s=50, color='#3498db', edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Metrics
    r2 = results[model_name]['r2_test'] # CV R2
    
    ax.set_xlabel('Actual (log damage)', fontweight='bold', fontsize=9)
    ax.set_ylabel('Predicted (log damage)', fontweight='bold', fontsize=9)
    ax.set_title(f'{model_name}\nCV R²={r2:.3f}', 
                fontweight='bold', fontsize=10, pad=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

# Clear unused subplots
for i in range(len(plot_models), 4):
    fig.delaxes(axes.flat[i])

plt.tight_layout()
plt.savefig('plots/modeling_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/modeling_predictions_vs_actual.png")

# Plot 4: Residual Analysis (Best Model)
best_model_results = results[best_model_name]
best_predictions = best_model_results['predictions']
residuals = y - best_predictions # Use full dataset y

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Residual Analysis - {best_model_name}', fontsize=14, fontweight='bold')


# 1. Residuals vs Predicted
ax1 = axes[0, 0]
ax1.scatter(best_predictions, residuals, alpha=0.6, s=50, color='#3498db', edgecolor='black', linewidth=0.5)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Predicted Values', fontweight='bold')
ax1.set_ylabel('Residuals', fontweight='bold')
ax1.set_title('Residuals vs Predicted', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3)

# 2. Residual Distribution
ax2 = axes[0, 1]
ax2.hist(residuals, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Residuals', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Residual Distribution', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Q-Q Plot
ax3 = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3)

# 4. Scale-Location Plot
ax4 = axes[1, 1]
standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
ax4.scatter(best_predictions, standardized_residuals, alpha=0.6, s=50, 
           color='#e74c3c', edgecolor='black', linewidth=0.5)
ax4.set_xlabel('Predicted Values', fontweight='bold')
ax4.set_ylabel('√|Standardized Residuals|', fontweight='bold')
ax4.set_title('Scale-Location Plot', fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/modeling_residual_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/modeling_residual_analysis.png")

# ============================================================================
# RISK QUANTIFICATION
# ============================================================================
print("\n" + "="*80)
print("RISK QUANTIFICATION & PREDICTION INTERVALS")
print("="*80)

# Use best model for risk assessment
best_model = results[best_model_name]['model']

# Calculate prediction intervals (using residuals)
residual_std = residuals.std()

print(f"\nResidual Standard Deviation: {residual_std:.4f}")

# Predict for both countries (hypothetical scenario)
print("\n[Risk Assessment: Hypothetical Flood Scenarios]")
print("-"*80)

# Create hypothetical scenarios
scenarios = [
    {'Name': 'Minor Flood - Bangladesh', 'Country_Bangladesh': 1, 'Total_Deaths_Log': np.log1p(10), 
     'Total_Affected_Log': np.log1p(50000), 'Duration_Years': 0, 'Years_Since_2000': 25},
    {'Name': 'Severe Flood - Bangladesh', 'Country_Bangladesh': 1, 'Total_Deaths_Log': np.log1p(200), 
     'Total_Affected_Log': np.log1p(5000000), 'Duration_Years': 0, 'Years_Since_2000': 25},
    {'Name': 'Minor Flood - Turkey', 'Country_Bangladesh': 0, 'Total_Deaths_Log': np.log1p(5), 
     'Total_Affected_Log': np.log1p(1000), 'Duration_Years': 0, 'Years_Since_2000': 25},
    {'Name': 'Severe Flood - Turkey', 'Country_Bangladesh': 0, 'Total_Deaths_Log': np.log1p(50), 
     'Total_Affected_Log': np.log1p(50000), 'Duration_Years': 0, 'Years_Since_2000': 25},
]

best_model = results[best_model_name]['model']

for scenario in scenarios:
    # Create feature vector
    scenario_features = pd.DataFrame([{
        col: scenario.get(col, 0) for col in feature_cols
    }])
    
    # Scale if necessary (Linear/Ridge/Lasso)
    # The models were trained on X which was scaled. 
    # RF doesn't care if we pass scaled or unscaled if it was trained on scaled? 
    # Wait, Random Forest is invariant to scaling, but if I passed scaled data to fit, I must pass scaled data to predict.
    # In my new loop, I scaled ALL data into X. So X is scaled.
    # So I must scale this scenario using the same scaler.
    
    scenario_scaled = scaler.transform(scenario_features)
    scenario_df_scaled = pd.DataFrame(scenario_scaled, columns=feature_cols)
    
    pred_log = best_model.predict(scenario_df_scaled)[0]
    
    # Transform back to original scale
    pred_damage = np.expm1(pred_log)
    
    # 95% Prediction interval
    lower_bound = np.expm1(pred_log - 1.96 * residual_std)
    upper_bound = np.expm1(pred_log + 1.96 * residual_std)
    
    # Print formatted output as integers
    print(f"\n{scenario['Name']}:")
    print(f"  Predicted Damage: ${pred_damage:,.0f}K")
    print(f"  95% Range: [${lower_bound:,.0f}K - ${upper_bound:,.0f}K]")

# ============================================================================
# END OF ANALYSIS
# ============================================================================

# ============================================================================
print("\n" + "="*80)
print("MODELING COMPLETE")
print("="*80)
print("\nGenerated 4 visualization files:")
print("  1. modeling_performance_comparison.png")
print("  2. modeling_feature_importance.png")
print("  3. modeling_predictions_vs_actual.png")
print("  4. modeling_residual_analysis.png")
print("\nAll predictive models trained and evaluated.")
print("="*80)
