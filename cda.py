"""
================================================================================
CONFIRMATORY DATA ANALYSIS (CDA)
================================================================================
Statistical hypothesis testing and modeling for Bangladesh vs Turkey flood analysis
Based on 5 Research Questions identified in EDA

Author: Statistical Analysis
Date: January 2026
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, levene, shapiro, kstest, anderson
from scipy.stats import lognorm, weibull_min, expon, gamma
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("CONFIRMATORY DATA ANALYSIS (CDA)")
print("="*80)
print("\nLoading datasets...")

# Load datasets
emdat_bgd = pd.read_excel('EMDAT Bangladesh.xlsx')
emdat_tr = pd.read_excel('EMDAT Türkiye.xlsx')
flood_obs_bgd = pd.read_excel('flood_observatory_bangladesh.xlsx')
flood_obs_tr = pd.read_excel('Turkey_flood_observatory.xlsx')

# Add country identifier
emdat_bgd['Country'] = 'Bangladesh'
emdat_tr['Country'] = 'Turkey'
flood_obs_bgd['Country'] = 'Bangladesh'
flood_obs_tr['Country'] = 'Turkey'

# Combine datasets
emdat_combined = pd.concat([emdat_bgd, emdat_tr], ignore_index=True)

# Filter for floods only and date range 2000-2025
emdat_floods = emdat_combined[
    (emdat_combined['Disaster Type'] == 'Flood') & 
    (emdat_combined['Start Year'] >= 2000) & 
    (emdat_combined['Start Year'] <= 2025)
].copy()

print(f"✓ EMDAT Flood Events: {len(emdat_floods)} (Bangladesh: {len(emdat_floods[emdat_floods['Country']=='Bangladesh'])}, Turkey: {len(emdat_floods[emdat_floods['Country']=='Turkey'])})")

# Colors for consistency
colors = {'Bangladesh': '#e74c3c', 'Turkey': '#3498db'}

# ============================================================================
# RQ1: COMPARATIVE RISK PROFILING - HYPOTHESIS TESTING
# ============================================================================
print("\n" + "="*80)
print("RQ1: COMPARATIVE RISK PROFILING - HYPOTHESIS TESTING")
print("="*80)

print("\n[H1] Testing if Bangladesh has significantly higher mortality than Turkey")
print("-"*80)

# Prepare data
bgd_deaths = emdat_floods[emdat_floods['Country'] == 'Bangladesh']['Total Deaths'].dropna()
tr_deaths = emdat_floods[emdat_floods['Country'] == 'Turkey']['Total Deaths'].dropna()

# Normality tests
print("\n1. Normality Assessment:")
print("   Shapiro-Wilk Test (H0: Data is normally distributed)")
shapiro_bgd_deaths = shapiro(bgd_deaths)
shapiro_tr_deaths = shapiro(tr_deaths)
print(f"   Bangladesh Deaths: W={shapiro_bgd_deaths.statistic:.4f}, p={shapiro_bgd_deaths.pvalue:.4f} {'(Normal)' if shapiro_bgd_deaths.pvalue > 0.05 else '(Not Normal)'}")
print(f"   Turkey Deaths: W={shapiro_tr_deaths.statistic:.4f}, p={shapiro_tr_deaths.pvalue:.4f} {'(Normal)' if shapiro_tr_deaths.pvalue > 0.05 else '(Not Normal)'}")

# Since data is not normal, use Mann-Whitney U test
print("\n2. Mann-Whitney U Test (H0: Distributions are equal)")
u_stat, p_value = mannwhitneyu(bgd_deaths, tr_deaths, alternative='greater')
print(f"   U-statistic: {u_stat:.2f}")
print(f"   p-value: {p_value:.4f}")
print(f"   Result: {'REJECT H0 - Bangladesh has significantly higher deaths' if p_value < 0.05 else 'FAIL TO REJECT H0 - No significant difference'}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((bgd_deaths.std()**2 + tr_deaths.std()**2) / 2)
cohens_d = (bgd_deaths.mean() - tr_deaths.mean()) / pooled_std
print(f"   Cohen's d: {cohens_d:.3f} ({'Large effect' if abs(cohens_d) > 0.8 else 'Medium effect' if abs(cohens_d) > 0.5 else 'Small effect'})")

# Variance homogeneity
levene_stat, levene_p = levene(bgd_deaths, tr_deaths)
print(f"\n3. Levene's Test for Variance Homogeneity: F={levene_stat:.2f}, p={levene_p:.4f}")
print(f"   Result: {'Equal variances' if levene_p > 0.05 else 'Unequal variances'}")

# ============================================================================
print("\n[H2] Testing Economic Damage Differences")
print("-"*80)

bgd_damage = emdat_floods[(emdat_floods['Country'] == 'Bangladesh') & 
                          (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]['Total Damage, Adjusted (\'000 US$)'].dropna()
tr_damage = emdat_floods[(emdat_floods['Country'] == 'Turkey') & 
                         (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]['Total Damage, Adjusted (\'000 US$)'].dropna()

# Mann-Whitney U test for damage
u_stat_dmg, p_value_dmg = mannwhitneyu(bgd_damage, tr_damage, alternative='two-sided')
print(f"\n   Mann-Whitney U Test:")
print(f"   U-statistic: {u_stat_dmg:.2f}, p-value: {p_value_dmg:.4f}")
print(f"   Result: {'REJECT H0 - Significant difference in economic damage' if p_value_dmg < 0.05 else 'FAIL TO REJECT H0 - No significant difference'}")

# Median comparison
print(f"\n   Median Economic Damage:")
print(f"   Bangladesh: ${bgd_damage.median():,.0f}K (${bgd_damage.median()/1000:.0f}M)")
print(f"   Turkey: ${tr_damage.median():,.0f}K (${tr_damage.median()/1000:.0f}M)")
print(f"   Ratio: {bgd_damage.median() / tr_damage.median():.2f}x")

# ============================================================================
print("\n[H3] Testing Affected Population Differences")
print("-"*80)

bgd_affected = emdat_floods[(emdat_floods['Country'] == 'Bangladesh') & 
                            (emdat_floods['Total Affected'] > 0)]['Total Affected'].dropna()
tr_affected = emdat_floods[(emdat_floods['Country'] == 'Turkey') & 
                           (emdat_floods['Total Affected'] > 0)]['Total Affected'].dropna()

u_stat_aff, p_value_aff = mannwhitneyu(bgd_affected, tr_affected, alternative='greater')
print(f"\n   Mann-Whitney U Test:")
print(f"   U-statistic: {u_stat_aff:.2f}, p-value: {p_value_aff:.4f}")
print(f"   Result: {'REJECT H0 - Bangladesh has significantly more affected population' if p_value_aff < 0.05 else 'FAIL TO REJECT H0'}")

print(f"\n   Median Affected Population:")
print(f"   Bangladesh: {bgd_affected.median():,.0f}")
print(f"   Turkey: {tr_affected.median():,.0f}")
print(f"   Ratio: {bgd_affected.median() / tr_affected.median():.0f}x")

# ============================================================================
# RQ2: STATISTICAL SIGNIFICANCE OF SEVERITY DIFFERENCES
# ============================================================================
print("\n" + "="*80)
print("RQ2: SEVERITY INDEX CONSTRUCTION & TESTING")
print("="*80)

print("\n[Creating Composite Severity Index]")
print("-"*80)

# Create severity index (normalized composite score)
severity_data = emdat_floods[
    (emdat_floods['Total Deaths'].notna()) & 
    (emdat_floods['Total Affected'].notna()) & 
    (emdat_floods['Total Damage, Adjusted (\'000 US$)'].notna()) &
    (emdat_floods['Total Deaths'] > 0) &
    (emdat_floods['Total Affected'] > 0) &
    (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)
].copy()

# Log transform to handle skewness
severity_data['log_deaths'] = np.log10(severity_data['Total Deaths'])
severity_data['log_affected'] = np.log10(severity_data['Total Affected'])
severity_data['log_damage'] = np.log10(severity_data['Total Damage, Adjusted (\'000 US$)'])

# Standardize
scaler = StandardScaler()
severity_data[['std_deaths', 'std_affected', 'std_damage']] = scaler.fit_transform(
    severity_data[['log_deaths', 'log_affected', 'log_damage']]
)

# Composite severity index (equal weights)
severity_data['severity_index'] = (
    severity_data['std_deaths'] + 
    severity_data['std_affected'] + 
    severity_data['std_damage']
) / 3

print(f"\n   Events with complete severity data: {len(severity_data)}")
print(f"   Bangladesh: {len(severity_data[severity_data['Country']=='Bangladesh'])}")
print(f"   Turkey: {len(severity_data[severity_data['Country']=='Turkey'])}")

# Statistical test on severity index
bgd_severity = severity_data[severity_data['Country'] == 'Bangladesh']['severity_index']
tr_severity = severity_data[severity_data['Country'] == 'Turkey']['severity_index']

print("\n[Hypothesis Test on Severity Index]")
u_stat_sev, p_value_sev = mannwhitneyu(bgd_severity, tr_severity, alternative='two-sided')
print(f"   Mann-Whitney U Test: U={u_stat_sev:.2f}, p={p_value_sev:.4f}")
print(f"   Result: {'REJECT H0 - Significant severity difference' if p_value_sev < 0.05 else 'FAIL TO REJECT H0 - No significant severity difference'}")

print(f"\n   Mean Severity Index:")
print(f"   Bangladesh: {bgd_severity.mean():.3f} (±{bgd_severity.std():.3f})")
print(f"   Turkey: {tr_severity.mean():.3f} (±{tr_severity.std():.3f})")

# ============================================================================
# RQ3: TEMPORAL TREND ANALYSIS - TIME SERIES HYPOTHESIS TESTING
# ============================================================================
print("\n" + "="*80)
print("RQ3: TEMPORAL TREND ANALYSIS")
print("="*80)

print("\n[Testing for Temporal Trends in Event Frequency]")
print("-"*80)

# Yearly event counts
yearly_events = emdat_floods.groupby(['Country', 'Start Year']).size().reset_index(name='Count')

for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_events[yearly_events['Country'] == country]
    years = country_data['Start Year'].values
    counts = country_data['Count'].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, counts)
    
    print(f"\n   {country}:")
    print(f"   Slope: {slope:.4f} events/year")
    print(f"   R²: {r_value**2:.4f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Result: {'Significant trend' if p_value < 0.05 else 'No significant trend'}")
    print(f"   Direction: {'Increasing' if slope > 0 else 'Decreasing'}")

# Mann-Kendall trend test (non-parametric)
def mann_kendall_test(data):
    """Mann-Kendall trend test"""
    n = len(data)
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(data[j] - data[i])
    
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

print("\n[Mann-Kendall Trend Test]")
for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_events[yearly_events['Country'] == country].sort_values('Start Year')
    z_stat, p_val = mann_kendall_test(country_data['Count'].values)
    print(f"   {country}: Z={z_stat:.3f}, p={p_val:.4f} {'(Significant trend)' if p_val < 0.05 else '(No significant trend)'}")

# ============================================================================
print("\n[Testing for Stationarity in Impact Metrics]")
print("-"*80)

# Augmented Dickey-Fuller test for stationarity
from statsmodels.tsa.stattools import adfuller

yearly_deaths = emdat_floods.groupby(['Country', 'Start Year'])['Total Deaths'].sum().reset_index()

for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_deaths[yearly_deaths['Country'] == country].sort_values('Start Year')
    deaths = country_data['Total Deaths'].values
    
    if len(deaths) > 3:
        result = adfuller(deaths, autolag='AIC')
        print(f"\n   {country} - Augmented Dickey-Fuller Test:")
        print(f"   ADF Statistic: {result[0]:.4f}")
        print(f"   p-value: {result[1]:.4f}")
        print(f"   Result: {'Stationary' if result[1] < 0.05 else 'Non-stationary'}")

# ============================================================================
# RQ4: DISTRIBUTION MODELING & GOODNESS-OF-FIT TESTS
# ============================================================================
print("\n" + "="*80)
print("RQ4: DISTRIBUTION FITTING & GOODNESS-OF-FIT TESTS")
print("="*80)

distributions = {
    'Lognormal': lognorm,
    'Weibull': weibull_min,
    'Exponential': expon,
    'Gamma': gamma
}

metrics = {
    'Total Deaths': 'Total Deaths',
    'Total Affected': 'Total Affected',
    'Economic Damage': 'Total Damage, Adjusted (\'000 US$)'
}

print("\n[Goodness-of-Fit Tests: Kolmogorov-Smirnov Test]")
print("-"*80)

fit_results = []

for metric_name, metric_col in metrics.items():
    print(f"\n{metric_name}:")
    print("   " + "-"*70)
    
    for country in ['Bangladesh', 'Turkey']:
        raw_data = emdat_floods[(emdat_floods['Country'] == country) & 
                           (emdat_floods[metric_col] > 0)][metric_col].dropna().values
        
        # Remove outliers (IQR method)
        Q1 = np.percentile(raw_data, 25)
        Q3 = np.percentile(raw_data, 75)
        IQR = Q3 - Q1
        data = raw_data[(raw_data >= Q1 - 1.5 * IQR) & (raw_data <= Q3 + 1.5 * IQR)]
        
        if len(data) < 10:
            continue
        
        print(f"\n   {country} (n={len(data)}, outliers_removed={len(raw_data)-len(data)}):")
        best_fit = None
        best_p = 0
        
        for dist_name, dist in distributions.items():
            try:
                # Fit distribution
                params = dist.fit(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(data, dist.cdf, args=params)
                
                print(f"      {dist_name:12s}: KS={ks_stat:.4f}, p={ks_p:.4f} {'✓' if ks_p > 0.05 else '✗'}")
                
                if ks_p > best_p:
                    best_p = ks_p
                    best_fit = dist_name
                
                fit_results.append({
                    'Country': country,
                    'Metric': metric_name,
                    'Distribution': dist_name,
                    'KS_statistic': ks_stat,
                    'p_value': ks_p,
                    'Fit': 'Good' if ks_p > 0.05 else 'Poor'
                })
            except:
                pass
        
        if best_fit:
            print(f"      Best fit: {best_fit} (p={best_p:.4f})")

# ============================================================================
# RQ5: EXTREME VALUE ANALYSIS - THRESHOLD EXCEEDANCE
# ============================================================================
print("\n" + "="*80)
print("RQ5: EXTREME VALUE ANALYSIS")
print("="*80)

print("\n[Threshold Exceedance Analysis]")
print("-"*80)

# Define extreme event thresholds (90th percentile)
for metric_name, metric_col in metrics.items():
    print(f"\n{metric_name}:")
    
    for country in ['Bangladesh', 'Turkey']:
        data = emdat_floods[(emdat_floods['Country'] == country) & 
                           (emdat_floods[metric_col] > 0)][metric_col].dropna()
        
        if len(data) == 0:
            continue
        
        # Thresholds
        p50 = data.quantile(0.50)
        p90 = data.quantile(0.90)
        p95 = data.quantile(0.95)
        p99 = data.quantile(0.99)
        
        extreme_count = (data > p90).sum()
        extreme_pct = (extreme_count / len(data)) * 100
        
        print(f"\n   {country}:")
        print(f"      Median (P50): {p50:,.0f}")
        print(f"      P90 threshold: {p90:,.0f}")
        print(f"      P95 threshold: {p95:,.0f}")
        print(f"      P99 threshold: {p99:,.0f}")
        print(f"      Events exceeding P90: {extreme_count} ({extreme_pct:.1f}%)")

# ============================================================================
print("\n[Return Period Analysis]")
print("-"*80)

# Calculate return periods for deaths
print("\nEstimated Return Periods (based on empirical distribution):")

for country in ['Bangladesh', 'Turkey']:
    deaths_data = emdat_floods[(emdat_floods['Country'] == country) & 
                               (emdat_floods['Total Deaths'] > 0)]['Total Deaths'].dropna().sort_values(ascending=False)
    
    if len(deaths_data) == 0:
        continue
    
    print(f"\n   {country}:")
    
    # Calculate return periods
    n = len(deaths_data)
    years_span = 25  # 2000-2025
    
    # Top events with estimated return periods
    top_events = deaths_data.head(10)
    for i, deaths in enumerate(top_events.values, 1):
        return_period = years_span * n / i
        print(f"      {deaths:5.0f} deaths → ~{return_period:.1f} year return period")

# ============================================================================
# VISUALIZATION: CDA SUMMARY PLOTS
# ============================================================================
print("\n" + "="*80)
print("GENERATING CDA VISUALIZATION PLOTS")
print("="*80)

# Plot 1: Hypothesis Test Results Summary
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CDA: Hypothesis Testing Results Summary', fontsize=14, fontweight='bold')

# 1. Deaths comparison with statistical annotation
ax1 = axes[0, 0]
bp1 = ax1.boxplot([bgd_deaths, tr_deaths], labels=['Bangladesh', 'Turkey'], 
                   patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor(colors['Bangladesh'])
bp1['boxes'][1].set_facecolor(colors['Turkey'])
ax1.set_ylabel('Total Deaths', fontweight='bold')
ax1.set_title('Mortality Comparison', fontweight='bold', pad=10)
ax1.text(0.5, 0.95, f'Mann-Whitney U: p={p_value:.4f}\nCohen\'s d={cohens_d:.3f}',
         transform=ax1.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.grid(True, alpha=0.3, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 2. Damage comparison
ax2 = axes[0, 1]
bp2 = ax2.boxplot([bgd_damage, tr_damage], labels=['Bangladesh', 'Turkey'],
                   patch_artist=True, widths=0.6)
bp2['boxes'][0].set_facecolor(colors['Bangladesh'])
bp2['boxes'][1].set_facecolor(colors['Turkey'])
ax2.set_ylabel('Economic Damage (\'000 US$)', fontweight='bold')
ax2.set_title('Economic Impact Comparison', fontweight='bold', pad=10)
ax2.set_yscale('log')
ax2.text(0.5, 0.95, f'Mann-Whitney U: p={p_value_dmg:.4f}',
         transform=ax2.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Severity index comparison
ax3 = axes[1, 0]
parts = ax3.violinplot([bgd_severity, tr_severity], positions=[0, 1], 
                       widths=0.7, showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors['Bangladesh'] if i == 0 else colors['Turkey'])
    pc.set_alpha(0.7)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Bangladesh', 'Turkey'])
ax3.set_ylabel('Composite Severity Index', fontweight='bold')
ax3.set_title('Severity Index Distribution', fontweight='bold', pad=10)
ax3.text(0.5, 0.95, f'Mann-Whitney U: p={p_value_sev:.4f}',
         transform=ax3.transAxes, ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax3.grid(True, alpha=0.3, axis='y')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 4. Temporal trend
ax4 = axes[1, 1]
for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_events[yearly_events['Country'] == country]
    years = country_data['Start Year'].values
    counts = country_data['Count'].values
    
    ax4.scatter(years, counts, color=colors[country], label=country, s=50, alpha=0.7)
    
    # Trend line
    slope, intercept, r_value, p_value_trend, std_err = stats.linregress(years, counts)
    trend_line = slope * years + intercept
    ax4.plot(years, trend_line, color=colors[country], linestyle='--', linewidth=2, alpha=0.8)

ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Number of Events', fontweight='bold')
ax4.set_title('Temporal Trend in Event Frequency', fontweight='bold', pad=10)
ax4.legend(framealpha=0.9)
ax4.grid(True, alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/cda_hypothesis_testing_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: plots/cda_hypothesis_testing_summary.png")

# Plot 2: Distribution Fitting Results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('CDA: Distribution Fitting Results (Q-Q Plots)', fontsize=14, fontweight='bold')

plot_idx = 0
for metric_name, metric_col in metrics.items():
    ax = axes.flat[plot_idx]
    plot_idx += 1
    
    for country in ['Bangladesh', 'Turkey']:
        raw_data = emdat_floods[(emdat_floods['Country'] == country) & 
                           (emdat_floods[metric_col] > 0)][metric_col].dropna().values
        
        # Remove outliers (IQR method)
        Q1 = np.percentile(raw_data, 25)
        Q3 = np.percentile(raw_data, 75)
        IQR = Q3 - Q1
        data = raw_data[(raw_data >= Q1 - 1.5 * IQR) & (raw_data <= Q3 + 1.5 * IQR)]
        
        if len(data) < 10:
            continue
        
        # Fit lognormal (best fit for most flood data)
        params = lognorm.fit(data)
        
        # Generate theoretical quantiles
        theoretical = lognorm.rvs(*params, size=len(data))
        
        # Q-Q plot
        stats.probplot(data, dist=lognorm, sparams=params, plot=ax, fit=False)
        ax.get_lines()[0].set_color(colors[country])
        ax.get_lines()[0].set_marker('o')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[0].set_label(country)
    
    ax.set_title(f'{metric_name} (Lognormal)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Theoretical Quantiles', fontsize=9)
    ax.set_ylabel('Sample Quantiles', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Remove extra subplots
for i in range(plot_idx, 6):
    fig.delaxes(axes.flat[i])

plt.tight_layout()
plt.savefig('plots/cda_distribution_fitting.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/cda_distribution_fitting.png")

# Plot 3: Extreme Value Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CDA: Extreme Value Analysis', fontsize=14, fontweight='bold')

# Return period plot
ax1 = axes[0]
for country in ['Bangladesh', 'Turkey']:
    deaths_data = emdat_floods[(emdat_floods['Country'] == country) & 
                               (emdat_floods['Total Deaths'] > 0)]['Total Deaths'].dropna().sort_values(ascending=False)
    
    n = len(deaths_data)
    years_span = 25
    
    # Calculate return periods for all events
    return_periods = [years_span * n / (i+1) for i in range(n)]
    
    ax1.scatter(return_periods, deaths_data.values, color=colors[country], 
               label=country, alpha=0.6, s=50)

ax1.set_xlabel('Return Period (years)', fontweight='bold')
ax1.set_ylabel('Deaths', fontweight='bold')
ax1.set_title('Return Period vs. Mortality', fontweight='bold', pad=10)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(framealpha=0.9)
ax1.grid(True, alpha=0.3, which='both')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Exceedance probability
ax2 = axes[1]
for country in ['Bangladesh', 'Turkey']:
    data = emdat_floods[(emdat_floods['Country'] == country) & 
                       (emdat_floods['Total Deaths'] > 0)]['Total Deaths'].dropna().sort_values()
    
    # Calculate exceedance probability
    n = len(data)
    exceedance_prob = [(n - i) / n for i in range(n)]
    
    ax2.plot(data.values, exceedance_prob, color=colors[country], 
            label=country, linewidth=2.5, marker='o', markersize=4, alpha=0.7)

ax2.set_xlabel('Deaths', fontweight='bold')
ax2.set_ylabel('Exceedance Probability', fontweight='bold')
ax2.set_title('Exceedance Probability Curve', fontweight='bold', pad=10)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(framealpha=0.9)
ax2.grid(True, alpha=0.3, which='both')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/cda_extreme_value_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: plots/cda_extreme_value_analysis.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("CDA SUMMARY: HYPOTHESIS TEST RESULTS")
print("="*80)

summary_data = {
    'Research Question': [
        'RQ1: Mortality Difference',
        'RQ1: Economic Damage Difference',
        'RQ1: Affected Population Difference',
        'RQ2: Severity Index Difference',
        'RQ3: Temporal Trend (Bangladesh)',
        'RQ3: Temporal Trend (Turkey)',
    ],
    'Test': [
        'Mann-Whitney U',
        'Mann-Whitney U',
        'Mann-Whitney U',
        'Mann-Whitney U',
        'Linear Regression',
        'Linear Regression',
    ],
    'p-value': [
        f'{p_value:.4f}',
        f'{p_value_dmg:.4f}',
        f'{p_value_aff:.4f}',
        f'{p_value_sev:.4f}',
        'See above',
        'See above',
    ],
    'Result': [
        'Significant' if p_value < 0.05 else 'Not Significant',
        'Significant' if p_value_dmg < 0.05 else 'Not Significant',
        'Significant' if p_value_aff < 0.05 else 'Not Significant',
        'Significant' if p_value_sev < 0.05 else 'Not Significant',
        'See above',
        'See above',
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "="*80)
print("CDA COMPLETE")
print("="*80)
print("\nGenerated 3 CDA visualization files:")
print("  1. cda_hypothesis_testing_summary.png")
print("  2. cda_distribution_fitting.png")
print("  3. cda_extreme_value_analysis.png")
print("\nAll statistical tests and confirmatory analyses completed.")
print("="*80)
