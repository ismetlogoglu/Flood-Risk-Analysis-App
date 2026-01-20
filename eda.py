"""
Exploratory Data Analysis - Data Structure & Head View
Displays head and structure info for all 4 datasets
"""

import pandas as pd
import numpy as np

# File paths
files = {
    "EMDAT Bangladesh": "EMDAT Bangladesh.xlsx",
    "EMDAT Turkey": "EMDAT Türkiye.xlsx",
    "Flood Observatory Bangladesh": "flood_observatory_bangladesh.xlsx",
    "Flood Observatory Turkey": "Turkey_flood_observatory.xlsx"
}

print("="*80)
print("DATA STRUCTURE & HEAD PREVIEW")
print("="*80)

for name, filepath in files.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"File: {filepath}")
    print(f"{'='*80}\n")
    
    try:
        # Read the Excel file
        df = pd.read_excel(filepath, engine="openpyxl")
        
        # Basic info
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
        
        # Column names and types
        print("COLUMN STRUCTURE:")
        print("-" * 60)
        for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"{i:2d}. {col:40s} | {str(dtype):15s} | NA: {null_count:4d} ({null_pct:5.1f}%)")
        
        # Head preview
        print(f"\n\nFIRST 5 ROWS:")
        print("-" * 80)
        print(df.head())
        
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    print("\n")

print("="*80)
print("DATA MERGING STRATEGY")
print("="*80)

# Load all datasets
emdat_bgd = pd.read_excel("EMDAT Bangladesh.xlsx", engine="openpyxl")
emdat_tr = pd.read_excel("EMDAT Türkiye.xlsx", engine="openpyxl")
flood_obs_bgd = pd.read_excel("flood_observatory_bangladesh.xlsx", engine="openpyxl")
flood_obs_tr = pd.read_excel("Turkey_flood_observatory.xlsx", engine="openpyxl")

# Add Country identifier to Flood Observatory data
flood_obs_bgd['Country'] = 'Bangladesh'
flood_obs_tr['Country'] = 'Turkey'

# Extract year from Flood Observatory 'Began' column for temporal analysis
flood_obs_bgd['Year'] = pd.to_datetime(flood_obs_bgd['Began']).dt.year
flood_obs_tr['Year'] = pd.to_datetime(flood_obs_tr['Began']).dt.year

# Calculate duration in days for Flood Observatory
flood_obs_bgd['Duration_Days'] = (flood_obs_tr['Ended'] - flood_obs_bgd['Began']).dt.days
flood_obs_tr['Duration_Days'] = (flood_obs_tr['Ended'] - flood_obs_tr['Began']).dt.days

print("\n1. MERGED EMDAT DATA (Both Countries)")
print("-" * 80)
# Add country source tag
emdat_bgd['Country'] = 'Bangladesh'
emdat_tr['Country'] = 'Turkey'
emdat_combined = pd.concat([emdat_bgd, emdat_tr], ignore_index=True)
print(f"Shape: {emdat_combined.shape[0]} rows x {emdat_combined.shape[1]} columns")
print(f"Bangladesh events: {len(emdat_bgd)}")
print(f"Turkey events: {len(emdat_tr)}")
print(f"\nYear range: {emdat_combined['Start Year'].min()} - {emdat_combined['Start Year'].max()}")
print("\nCountry distribution:")
print(emdat_combined['Country'].value_counts())

print("\n2. MERGED FLOOD OBSERVATORY DATA (Both Countries)")
print("-" * 80)
flood_obs_combined = pd.concat([flood_obs_bgd, flood_obs_tr], ignore_index=True)
print(f"Shape: {flood_obs_combined.shape[0]} rows x {flood_obs_combined.shape[1]} columns")
print(f"Bangladesh events: {len(flood_obs_bgd)}")
print(f"Turkey events: {len(flood_obs_tr)}")
print(f"\nYear range: {flood_obs_combined['Year'].min()} - {flood_obs_combined['Year'].max()}")
print("\nCountry distribution:")
print(flood_obs_combined['Country'].value_counts())

print("\n3. KEY METRICS COMPARISON")
print("-" * 80)
print("\nEMDAT Summary by Country:")
emdat_summary = emdat_combined.groupby('Country').agg({
    'DisNo.': 'count',
    'Total Deaths': lambda x: x.sum(skipna=True),
    'Total Affected': lambda x: x.sum(skipna=True),
    'Total Damage, Adjusted (\'000 US$)': lambda x: x.sum(skipna=True)
}).rename(columns={
    'DisNo.': 'Event_Count',
    'Total Deaths': 'Total_Deaths',
    'Total Affected': 'Total_Affected',
    'Total Damage, Adjusted (\'000 US$)': 'Total_Damage_Adjusted'
})
print(emdat_summary)

print("\nFlood Observatory Summary by Country:")
flood_obs_summary = flood_obs_combined.groupby('Country').agg({
    'ID': 'count',
    'Dead': 'sum',
    'Displaced': 'sum',
    'Area': 'sum',
    'Severity': 'mean'
}).rename(columns={
    'ID': 'Event_Count',
    'Dead': 'Total_Dead',
    'Displaced': 'Total_Displaced',
    'Area': 'Total_Area_sqkm',
    'Severity': 'Avg_Severity'
})
print(flood_obs_summary)

print("\n4. TEMPORAL OVERLAP ANALYSIS")
print("-" * 80)
print(f"\nEMDAT year range: {emdat_combined['Start Year'].min()} - {emdat_combined['Start Year'].max()}")
print(f"Flood Observatory year range: {flood_obs_combined['Year'].min()} - {flood_obs_combined['Year'].max()}")
print(f"\nOverlap period: {max(emdat_combined['Start Year'].min(), flood_obs_combined['Year'].min())} - {min(emdat_combined['Start Year'].max(), flood_obs_combined['Year'].max())}")

# Events per year comparison
print("\nEvents per year (recent 5 years):")
emdat_yearly = emdat_combined.groupby(['Country', 'Start Year']).size().reset_index(name='EMDAT_Count')
flood_yearly = flood_obs_combined.groupby(['Country', 'Year']).size().reset_index(name='FloodObs_Count')
flood_yearly.rename(columns={'Year': 'Start Year'}, inplace=True)

yearly_comparison = pd.merge(emdat_yearly, flood_yearly, on=['Country', 'Start Year'], how='outer').fillna(0)
yearly_comparison = yearly_comparison.sort_values(['Country', 'Start Year'], ascending=[True, False])
print("\nRecent years comparison:")
print(yearly_comparison.head(10))

print("\n5. POTENTIAL MATCHING CRITERIA")
print("-" * 80)
print("\nTo link EMDAT with Flood Observatory events, we could use:")
print("  • GlideNumber (when available)")
print("  • Year + Country match")
print("  • Geographic proximity (lat/long)")
print("  • Date overlap (Began/Ended vs Start/End)")
print("\nGlideNumber availability:")
print(f"  Flood Obs Bangladesh: {flood_obs_bgd['GlideNumber'].notna().sum()}/{len(flood_obs_bgd)} ({flood_obs_bgd['GlideNumber'].notna().sum()/len(flood_obs_bgd)*100:.1f}%)")
print(f"  Flood Obs Turkey: {flood_obs_tr['GlideNumber'].notna().sum()}/{len(flood_obs_tr)} ({flood_obs_tr['GlideNumber'].notna().sum()/len(flood_obs_tr)*100:.1f}%)")

print("\n6. MERGED DATASETS PREVIEW")
print("-" * 80)
print("\nEMDAT Combined (first 3 rows per country):")
print(emdat_combined.groupby('Country').head(3)[['DisNo.', 'Country', 'Start Year', 'Disaster Subtype', 'Total Deaths', 'Total Affected']])

print("\nFlood Observatory Combined (first 3 rows per country):")
print(flood_obs_combined.groupby('Country').head(3)[['ID', 'Country', 'Year', 'Dead', 'Displaced', 'Severity', 'MainCause']])

print("\n" + "="*80)
print("MERGING COMPLETE - Datasets ready for analysis")
print("="*80)

# ============================================================================
# EXPLORATORY DATA ANALYSIS - VISUALIZATIONS
# Research Questions Focused
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create plots directory
import os
os.makedirs('plots', exist_ok=True)

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
colors = {'Bangladesh': '#e74c3c', 'Turkey': '#3498db'}

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS - VISUALIZATIONS")
print("="*80)

# Filter only flood events for EMDAT
emdat_floods = emdat_combined[emdat_combined['Disaster Type'] == 'Flood'].copy()

# ============================================================================
# RQ1: COMPARATIVE RISK PROFILING
# ============================================================================
print("\n[RQ1] Comparative Risk Profiling Analysis...")

# 1.1 Mortality Comparison (Box plots)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('RQ1: Comparative Risk Profiling - Bangladesh vs Turkey', fontsize=15, fontweight='bold')

# Modern color palette
box_colors = {'Bangladesh': '#e74c3c', 'Turkey': '#3498db'}

# Deaths
deaths_data = emdat_floods[emdat_floods['Total Deaths'].notna()]
if len(deaths_data) > 0:
    bp1 = sns.boxplot(data=deaths_data, x='Country', y='Total Deaths', ax=axes[0], palette=box_colors)
    axes[0].set_yscale('log')
    axes[0].set_title('Mortality Distribution', fontsize=13, fontweight='bold', pad=12)
    axes[0].set_ylabel('Total Deaths (log scale)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Country', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Add median values as text
    for i, country in enumerate(['Bangladesh', 'Turkey']):
        median_val = deaths_data[deaths_data['Country'] == country]['Total Deaths'].median()
        axes[0].text(i, median_val, f'Med: {int(median_val)}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Affected Population
affected_data = emdat_floods[emdat_floods['Total Affected'].notna()]
if len(affected_data) > 0:
    bp2 = sns.boxplot(data=affected_data, x='Country', y='Total Affected', ax=axes[1], palette=box_colors)
    axes[1].set_yscale('log')
    axes[1].set_title('Affected Population', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_ylabel('Total Affected (log scale)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Country', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Add median values
    for i, country in enumerate(['Bangladesh', 'Turkey']):
        median_val = affected_data[affected_data['Country'] == country]['Total Affected'].median()
        if median_val >= 1000000:
            label = f'Med: {median_val/1000000:.1f}M'
        elif median_val >= 1000:
            label = f'Med: {median_val/1000:.0f}K'
        else:
            label = f'Med: {int(median_val)}'
        axes[1].text(i, median_val, label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Economic Damage
damage_data = emdat_floods[emdat_floods['Total Damage, Adjusted (\'000 US$)'].notna()]
if len(damage_data) > 0:
    bp3 = sns.boxplot(data=damage_data, x='Country', y='Total Damage, Adjusted (\'000 US$)', ax=axes[2], palette=box_colors)
    axes[2].set_yscale('log')
    axes[2].set_title('Economic Damage', fontsize=13, fontweight='bold', pad=12)
    axes[2].set_ylabel('Damage (\'000 US$, log scale)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Country', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    # Add median values (remember: values are in '000 US$)
    for i, country in enumerate(['Bangladesh', 'Turkey']):
        median_val = damage_data[damage_data['Country'] == country]['Total Damage, Adjusted (\'000 US$)'].median()
        # median_val is already in thousands, so divide by 1000 for millions
        if median_val >= 1000:  # >= 1,000,000 dollars
            label = f'Med: ${median_val/1000:.0f}M'
        else:  # < 1,000,000 dollars
            label = f'Med: ${median_val:.0f}K'
        axes[2].text(i, median_val, label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('plots/rq1_comparative_risk_profiling.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq1_comparative_risk_profiling.png")

# 1.2 Summary Statistics Table
print("\n   Summary Statistics by Country:")
print("   " + "-"*70)
for country in ['Bangladesh', 'Turkey']:
    country_data = emdat_floods[emdat_floods['Country'] == country]
    print(f"\n   {country}:")
    print(f"      Events: {len(country_data)}")
    print(f"      Deaths - Mean: {country_data['Total Deaths'].mean():.1f}, Median: {country_data['Total Deaths'].median():.1f}")
    print(f"      Affected - Mean: {country_data['Total Affected'].mean():.0f}, Median: {country_data['Total Affected'].median():.0f}")
    damage_col = 'Total Damage, Adjusted (\'000 US$)'
    print(f"      Damage - Mean: {country_data[damage_col].mean():.0f}, Median: {country_data[damage_col].median():.0f}")

# ============================================================================
# RQ2: STATISTICAL SIGNIFICANCE OF SEVERITY
# ============================================================================
print("\n[RQ2] Statistical Significance Testing...")

# Severity comparison using Flood Observatory data
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ2: Severity and Duration Statistical Analysis', fontsize=14, fontweight='bold')

# Severity distribution
sns.violinplot(data=flood_obs_combined, x='Country', y='Severity', ax=axes[0], palette=colors)
axes[0].set_title('Flood Severity Distribution')
axes[0].set_ylabel('Severity Score')

# Mann-Whitney U test for severity
bgd_severity = flood_obs_bgd['Severity'].dropna()
tr_severity = flood_obs_tr['Severity'].dropna()
stat_sev, p_sev = stats.mannwhitneyu(bgd_severity, tr_severity, alternative='two-sided')
axes[0].text(0.5, 0.95, f'Mann-Whitney U: p={p_sev:.4f}', 
             transform=axes[0].transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Duration comparison
duration_data = flood_obs_combined[flood_obs_combined['Duration_Days'].notna() & (flood_obs_combined['Duration_Days'] > 0)]
if len(duration_data) > 0:
    sns.boxplot(data=duration_data, x='Country', y='Duration_Days', ax=axes[1], palette=colors)
    axes[1].set_title('Flood Duration Distribution')
    axes[1].set_ylabel('Duration (days)')
    axes[1].set_yscale('log')
    
    # Mann-Whitney U test for duration
    bgd_dur = flood_obs_bgd[flood_obs_bgd['Duration_Days'] > 0]['Duration_Days'].dropna()
    tr_dur = flood_obs_tr[flood_obs_tr['Duration_Days'] > 0]['Duration_Days'].dropna()
    if len(bgd_dur) > 0 and len(tr_dur) > 0:
        stat_dur, p_dur = stats.mannwhitneyu(bgd_dur, tr_dur, alternative='two-sided')
        axes[1].text(0.5, 0.95, f'Mann-Whitney U: p={p_dur:.4f}', 
                     transform=axes[1].transAxes, ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/rq2_statistical_significance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq2_statistical_significance.png")
print(f"   Severity difference p-value: {p_sev:.4f} {'(Significant)' if p_sev < 0.05 else '(Not significant)'}")

# ============================================================================
# RQ3: TEMPORAL DYNAMICS AND STATIONARITY
# ============================================================================
print("\n[RQ3] Temporal Dynamics Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('RQ3: Temporal Dynamics (2000-2025)', fontsize=15, fontweight='bold')

# 3.1 Event frequency over time
yearly_events = emdat_floods.groupby(['Country', 'Start Year']).size().reset_index(name='Count')
for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_events[yearly_events['Country'] == country]
    axes[0, 0].plot(country_data['Start Year'], country_data['Count'], 
                    marker='o', label=country, color=colors[country], linewidth=2.5, markersize=6)
    # Add data labels on points
    for x, y in zip(country_data['Start Year'], country_data['Count']):
        if y > 3:  # Only show labels for higher values to avoid clutter
            axes[0, 0].text(x, y + 0.3, str(int(y)), ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[0, 0].set_title('Annual Flood Frequency', fontsize=13, fontweight='bold', pad=12)
axes[0, 0].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Number of Events', fontsize=11, fontweight='bold')
axes[0, 0].legend(framealpha=0.9, fontsize=10)
axes[0, 0].grid(True, alpha=0.3, linestyle='--')
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)

# 3.2 Mortality trend over time
yearly_deaths = emdat_floods.groupby(['Country', 'Start Year'])['Total Deaths'].sum().reset_index()
for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_deaths[yearly_deaths['Country'] == country]
    axes[0, 1].plot(country_data['Start Year'], country_data['Total Deaths'], 
                    marker='s', label=country, color=colors[country], linewidth=2.5, markersize=6)

axes[0, 1].set_title('Annual Flood Mortality', fontsize=13, fontweight='bold', pad=12)
axes[0, 1].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Total Deaths', fontsize=11, fontweight='bold')
axes[0, 1].legend(framealpha=0.9, fontsize=10)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)

# 3.3 Economic damage trend
yearly_damage = emdat_floods.groupby(['Country', 'Start Year'])['Total Damage, Adjusted (\'000 US$)'].sum().reset_index()
for country in ['Bangladesh', 'Turkey']:
    country_data = yearly_damage[yearly_damage['Country'] == country]
    axes[1, 0].plot(country_data['Start Year'], country_data['Total Damage, Adjusted (\'000 US$)'], 
                    marker='^', label=country, color=colors[country], linewidth=2.5, markersize=6)
axes[1, 0].set_title('Annual Economic Damage (Adjusted)', fontsize=13, fontweight='bold', pad=12)
axes[1, 0].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Total Damage (\'000 US$)', fontsize=11, fontweight='bold')
axes[1, 0].legend(framealpha=0.9, fontsize=10)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')
axes[1, 0].set_yscale('log')
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)

# 3.4 Rolling average (5-year window) for stationarity check
window = 5
for country in ['Bangladesh', 'Turkey']:
    country_yearly = yearly_events[yearly_events['Country'] == country].set_index('Start Year')
    if len(country_yearly) >= window:
        rolling_mean = country_yearly['Count'].rolling(window=window, center=True).mean()
        axes[1, 1].plot(rolling_mean.index, rolling_mean.values, 
                       label=f'{country} (5-yr avg)', color=colors[country], linewidth=3, alpha=0.8)
axes[1, 1].set_title(f'Frequency Trend (5-Year Moving Average)', fontsize=13, fontweight='bold', pad=12)
axes[1, 1].set_xlabel('Year', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Average Events per Year', fontsize=11, fontweight='bold')
axes[1, 1].legend(framealpha=0.9, fontsize=10)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].spines['top'].set_visible(False)
axes[1, 1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots/rq3_temporal_dynamics.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq3_temporal_dynamics.png")

# ============================================================================
# RQ4: DISTRIBUTIONAL MODEL SELECTION
# ============================================================================
print("\n[RQ4] Distribution Analysis (Tail Thickness)...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('RQ4: Loss Distribution Characteristics', fontsize=14, fontweight='bold')

metrics = [
    ('Total Deaths', 'Deaths'),
    ('Total Affected', 'Affected Population'),
    ('Total Damage, Adjusted (\'000 US$)', 'Economic Damage (\'000 US$)')
]

for idx, (col, title) in enumerate(metrics):
    # Histogram with KDE
    for country in ['Bangladesh', 'Turkey']:
        data = emdat_floods[(emdat_floods['Country'] == country) & (emdat_floods[col].notna()) & (emdat_floods[col] > 0)][col]
        if len(data) > 0:
            axes[0, idx].hist(np.log10(data), bins=15, alpha=0.6, label=country, color=colors[country], edgecolor='black')
    axes[0, idx].set_title(f'{title} Distribution (log10)')
    axes[0, idx].set_xlabel(f'log10({title})')
    axes[0, idx].set_ylabel('Frequency')
    axes[0, idx].legend()
    axes[0, idx].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check (combined)
    for country in ['Bangladesh', 'Turkey']:
        data = emdat_floods[(emdat_floods['Country'] == country) & (emdat_floods[col].notna()) & (emdat_floods[col] > 0)][col]
        if len(data) > 5:
            log_data = np.log10(data)
            stats.probplot(log_data, dist="norm", plot=axes[1, idx])
            axes[1, idx].get_lines()[0].set_color(colors[country])
            axes[1, idx].get_lines()[0].set_marker('o')
            axes[1, idx].get_lines()[0].set_markersize(4)
            axes[1, idx].get_lines()[0].set_alpha(0.6)
    axes[1, idx].set_title(f'Q-Q Plot: {title}')
    axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/rq4_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq4_distribution_analysis.png")

# Descriptive statistics for tail behavior
print("\n   Skewness and Kurtosis (indicators of tail thickness):")
print("   " + "-"*70)
for country in ['Bangladesh', 'Turkey']:
    print(f"\n   {country}:")
    for col, title in metrics:
        data = emdat_floods[(emdat_floods['Country'] == country) & (emdat_floods[col].notna()) & (emdat_floods[col] > 0)][col]
        if len(data) > 3:
            skew = stats.skew(data)
            kurt = stats.kurtosis(data)
            print(f"      {title}: Skewness={skew:.2f}, Kurtosis={kurt:.2f}")

# ============================================================================
# RQ5: EXTREME EVENT ESTIMATION (EVT)
# ============================================================================
print("\n[RQ5] Extreme Event Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('RQ5: Extreme Event Patterns', fontsize=14, fontweight='bold')

# 5.1 Top 10 deadliest events
top_deadly = emdat_floods.nlargest(10, 'Total Deaths')[['Country', 'Start Year', 'Total Deaths', 'Disaster Subtype']].copy()
top_deadly = top_deadly.reset_index(drop=True)
countries_top = top_deadly['Country'].values
colors_list = [colors[c] for c in countries_top]

bars1 = axes[0].barh(range(len(top_deadly)), top_deadly['Total Deaths'], color=colors_list, edgecolor='black', linewidth=1.2)
axes[0].set_yticks(range(len(top_deadly)))
axes[0].set_yticklabels([f"{row['Country'][:3]} {int(row['Start Year'])}" for _, row in top_deadly.iterrows()], fontsize=10)
axes[0].set_xlabel('Total Deaths', fontsize=11, fontweight='bold')
axes[0].set_title('Top 10 Deadliest Flood Events', fontsize=12, fontweight='bold', pad=15)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x', linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add value labels on bars
for idx, (bar, val) in enumerate(zip(bars1, top_deadly['Total Deaths'])):
    width = bar.get_width()
    axes[0].text(width + max(top_deadly['Total Deaths']) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{int(val):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')

# 5.2 Top 10 most damaging events
top_damage = emdat_floods.nlargest(10, 'Total Damage, Adjusted (\'000 US$)')[['Country', 'Start Year', 'Total Damage, Adjusted (\'000 US$)', 'Disaster Subtype']].copy()
top_damage = top_damage.reset_index(drop=True)
countries_dmg = top_damage['Country'].values
colors_list_dmg = [colors[c] for c in countries_dmg]

bars2 = axes[1].barh(range(len(top_damage)), top_damage['Total Damage, Adjusted (\'000 US$)'], 
                     color=colors_list_dmg, edgecolor='black', linewidth=1.2)
axes[1].set_yticks(range(len(top_damage)))
axes[1].set_yticklabels([f"{row['Country'][:3]} {int(row['Start Year'])}" for _, row in top_damage.iterrows()], fontsize=10)
axes[1].set_xlabel('Economic Damage (\'000 US$)', fontsize=11, fontweight='bold')
axes[1].set_title('Top 10 Most Damaging Flood Events', fontsize=12, fontweight='bold', pad=15)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x', linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add value labels on bars with K/M suffix (remember: values are in '000 US$)
for idx, (bar, val) in enumerate(zip(bars2, top_damage['Total Damage, Adjusted (\'000 US$)'])):
    width = bar.get_width()
    # val is already in thousands, so divide by 1000 for millions
    if val >= 1000:  # >= 1,000,000 dollars (1000 * 1000)
        label = f'${val/1000:.0f}M'
    else:  # < 1,000,000 dollars
        label = f'${val:.0f}K'
    axes[1].text(width + max(top_damage['Total Damage, Adjusted (\'000 US$)']) * 0.02, 
                bar.get_y() + bar.get_height()/2,
                label,
                ha='left', va='center', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors['Bangladesh'], edgecolor='black', label='Bangladesh'),
                   Patch(facecolor=colors['Turkey'], edgecolor='black', label='Turkey')]
axes[0].legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)
axes[1].legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig('plots/rq5_extreme_events.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq5_extreme_events.png")

# ============================================================================
# ADDITIONAL: COMPREHENSIVE OVERVIEW DASHBOARD
# ============================================================================
print("\n[BONUS] Creating comprehensive overview dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall event count by country
ax1 = fig.add_subplot(gs[0, 0])
country_counts = emdat_floods['Country'].value_counts()
ax1.bar(country_counts.index, country_counts.values, color=[colors[c] for c in country_counts.index])
ax1.set_title('Total Flood Events (2000-2025)')
ax1.set_ylabel('Count')
ax1.grid(True, alpha=0.3, axis='y')

# Disaster subtype distribution
ax2 = fig.add_subplot(gs[0, 1:])
subtype_country = emdat_floods.groupby(['Country', 'Disaster Subtype']).size().reset_index(name='Count')
subtype_pivot = subtype_country.pivot(index='Disaster Subtype', columns='Country', values='Count').fillna(0)
subtype_pivot.plot(kind='barh', ax=ax2, color=[colors['Bangladesh'], colors['Turkey']])
ax2.set_title('Flood Subtypes by Country')
ax2.set_xlabel('Number of Events')
ax2.legend(title='Country')
ax2.grid(True, alpha=0.3, axis='x')

# Monthly distribution
ax3 = fig.add_subplot(gs[1, 0])
monthly = emdat_floods.groupby(['Country', 'Start Month']).size().reset_index(name='Count')
for country in ['Bangladesh', 'Turkey']:
    country_data = monthly[monthly['Country'] == country]
    ax3.plot(country_data['Start Month'], country_data['Count'], 
             marker='o', label=country, color=colors[country], linewidth=2)
ax3.set_title('Seasonal Pattern (Month of Year)')
ax3.set_xlabel('Month')
ax3.set_ylabel('Event Count')
ax3.set_xticks(range(1, 13))
ax3.legend()
ax3.grid(True, alpha=0.3)

# Area affected (Flood Observatory)
ax4 = fig.add_subplot(gs[1, 1])
area_data = flood_obs_combined[flood_obs_combined['Area'] > 0]
sns.boxplot(data=area_data, x='Country', y='Area', ax=ax4, palette=colors)
ax4.set_yscale('log')
ax4.set_title('Affected Area (sq km, log scale)')
ax4.set_ylabel('Area (sq km)')
ax4.grid(True, alpha=0.3, axis='y')

# Casualties vs Damage scatter
ax5 = fig.add_subplot(gs[1, 2])
scatter_data = emdat_floods[(emdat_floods['Total Deaths'] > 0) & (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]
for country in ['Bangladesh', 'Turkey']:
    country_scatter = scatter_data[scatter_data['Country'] == country]
    ax5.scatter(country_scatter['Total Deaths'], 
                country_scatter['Total Damage, Adjusted (\'000 US$)'],
                alpha=0.6, label=country, color=colors[country], s=50)
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_xlabel('Total Deaths (log scale)')
ax5.set_ylabel('Economic Damage (\'000 US$, log)')
ax5.set_title('Mortality vs Economic Impact')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Cumulative events over time
ax6 = fig.add_subplot(gs[2, :])
for country in ['Bangladesh', 'Turkey']:
    country_data = emdat_floods[emdat_floods['Country'] == country].sort_values('Start Year')
    years = country_data['Start Year'].values
    cumulative = np.arange(1, len(years) + 1)
    ax6.plot(years, cumulative, label=country, color=colors[country], linewidth=2.5)
ax6.set_title('Cumulative Flood Events Over Time')
ax6.set_xlabel('Year')
ax6.set_ylabel('Cumulative Event Count')
ax6.legend()
ax6.grid(True, alpha=0.3)

fig.suptitle('Comprehensive EDA Dashboard: Bangladesh vs Turkey Flood Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('plots/comprehensive_eda_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/comprehensive_eda_dashboard.png")

# ============================================================================
# ADDITIONAL RQ-FOCUSED VISUALIZATIONS
# ============================================================================
print("\n[ADDITIONAL] Generating deeper RQ-focused analyses...")

# ============================================================================
# RQ1 EXTENDED: Impact Per Capita & Intensity Comparison
# ============================================================================
print("\n[RQ1+] Impact Intensity Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RQ1 Extended: Impact Intensity & Relationships', fontsize=14, fontweight='bold')

# Deaths per event comparison
deaths_per_event = emdat_floods.groupby('Country')['Total Deaths'].agg(['mean', 'median', 'std']).reset_index()
x = np.arange(len(deaths_per_event))
width = 0.25

bars1 = axes[0, 0].bar(x - width, deaths_per_event['mean'], width, label='Mean', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = axes[0, 0].bar(x, deaths_per_event['median'], width, label='Median', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars3 = axes[0, 0].bar(x + width, deaths_per_event['std'], width, label='Std Dev', color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.2)

axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(deaths_per_event['Country'], fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Deaths per Event', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Mortality Statistics per Event', fontsize=12, fontweight='bold', pad=12)
axes[0, 0].legend(framealpha=0.9, fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

# Damage per death ratio (economic efficiency)
damage_death_data = emdat_floods[(emdat_floods['Total Deaths'] > 0) & 
                                 (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)].copy()
damage_death_data['Damage_per_Death'] = damage_death_data['Total Damage, Adjusted (\'000 US$)'] / damage_death_data['Total Deaths']
sns.boxplot(data=damage_death_data, x='Country', y='Damage_per_Death', ax=axes[0, 1], palette=colors)
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Economic Loss per Casualty (log scale)')
axes[0, 1].set_ylabel('Damage per Death (\'000 US$ / death)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Affected vs Damage correlation
for country in ['Bangladesh', 'Turkey']:
    country_corr = emdat_floods[(emdat_floods['Country'] == country) & 
                                (emdat_floods['Total Affected'] > 0) & 
                                (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]
    if len(country_corr) > 0:
        axes[1, 0].scatter(country_corr['Total Affected'], 
                          country_corr['Total Damage, Adjusted (\'000 US$)'],
                          label=country, color=colors[country], alpha=0.6, s=60)
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].set_xlabel('Total Affected (log scale)')
axes[1, 0].set_ylabel('Economic Damage (\'000 US$, log)')
axes[1, 0].set_title('Population Impact vs Economic Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Event severity classification (combining multiple factors)
severity_data = emdat_floods[(emdat_floods['Total Deaths'].notna()) & 
                             (emdat_floods['Total Affected'].notna())].copy()
# Create severity score: normalized deaths + normalized affected
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
severity_data['Severity_Score'] = (
    scaler.fit_transform(severity_data[['Total Deaths']]) * 0.5 +
    scaler.fit_transform(severity_data[['Total Affected']]) * 0.5
).flatten()
severity_data['Severity_Category'] = pd.cut(severity_data['Severity_Score'], 
                                             bins=[0, 0.2, 0.5, 1.0],
                                             labels=['Low', 'Medium', 'High'])
severity_counts = severity_data.groupby(['Country', 'Severity_Category']).size().unstack(fill_value=0)
severity_counts.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#f39c12', '#e74c3c'], stacked=False)
axes[1, 1].set_title('Event Severity Distribution')
axes[1, 1].set_ylabel('Number of Events')
axes[1, 1].set_xlabel('Country')
axes[1, 1].legend(title='Severity')
axes[1, 1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('plots/rq1_extended_impact_intensity.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq1_extended_impact_intensity.png")

# ============================================================================
# RQ2 EXTENDED: Detailed Statistical Testing & Effect Sizes
# ============================================================================
print("\n[RQ2+] Advanced Statistical Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RQ2 Extended: Statistical Tests & Effect Sizes', fontsize=14, fontweight='bold')

# Cohen's d effect size for deaths
bgd_deaths = emdat_floods[(emdat_floods['Country'] == 'Bangladesh') & (emdat_floods['Total Deaths'].notna())]['Total Deaths']
tr_deaths = emdat_floods[(emdat_floods['Country'] == 'Turkey') & (emdat_floods['Total Deaths'].notna())]['Total Deaths']

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

if len(bgd_deaths) > 1 and len(tr_deaths) > 1:
    d_deaths = cohens_d(bgd_deaths, tr_deaths)
    stat_deaths, p_deaths = stats.mannwhitneyu(bgd_deaths, tr_deaths, alternative='two-sided')
    
    # Violin plot with test results
    deaths_combined = pd.DataFrame({
        'Deaths': np.concatenate([bgd_deaths, tr_deaths]),
        'Country': ['Bangladesh']*len(bgd_deaths) + ['Turkey']*len(tr_deaths)
    })
    sns.violinplot(data=deaths_combined, x='Country', y='Deaths', ax=axes[0, 0], palette=colors)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Mortality Distribution with Effect Size')
    axes[0, 0].text(0.5, 0.95, f"Cohen's d = {d_deaths:.3f}\nMann-Whitney p = {p_deaths:.4f}",
                   transform=axes[0, 0].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    axes[0, 0].grid(True, alpha=0.3, axis='y')

# Affected population comparison
bgd_affected = emdat_floods[(emdat_floods['Country'] == 'Bangladesh') & (emdat_floods['Total Affected'].notna())]['Total Affected']
tr_affected = emdat_floods[(emdat_floods['Country'] == 'Turkey') & (emdat_floods['Total Affected'].notna())]['Total Affected']

if len(bgd_affected) > 1 and len(tr_affected) > 1:
    d_affected = cohens_d(bgd_affected, tr_affected)
    stat_affected, p_affected = stats.mannwhitneyu(bgd_affected, tr_affected, alternative='two-sided')
    
    affected_combined = pd.DataFrame({
        'Affected': np.concatenate([bgd_affected, tr_affected]),
        'Country': ['Bangladesh']*len(bgd_affected) + ['Turkey']*len(tr_affected)
    })
    sns.violinplot(data=affected_combined, x='Country', y='Affected', ax=axes[0, 1], palette=colors)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Affected Population with Effect Size')
    axes[0, 1].text(0.5, 0.95, f"Cohen's d = {d_affected:.3f}\nMann-Whitney p = {p_affected:.4f}",
                   transform=axes[0, 1].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    axes[0, 1].grid(True, alpha=0.3, axis='y')

# Empirical CDF comparison for damage
damage_bgd = emdat_floods[(emdat_floods['Country'] == 'Bangladesh') & 
                          (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]['Total Damage, Adjusted (\'000 US$)'].sort_values()
damage_tr = emdat_floods[(emdat_floods['Country'] == 'Turkey') & 
                         (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]['Total Damage, Adjusted (\'000 US$)'].sort_values()

if len(damage_bgd) > 0:
    axes[1, 0].plot(damage_bgd, np.arange(1, len(damage_bgd)+1)/len(damage_bgd), 
                   label='Bangladesh', color=colors['Bangladesh'], linewidth=2)
if len(damage_tr) > 0:
    axes[1, 0].plot(damage_tr, np.arange(1, len(damage_tr)+1)/len(damage_tr),
                   label='Turkey', color=colors['Turkey'], linewidth=2)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('Economic Damage (\'000 US$, log scale)')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Empirical CDF: Economic Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# KS test for damage distributions
if len(damage_bgd) > 0 and len(damage_tr) > 0:
    ks_stat, ks_p = stats.ks_2samp(damage_bgd, damage_tr)
    axes[1, 0].text(0.5, 0.15, f'KS test: D={ks_stat:.3f}, p={ks_p:.4f}',
                   transform=axes[1, 0].transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Variance comparison (Levene's test)
metrics_var = []
for metric, col in [('Deaths', 'Total Deaths'), 
                    ('Affected', 'Total Affected'),
                    ('Damage', 'Total Damage, Adjusted (\'000 US$)')]:
    bgd_data = emdat_floods[(emdat_floods['Country'] == 'Bangladesh') & (emdat_floods[col].notna())][col]
    tr_data = emdat_floods[(emdat_floods['Country'] == 'Turkey') & (emdat_floods[col].notna())][col]
    if len(bgd_data) > 1 and len(tr_data) > 1:
        stat_var, p_var = stats.levene(bgd_data, tr_data)
        metrics_var.append({
            'Metric': metric,
            'Bangladesh Var': np.var(bgd_data),
            'Turkey Var': np.var(tr_data),
            'Variance Ratio': np.var(bgd_data) / np.var(tr_data) if np.var(tr_data) > 0 else np.nan,
            'Levene p': p_var
        })

var_df = pd.DataFrame(metrics_var)
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=var_df.values, colLabels=var_df.columns,
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.22, 0.22, 0.22, 0.19])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
axes[1, 1].set_title('Variance Homogeneity Tests (Levene)', pad=20)

plt.tight_layout()
plt.savefig('plots/rq2_extended_statistical_tests.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq2_extended_statistical_tests.png")

# ============================================================================
# RQ3 EXTENDED: Change Point Detection & Trend Analysis
# ============================================================================
print("\n[RQ3+] Change Point & Trend Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RQ3 Extended: Temporal Trends & Change Points', fontsize=14, fontweight='bold')

# Decade comparison
emdat_floods['Decade'] = (emdat_floods['Start Year'] // 10) * 10
decade_stats = emdat_floods.groupby(['Country', 'Decade']).agg({
    'DisNo.': 'count',
    'Total Deaths': 'sum',
    'Total Affected': 'sum'
}).reset_index()
decade_stats.columns = ['Country', 'Decade', 'Events', 'Deaths', 'Affected']

decade_pivot = decade_stats.pivot(index='Decade', columns='Country', values='Events').fillna(0)
decade_pivot.plot(kind='bar', ax=axes[0, 0], color=[colors['Bangladesh'], colors['Turkey']])
axes[0, 0].set_title('Event Frequency by Decade')
axes[0, 0].set_xlabel('Decade')
axes[0, 0].set_ylabel('Number of Events')
axes[0, 0].legend(title='Country')
axes[0, 0].grid(True, alpha=0.3, axis='y')
plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

# Inter-event time analysis (return periods)
for country in ['Bangladesh', 'Turkey']:
    country_events = emdat_floods[emdat_floods['Country'] == country].sort_values('Start Year')
    if len(country_events) > 1:
        years = country_events['Start Year'].values
        inter_event_times = np.diff(years)
        if len(inter_event_times) > 0:
            axes[0, 1].hist(inter_event_times, bins=15, alpha=0.6, label=country, 
                           color=colors[country], edgecolor='black')

axes[0, 1].set_xlabel('Years Between Events')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Inter-Event Time Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Moving average with confidence bands
window = 3
for country in ['Bangladesh', 'Turkey']:
    country_yearly = emdat_floods[emdat_floods['Country'] == country].groupby('Start Year').size().reset_index(name='Count')
    country_yearly = country_yearly.set_index('Start Year').reindex(range(2000, 2026), fill_value=0).reset_index()
    country_yearly.columns = ['Year', 'Count']
    
    rolling_mean = country_yearly['Count'].rolling(window=window, center=True).mean()
    rolling_std = country_yearly['Count'].rolling(window=window, center=True).std()
    
    axes[1, 0].plot(country_yearly['Year'], rolling_mean, label=f'{country} (MA-{window})',
                   color=colors[country], linewidth=2.5)
    axes[1, 0].fill_between(country_yearly['Year'], 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std,
                           alpha=0.2, color=colors[country])

axes[1, 0].set_title(f'Event Frequency Trend with ±1 SD Band')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Events per Year')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cumulative impact over time (deaths)
for country in ['Bangladesh', 'Turkey']:
    country_yearly = emdat_floods[emdat_floods['Country'] == country].groupby('Start Year')['Total Deaths'].sum().reset_index()
    country_yearly['Cumulative_Deaths'] = country_yearly['Total Deaths'].cumsum()
    axes[1, 1].plot(country_yearly['Start Year'], country_yearly['Cumulative_Deaths'],
                   label=country, color=colors[country], linewidth=2.5, marker='o', markersize=4)

axes[1, 1].set_title('Cumulative Mortality Over Time')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Cumulative Deaths')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/rq3_extended_temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq3_extended_temporal_analysis.png")

# ============================================================================
# RQ4 EXTENDED: Distribution Fitting & Goodness-of-Fit
# ============================================================================
print("\n[RQ4+] Distribution Fitting Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('RQ4 Extended: Distribution Fitting & Tail Analysis', fontsize=14, fontweight='bold')

from scipy.stats import lognorm, weibull_min, expon, gamma

metrics_fit = [
    ('Total Deaths', 'Deaths'),
    ('Total Affected', 'Affected'),
    ('Total Damage, Adjusted (\'000 US$)', 'Damage')
]

for idx, (col, title) in enumerate(metrics_fit):
    # For each country
    for jdx, country in enumerate(['Bangladesh', 'Turkey']):
        data = emdat_floods[(emdat_floods['Country'] == country) & 
                           (emdat_floods[col].notna()) & 
                           (emdat_floods[col] > 0)][col].values
        
        if len(data) > 10:
            # Fit distributions
            fits = {}
            for dist_name, dist in [('Lognormal', lognorm), ('Weibull', weibull_min), 
                                    ('Exponential', expon), ('Gamma', gamma)]:
                try:
                    params = dist.fit(data)
                    fits[dist_name] = {'params': params, 'dist': dist}
                except:
                    pass
            
            # Plot histogram
            axes[jdx, idx].hist(data, bins=20, density=True, alpha=0.6, 
                               color=colors[country], edgecolor='black', label='Data')
            
            # Plot fitted distributions
            x_range = np.linspace(data.min(), data.max(), 200)
            for dist_name, fit_info in fits.items():
                pdf = fit_info['dist'].pdf(x_range, *fit_info['params'])
                axes[jdx, idx].plot(x_range, pdf, linewidth=2, label=dist_name)
            
            axes[jdx, idx].set_xlabel(title)
            axes[jdx, idx].set_ylabel('Density')
            axes[jdx, idx].set_title(f'{country} - {title}')
            axes[jdx, idx].legend(fontsize=8)
            axes[jdx, idx].set_xscale('log')
            axes[jdx, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/rq4_extended_distribution_fitting.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq4_extended_distribution_fitting.png")

# ============================================================================
# RQ5 EXTENDED: Return Level Plots & Exceedance Probabilities
# ============================================================================
print("\n[RQ5+] Extreme Value Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RQ5 Extended: Return Levels & Exceedance Analysis', fontsize=14, fontweight='bold')

# Exceedance probability plot for deaths
thresholds = np.logspace(0, 4, 50)
for country in ['Bangladesh', 'Turkey']:
    deaths = emdat_floods[(emdat_floods['Country'] == country) & 
                         (emdat_floods['Total Deaths'].notna()) & 
                         (emdat_floods['Total Deaths'] > 0)]['Total Deaths'].values
    if len(deaths) > 0:
        exceedance_probs = [np.mean(deaths > t) for t in thresholds]
        axes[0, 0].plot(thresholds, exceedance_probs, label=country, 
                       color=colors[country], linewidth=2.5, marker='o', markersize=3)

axes[0, 0].set_xscale('log')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Death Threshold (log scale)')
axes[0, 0].set_ylabel('Exceedance Probability (log scale)')
axes[0, 0].set_title('Death Exceedance Probability')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Return period estimation (simple empirical)
for country in ['Bangladesh', 'Turkey']:
    damage = emdat_floods[(emdat_floods['Country'] == country) & 
                         (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]['Total Damage, Adjusted (\'000 US$)'].sort_values(ascending=False).values
    if len(damage) > 0:
        n = len(damage)
        return_periods = n / np.arange(1, n+1)
        axes[0, 1].scatter(return_periods, damage, label=country, 
                          color=colors[country], alpha=0.7, s=50)

axes[0, 1].set_xscale('log')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlabel('Return Period (years)')
axes[0, 1].set_ylabel('Economic Damage (\'000 US$)')
axes[0, 1].set_title('Empirical Return Level: Damage')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Mean excess plot (for threshold selection in GPD)
for country in ['Bangladesh', 'Turkey']:
    deaths = emdat_floods[(emdat_floods['Country'] == country) & 
                         (emdat_floods['Total Deaths'] > 0)]['Total Deaths'].sort_values().values
    if len(deaths) > 10:
        thresholds_me = deaths[:-5]  # Leave some data above each threshold
        mean_excess = []
        for u in thresholds_me:
            excesses = deaths[deaths > u] - u
            if len(excesses) > 0:
                mean_excess.append(np.mean(excesses))
            else:
                mean_excess.append(np.nan)
        axes[1, 0].plot(thresholds_me, mean_excess, label=country, 
                       color=colors[country], linewidth=2, marker='o', markersize=3)

axes[1, 0].set_xlabel('Threshold')
axes[1, 0].set_ylabel('Mean Excess')
axes[1, 0].set_title('Mean Excess Plot (GPD Threshold Selection)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Percentile comparison
percentiles = [50, 75, 90, 95, 99]
perc_data = []
for country in ['Bangladesh', 'Turkey']:
    damage = emdat_floods[(emdat_floods['Country'] == country) & 
                         (emdat_floods['Total Damage, Adjusted (\'000 US$)'] > 0)]['Total Damage, Adjusted (\'000 US$)'].values
    if len(damage) > 0:
        percs = np.percentile(damage, percentiles)
        for p, val in zip(percentiles, percs):
            perc_data.append({'Country': country, 'Percentile': f'{p}th', 'Value': val})

perc_df = pd.DataFrame(perc_data)
perc_pivot = perc_df.pivot(index='Percentile', columns='Country', values='Value')
perc_pivot.plot(kind='bar', ax=axes[1, 1], color=[colors['Bangladesh'], colors['Turkey']])
axes[1, 1].set_title('Damage Percentiles Comparison')
axes[1, 1].set_ylabel('Damage (\'000 US$)')
axes[1, 1].set_xlabel('Percentile')
axes[1, 1].legend(title='Country')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('plots/rq5_extended_extreme_value_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: plots/rq5_extended_extreme_value_analysis.png")

print("\n" + "="*80)
print("EDA VISUALIZATION COMPLETE")
print("="*80)
print(f"\nGenerated {11} comprehensive visualization files in 'plots/' directory:")
print("  1. rq1_comparative_risk_profiling.png")
print("  2. rq1_extended_impact_intensity.png  [NEW]")
print("  3. rq2_statistical_significance.png")
print("  4. rq2_extended_statistical_tests.png  [NEW]")
print("  5. rq3_temporal_dynamics.png")
print("  6. rq3_extended_temporal_analysis.png  [NEW]")
print("  7. rq4_distribution_analysis.png")
print("  8. rq4_extended_distribution_fitting.png  [NEW]")
print("  9. rq5_extreme_events.png")
print(" 10. rq5_extended_extreme_value_analysis.png  [NEW]")
print(" 11. comprehensive_eda_dashboard.png")
print("\nAll visualizations support the 5 research questions with extended analyses.")
print("="*80)
