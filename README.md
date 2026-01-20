# Flood Risk Analysis Dashboard 🌊

A comprehensive interactive dashboard for analyzing flood risks, comparing historical data, and projecting future scenarios for **Bangladesh** and **Turkey**. This project combines statistical analysis, machine learning models, and interactive visualizations to provide insights for disaster risk management.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Data Sources](https://img.shields.io/badge/Data-EM--DAT%20%26%20Flood%20Observatory-blue)
![Analysis](https://img.shields.io/badge/Analysis-Python%20%28Pandas%2C%20Scikit--learn%29-orange)
![Techniques](https://img.shields.io/badge/Dashboard-HTML5%20%2F%20CSS3%20%2F%20Chart.js-yellow)

## 📊 Features

### 1. Interactive Risk Dashboard
- **Country Comparison**: Toggle between Bangladesh and Turkey.
- **Dynamic Filtering**: Filter by flood type (Riverine, Flash, Coastal).
- **Risk Indicators**: Analyze Economic Damage, Total Deaths, Affected Population, and Event Frequency.
- **Trend Analysis**: Visualize historical trends and future predictions.

### 2. Hazard Analysis Tab
- **Flood Statistics**: Frequency, Duration, Severity, and Area Affected.
- **Risk Zones**: Distribution of high, medium, and low-risk zones.
- **Timeline**: Historical event timeline with death toll overlay.
- **Dynamic Charts**: Doughnut charts for flood type distribution.

### 3. Cost-Benefit Analyzer (CBA) Tab
- **Investment Scenarios**: Compare Baseline, Aggressive, and Conservative investment strategies.
- **ROI Calculations**: Real-time Benefit-Cost Ratio and Break-even point analysis.
- **Financial Projections**: 10-25 year projections for investment vs. savings.
- **Mitigation Strategies**: Prioritized list of interventions (e.g., Early Warning Systems, Levees) with ROI estimates.

### 4. Advanced Analytics & Modeling
- **Predictive Scenarios**: Random Forest model predictions for future flood impacts.
- **Uncertainty Estimates**: 95% Confidence Intervals for damage and casualty projections.
- **Statistical Tests**: Mann-Whitney U tests and Cohen's d effect sizes for comparative analysis.

## 🚀 Getting Started

### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge).
- (Optional) Python 3.x if you want to run the analysis scripts.

### Running the Dashboard
You can run the dashboard using any simple local server.

**Using Python:**
1. Navigate to the project directory:
   ```bash
   cd flood_dashboard
   ```
2. Start the HTTP server:
   ```bash
   python3 -m http.server 8080
   ```
3. Open your browser and go to `http://localhost:8080`.

## 📂 Project Structure

```
├── flood_dashboard/        # Web Application Source
│   ├── index.html          # Main dashboard structure
│   ├── styles.css          # Modern, responsive styling
│   └── app.js              # Interactive logic and Chart.js integration
├── plots/                  # Generated analysis visualizations
├── cda.py                  # Confirmatory Data Analysis scripts
├── eda.py                  # Exploratory Data Analysis scripts
├── modeling.py             # Machine Learning modeling scripts
├── flood_risk_analysis_report.html # Detailed HTML report of the analysis
└── Data Files/             # Excel datasets (EMDAT, Flood Observatory)
```

## 🛠️ Technologies Used

- **Frontend**: HTML5, CSS3 (Flexbox/Grid), JavaScript (ES6+)
- **Visualization**: Chart.js for dynamic charts
- **Data Analysis**: Python (Pandas, NumPy, SciPy)
- **Machine Learning**: Scikit-learn (Random Forest, Linear Regression)

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---
*Developed for STAT-495 Statistical Analysis Project - January 2026*
