
# 📊 Bahamas Telecom Analytics

> End-to-end customer analytics platform for telecommunications churn prediction, retention optimization, and revenue forecasting in the Bahamas market

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tableau](https://img.shields.io/badge/Tableau-2023+-orange.svg)](https://www.tableau.com/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![Project Banner](https://via.placeholder.com/1200x300/1e3a8a/ffffff?text=Bahamas+Telecom+Analytics)

---

## 🎯 Project Overview

A comprehensive analytics solution designed to help telecom providers reduce customer churn, optimize revenue streams, and improve service delivery across the Bahamas. This project combines predictive modeling, SQL analytics, and interactive Tableau dashboards to deliver actionable business insights.

### 📈 At a Glance

| Metric | Value | Insight |
|--------|-------|---------|
| 👥 **Total Customers** | 7,043 | Across 6 regions |
| 💰 **Monthly Recurring Revenue** | $456K | Active revenue stream |
| ⚠️ **Revenue at Risk** | $1.67M | From potential churn |
| 🎯 **High-Risk Customers** | 501 | Requiring immediate action |
| 📉 **Current Churn Rate** | 26.54% | Industry benchmark: Unknown |

---

## ✨ Key Features

### 📊 **Interactive Dashboards**
- Executive KPI scorecards with real-time metrics
- Customer segmentation by value and risk
- Package performance and profitability analysis
- Regional operations and service quality tracking
- High-risk customer identification and targeting

### 🤖 **Predictive Analytics**
- Churn risk classification (High/Medium/Low)
- Customer lifetime value (CLV) estimation
- 12-month revenue forecasting
- Campaign ROI prediction and optimization

### 💼 **Business Intelligence**
- Market opportunity identification
- Competitive threat assessment
- Service quality impact analysis
- Data-driven retention strategies

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Tableau Desktop 2023.1+
SQLite3
```

### Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/jromer242/bahamas-telecom-analytics.git
cd bahamas-telecom-analytics

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run setup script (creates database & generates views)
python main.py
```

This will:
- ✅ Create the SQLite database from CSV data
- ✅ Execute SQL transformations
- ✅ Generate Tableau-ready views
- ✅ Export analytical datasets

### Usage

**For Data Analysis:**
```bash
# Run the main pipeline
python main.py

# Execute specific SQL queries
python sql_exec_script.py

# Open Jupyter notebook for exploration
jupyter notebook python/notebook.ipynb
```

**For Tableau Dashboards:**
1. Navigate to root folder
2. Open `.twb` or `.twbx` files in Tableau Desktop
3. Connect to any of the `views` csv in the root folder
4. Refresh data sources and explore!

---

## 📁 Project Structure

```
bahamas-telecom-analytics/
│
├── 📄 main.py                    # Main execution script (START HERE)
├── 📄 data_dictionary.csv        # Field definitions & metadata
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project documentation
├── cable_bahamas.db                  # SQLite database (generated)
│
├── 📂 python/                    # Core analytics engine       
│   ├── sql_exec_script.py       # SQL query executor
│   ├── predictive_analytics.py  # ML models & predictions
│   └── eda.ipynb                # Additional EDA notebooks
│
├── 📂 sql/                       # SQL transformation scripts
│   ├── create_views.sql         # Business intelligence views
│   └── [additional queries]
│
├── 📂 data/ (generated)          # Output datasets
│   ├── vw_tableau_master.csv           # Main Tableau source
│   ├── vw_high_risk_retention_targets.csv
│   ├── vw_package_performance.csv
│   ├── vw_customer_segments.csv
│   ├── vw_regional_performance.csv
│   ├── vw_executive_kpis.csv
│   ├── market_opportunities.csv
│   ├── retention_campaign_roi.csv
│   └── revenue_forecast_12m.csv
│
└── 📂 tableau/                   # Tableau workbooks
    └── [.twb/.twbx files]
```

---

## 🔍 Key Insights & Findings

### 📊 Customer Segmentation

| Segment | Count | Avg CLV | Churn Rate | Status |
|---------|-------|---------|------------|--------|
| **High Value / High Risk** | 9 | $8,523 | 11.1% | 🔴 Critical |
| **High Value / Low Risk** | 619 | $9,416 | 0.0% | 🟢 Healthy |
| **Medium Value / High Risk** | 252 | $4,702 | 73.4% | 🔴 **Urgent** |
| **Low Value / High Risk** | 1,610 | $838 | **79.1%** | ⚠️ High Volume |

**Key Finding:** Medium & Low Value High-Risk segments account for 1,862 customers with 75%+ churn rates - primary intervention target.

---

### 📦 Package Performance

| Package | Market Share | Churn Rate | Satisfaction | Revenue/Month |
|---------|--------------|------------|--------------|---------------|
| Triple Play | 27.6% | 29.4% | 6.74⭐ | $180,881 |
| Internet Only | **28.7%** | **34.4%** ⚠️ | 6.55⭐ | $122,905 |
| Internet Premium | 22.1% | 31.4% | 6.65⭐ | $120,165 |
| Basic Cable | 21.7% | **7.4%** ✅ | **7.24⭐** | $32,167 |

**Insight:** Basic Cable has the lowest churn (7.4%) and highest satisfaction (7.24) despite lower revenue. Internet Only has highest churn (34.4%) - needs immediate attention.

---

### 🗺️ Regional Analysis

| Region | Service Level | Customers | Churn Rate | Avg Outages | Satisfaction |
|--------|---------------|-----------|------------|-------------|--------------|
| Nassau | Full | 3,135 (44.5%) | 26.2% | 1.47 | 6.78⭐ |
| Freeport | Full | 1,766 (25.1%) | 26.3% | 2.00 | 6.79⭐ |
| Abaco | Partial | 703 (10.0%) | 25.8% | 3.51 | 6.76⭐ |
| Eleuthera | Partial | 607 (8.6%) | **30.0%** ⚠️ | 2.92 | 6.75⭐ |
| Exuma | Limited | 483 (6.9%) | 26.1% | **3.99** | 6.73⭐ |
| Other Islands | Limited | 349 (5.0%) | 26.9% | **4.99** | 6.79⭐ |

**Correlation:** Limited service areas experience 2-3x more outages, yet satisfaction remains stable. Eleuthera shows highest churn

---

## 🎯 Business Recommendations

### Immediate Actions (0-30 days)

1. **High-Risk Customer Outreach**
   - Target 501 high-risk customers identified in retention dataset
   - Focus on Medium/High Value segments (261 customers, $1.23M at risk)
   - Implement recommended actions: contract upgrades, payment method migration

2. **Internet Only Package Optimization**
   - Address 34.4% churn rate in largest segment (28.7% market share)
   - Investigate Basic Cable's success factors (7.4% churn, 7.24 satisfaction)
   - Consider bundling strategies or pricing adjustments

3. **Service Quality Improvements**
   - Prioritize Eleuthera region (30% churn)
   - Address outage frequency in Limited service areas
   - Deploy proactive support for customers with 3+ tickets

### Strategic Initiatives (30-90 days)

4. **Revenue Optimization**
   - Implement 12-month revenue forecast models
   - Test retention campaign strategies (estimated ROI available)
   - Focus on converting Month-to-Month to annual contracts

5. **Market Expansion**
   - Leverage market opportunities dataset
   - Expand Full service to Partial/Limited regions
   - Target competitor customers with received offers

6. **Data-Driven Culture**
   - Deploy Tableau dashboards to operations teams
   - Establish weekly churn review meetings
   - Create automated alerts for high-risk customers

---

## 🛠️ Technical Details

### Data Pipeline

```
Raw CSV Data → SQLite Database → SQL Transformations → Analytics Views → Tableau Dashboards
     ↓              ↓                    ↓                    ↓                ↓
  main.py    cable_bahamas.db      sql/*.sql               vw_*.csv      tableau/*.twb
```

### Key SQL Views

- **`vw_tableau_master`** - Enriched customer dataset with all features
- **`vw_customer_segments`** - Value-based cohort analysis
- **`vw_high_risk_retention_targets`** - Actionable customer list
- **`vw_executive_kpis`** - Top-level business metrics
- **`vw_package_performance`** - Service tier profitability
- **`vw_regional_performance`** - Geographic breakdowns

### Predictive Models

- **Churn Classification:** Multi-class classifier (High/Medium/Low risk)
- **CLV Estimation:** Regression model based on tenure, package, and engagement
- **Revenue Forecasting:** Time-series analysis with 12-month horizon
- **Campaign ROI:** Expected value calculation per retention strategy

---

## 📊 Sample Visualizations

### Executive Dashboard
![Executive KPI Dashboard](https://via.placeholder.com/800x450/0f172a/ffffff?text=Executive+Dashboard+Preview)

### Customer Segmentation
![Customer Segments](https://via.placeholder.com/800x450/1e3a8a/ffffff?text=Customer+Segmentation+Matrix)

### Regional Performance
![Regional Map](https://via.placeholder.com/800x450/075985/ffffff?text=Bahamas+Regional+Performance)

---

## 🤝 Contributing

This is a portfolio/demonstration project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Jose Romer**
- GitHub: [@jromer242](https://github.com/jromer242)
- LinkedIn: [Jyles Romer](https://linkedin.com/in/jylesromer)

---

## 🙏 Acknowledgments

- Telecommunications industry benchmarks from [Industry Source]
- Bahamas geographic data from [Data Source]
- Inspired by real-world telecom churn challenges

---

## 📫 Contact

Questions? Feedback? Want to discuss telecom analytics?

- 📧 Email: [your.email@example.com]
- 💼 LinkedIn: [Your LinkedIn]
- 🐦 Twitter: [@yourhandle]

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ and lots of ☕ by Jose Romer

</div>