-- ============================================================
-- CABLE BAHAMAS BI ANALYTICS - SQLite Compatible Version
-- ============================================================

-- Clean base view
CREATE VIEW IF NOT EXISTS vw_clean_customer_base AS
SELECT DISTINCT *
FROM cable_bahamas_customer_data
WHERE customerID IS NOT NULL
  AND MonthlyCharges > 0;

-- Executive KPIs
CREATE VIEW IF NOT EXISTS vw_executive_kpis AS
SELECT 
    COUNT(DISTINCT customerID) as total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as churn_rate_percent,
    ROUND(SUM(MonthlyCharges), 2) as monthly_recurring_revenue,
    ROUND(AVG(MonthlyCharges), 2) as avg_revenue_per_user,
    ROUND(SUM(TotalCharges), 2) as total_revenue_lifetime,
    ROUND(SUM(RevenueAtRisk), 2) as total_revenue_at_risk,
    ROUND(SUM(EstimatedCLV), 2) as total_customer_lifetime_value,
    ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction_score,
    ROUND(AVG(PaymentReliabilityScore), 2) as avg_payment_reliability
FROM vw_clean_customer_base;

-- Regional Performance
CREATE VIEW IF NOT EXISTS vw_regional_performance AS
SELECT 
    Region,
    ServiceAvailability,
    COUNT(DISTINCT customerID) as total_customers,
    ROUND(CAST(COUNT(DISTINCT customerID) AS FLOAT) * 100.0 / 
          (SELECT COUNT(DISTINCT customerID) FROM vw_clean_customer_base), 2) as customer_share_percent,
    ROUND(SUM(MonthlyCharges), 2) as monthly_revenue,
    ROUND(AVG(MonthlyCharges), 2) as avg_revenue_per_customer,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
    ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction,
    ROUND(AVG(OutagesExperienced), 2) as avg_outages,
    ROUND(AVG(SupportTickets_LastYear), 2) as avg_support_tickets
FROM vw_clean_customer_base
GROUP BY Region, ServiceAvailability
ORDER BY monthly_revenue DESC;

-- Package Performance
CREATE VIEW IF NOT EXISTS vw_package_performance AS
SELECT 
    ServicePackage,
    COUNT(DISTINCT customerID) as total_customers,
    ROUND(CAST(COUNT(DISTINCT customerID) AS FLOAT) * 100.0 / 
          (SELECT COUNT(DISTINCT customerID) FROM vw_clean_customer_base), 2) as market_share_percent,
    ROUND(AVG(PackagePrice_BSD), 2) as avg_package_price,
    ROUND(SUM(MonthlyCharges), 2) as total_monthly_revenue,
    ROUND(AVG(tenure), 1) as avg_customer_tenure,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate,
    ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction,
    ROUND(AVG(EngagementScore), 2) as avg_engagement,
    ROUND(AVG(SupportTickets_LastYear), 2) as avg_support_tickets
FROM vw_clean_customer_base
GROUP BY ServicePackage
ORDER BY total_monthly_revenue DESC;

-- Tenure Cohorts
CREATE VIEW IF NOT EXISTS vw_tenure_cohorts AS
SELECT 
    CASE 
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        WHEN tenure <= 48 THEN '37-48 months'
        ELSE '48+ months'
    END as tenure_cohort,
    COUNT(DISTINCT customerID) as customer_count,
    ROUND(AVG(MonthlyCharges), 2) as avg_monthly_revenue,
    ROUND(AVG(TotalCharges), 2) as avg_total_revenue,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as churn_rate,
    ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction,
    ROUND(AVG(OnlineAccountLogins_LastMonth), 2) as avg_online_logins
FROM vw_clean_customer_base
GROUP BY 
    CASE 
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        WHEN tenure <= 48 THEN '37-48 months'
        ELSE '48+ months'
    END;

-- Churn Drivers
CREATE VIEW IF NOT EXISTS vw_churn_drivers AS
SELECT 
    'Contract Type' as driver_category,
    Contract as driver_value,
    COUNT(*) as total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as churn_rate
FROM vw_clean_customer_base
GROUP BY Contract

UNION ALL

SELECT 
    'Payment Method',
    PaymentMethod,
    COUNT(*),
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END),
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2)
FROM vw_clean_customer_base
GROUP BY PaymentMethod

UNION ALL

SELECT 
    'Internet Service',
    InternetService,
    COUNT(*),
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END),
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2)
FROM vw_clean_customer_base
GROUP BY InternetService;

-- High Risk Retention Targets
CREATE VIEW IF NOT EXISTS vw_high_risk_retention_targets AS
SELECT 
    customerID,
    Region,
    ServicePackage,
    tenure,
    MonthlyCharges,
    ChurnRiskCategory,
    SatisfactionScore,
    SupportTickets_LastYear,
    EngagementScore,
    EstimatedCLV,
    ROUND(MonthlyCharges * 12, 2) as annual_revenue_at_risk,
    PrimaryCompetitor,
    ReceivedCompetitiveOffer,
    CASE 
        WHEN SatisfactionScore < 5 AND SupportTickets_LastYear > 5 
            THEN 'Immediate Escalation - Service Quality Issues'
        WHEN ReceivedCompetitiveOffer = 'Yes' AND EstimatedCLV > 5000 
            THEN 'Counter-Offer Campaign'
        WHEN Contract = 'Month-to-month' AND tenure > 24 
            THEN 'Contract Upgrade Incentive'
        WHEN PaymentMethod = 'Electronic check' AND AvgPaymentDelayDays > 10 
            THEN 'Payment Method Migration'
        WHEN EngagementScore < 3 
            THEN 'Re-engagement Campaign'
        ELSE 'Standard Retention Campaign'
    END as recommended_action
FROM vw_clean_customer_base
WHERE ChurnRiskCategory IN ('High', 'Medium')
  AND Churn = 'No'
ORDER BY 
    CASE ChurnRiskCategory 
        WHEN 'High' THEN 1 
        ELSE 2 
    END,
    EstimatedCLV DESC
LIMIT 500;

-- Service Quality Metrics
CREATE VIEW IF NOT EXISTS vw_service_quality_metrics AS
SELECT 
    Region,
    ServiceAvailability,
    ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction,
    ROUND(AVG(OutagesExperienced), 2) as avg_outages,
    ROUND(AVG(SupportTickets_LastYear), 2) as avg_support_tickets,
    ROUND(AVG(OnlineAccountLogins_LastMonth), 2) as avg_online_logins,
    ROUND(AVG(EngagementScore), 2) as avg_engagement_score,
    COUNT(DISTINCT customerID) as total_customers,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) * 100, 2) as churn_rate,
    CASE 
        WHEN AVG(SatisfactionScore) >= 8 AND AVG(OutagesExperienced) < 2 THEN 'A - Excellent'
        WHEN AVG(SatisfactionScore) >= 7 AND AVG(OutagesExperienced) < 3 THEN 'B - Good'
        WHEN AVG(SatisfactionScore) >= 6 AND AVG(OutagesExperienced) < 5 THEN 'C - Average'
        WHEN AVG(SatisfactionScore) >= 5 THEN 'D - Below Average'
        ELSE 'F - Poor'
    END as service_quality_grade
FROM vw_clean_customer_base
GROUP BY Region, ServiceAvailability
ORDER BY avg_satisfaction DESC;

-- Customer Segments (Simplified for SQLite)
CREATE VIEW IF NOT EXISTS vw_customer_segments AS
SELECT 
    CASE 
        WHEN EstimatedCLV >= 8000 THEN 'High Value'
        WHEN EstimatedCLV >= 3000 THEN 'Medium Value'
        ELSE 'Low Value'
    END as value_segment,
    ChurnRiskCategory,
    COUNT(DISTINCT customerID) as customer_count,
    ROUND(AVG(MonthlyCharges), 2) as avg_monthly_revenue,
    ROUND(AVG(EstimatedCLV), 2) as avg_lifetime_value,
    ROUND(AVG(tenure), 1) as avg_tenure_months,
    ROUND(AVG(SatisfactionScore), 2) as avg_satisfaction,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_count,
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / COUNT(*), 2) as churn_rate
FROM vw_clean_customer_base
GROUP BY 
    CASE 
        WHEN EstimatedCLV >= 8000 THEN 'High Value'
        WHEN EstimatedCLV >= 3000 THEN 'Medium Value'
        ELSE 'Low Value'
    END,
    ChurnRiskCategory;

-- Competitive Analysis
CREATE VIEW IF NOT EXISTS vw_competitive_analysis AS
SELECT 
    Region,
    PrimaryCompetitor,
    COUNT(DISTINCT customerID) as customers_exposed,
    SUM(CASE WHEN ReceivedCompetitiveOffer = 'Yes' THEN 1 ELSE 0 END) as received_offers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_to_competitor,
    ROUND(CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) * 100.0 / 
          NULLIF(SUM(CASE WHEN ReceivedCompetitiveOffer = 'Yes' THEN 1 ELSE 0 END), 0), 2) as conversion_rate_percent,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges * 12 ELSE 0 END), 2) as annual_revenue_lost
FROM vw_clean_customer_base
WHERE PrimaryCompetitor != 'None'
GROUP BY Region, PrimaryCompetitor
ORDER BY annual_revenue_lost DESC;

-- Tableau Master View
CREATE VIEW IF NOT EXISTS vw_tableau_master AS
SELECT 
    customerID,
    Region,
    ServiceAvailability,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    ServicePackage,
    PackagePrice_BSD,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    CurrentDiscount_Percent,
    tenure,
    MonthlyCharges,
    TotalCharges,
    EstimatedCLV,
    RevenueAtRisk,
    SupportTickets_LastYear,
    OutagesExperienced,
    SatisfactionScore,
    PaymentReliabilityScore,
    OnlineAccountLogins_LastMonth,
    EngagementScore,
    PrimaryCompetitor,
    ReceivedCompetitiveOffer,
    SignupDate,
    SignupCohort,
    MonthsSinceLastChange,
    Churn,
    ChurnRiskCategory,
    CASE 
        WHEN tenure <= 12 THEN 'New (0-12m)'
        WHEN tenure <= 24 THEN 'Growing (13-24m)'
        WHEN tenure <= 48 THEN 'Mature (25-48m)'
        ELSE 'Loyal (48m+)'
    END as tenure_segment,
    CASE 
        WHEN MonthlyCharges < 50 THEN 'Low (<$50)'
        WHEN MonthlyCharges < 100 THEN 'Medium ($50-$100)'
        ELSE 'High ($100+)'
    END as revenue_segment,
    CASE 
        WHEN SatisfactionScore >= 8 THEN 'Promoter'
        WHEN SatisfactionScore >= 6 THEN 'Passive'
        ELSE 'Detractor'
    END as nps_category
FROM vw_clean_customer_base;