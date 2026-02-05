"""
Cable Bahamas - Forecasting & Predictive Analytics Module
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CableBahamasPredictiveAnalytics:
    """
    Advanced analytics suite for customer retention, revenue forecasting,
    and market opportunity assessment
    """
    
    def __init__(self, data_path='../data/cable_bahamas_enriched.csv'):
        self.df = pd.read_csv(data_path)
        self.churn_model = None
        self.feature_importance = None
        self.le_dict = {}
        
    def prepare_features_for_modeling(self):
        """Prepare dataset for machine learning models"""
        print("Preparing features for predictive modeling...")
        
        # Select features for churn prediction
        feature_cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'PaymentMethod', 'InternetService',
            'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'PaperlessBilling', 'SeniorCitizen', 'Partner', 'Dependents',
            'Region', 'ServicePackage', 'PrimaryCompetitor',
            'SupportTickets_LastYear', 'OutagesExperienced', 
            'SatisfactionScore', 'PaymentReliabilityScore',
            'OnlineAccountLogins_LastMonth', 'EngagementScore',
            'CurrentDiscount_Percent', 'ReceivedCompetitiveOffer'
        ]
        
        # Create modeling dataset
        model_df = self.df[feature_cols + ['Churn']].copy()
        
        # Handle missing values
        model_df['TotalCharges'].fillna(model_df['MonthlyCharges'], inplace=True)
        
        # Encode categorical variables
        categorical_cols = model_df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        for col in categorical_cols:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            self.le_dict[col] = le
        
        # Encode target
        model_df['Churn'] = (model_df['Churn'] == 'Yes').astype(int)
        
        print(f"✓ Prepared {len(model_df)} records with {len(feature_cols)} features")
        return model_df, feature_cols
    
    def build_churn_prediction_model(self):
        """Build and evaluate churn prediction model"""
        print("\n" + "="*60)
        print("CHURN PREDICTION MODEL")
        print("="*60)
        
        # Prepare data
        model_df, feature_cols = self.prepare_features_for_modeling()
        
        X = model_df[feature_cols]
        y = model_df['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} | Test set: {len(X_test)}")
        print(f"Churn rate in training: {y_train.mean():.1%}")
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        self.churn_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42,
            class_weight='balanced'
        )
        
        self.churn_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.churn_model.predict(X_test)
        y_pred_proba = self.churn_model.predict_proba(X_test)[:, 1]
        
        # Evaluation metrics
        print("\n" + "-"*60)
        print("MODEL PERFORMANCE")
        print("-"*60)
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
        
        roc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_score:.3f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.churn_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "-"*60)
        print("TOP 10 CHURN DRIVERS")
        print("-"*60)
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"{row['Feature']:.<40} {row['Importance']:.3f}")
        
        # Save predictions back to original dataset
        self.df['ChurnProbability'] = self.churn_model.predict_proba(X)[:, 1]
        
        return self.churn_model, self.feature_importance
    
    def revenue_forecasting(self, months_ahead=12):
        """Forecast revenue based on cohort trends and churn predictions"""
        print("\n" + "="*60)
        print(f"REVENUE FORECASTING - {months_ahead} MONTHS AHEAD")
        print("="*60)
        
        # Current MRR
        current_mrr = self.df[self.df['Churn'] == 'No']['MonthlyCharges'].sum()
        
        # Average churn rate
        avg_churn_rate = (self.df['Churn'] == 'Yes').mean()
        
        # Average new customer MRR (from recent cohorts)
        recent_cohorts = self.df.sort_values('SignupCohort', ascending=False).head(1000)
        avg_new_customer_revenue = recent_cohorts['MonthlyCharges'].mean()
        
        # Forecast scenarios
        forecasts = {
            'Month': [],
            'Conservative_MRR': [],
            'Expected_MRR': [],
            'Optimistic_MRR': []
        }
        
        for month in range(1, months_ahead + 1):
            # Conservative: High churn, low acquisition
            conservative_churn = avg_churn_rate * 1.2
            conservative_new = 100
            conservative_mrr = current_mrr * ((1 - conservative_churn) ** month) + \
                             (conservative_new * avg_new_customer_revenue * month)
            
            # Expected: Current trends
            expected_churn = avg_churn_rate
            expected_new = 150
            expected_mrr = current_mrr * ((1 - expected_churn) ** month) + \
                          (expected_new * avg_new_customer_revenue * month)
            
            # Optimistic: Lower churn (retention campaigns), higher acquisition
            optimistic_churn = avg_churn_rate * 0.8
            optimistic_new = 200
            optimistic_mrr = current_mrr * ((1 - optimistic_churn) ** month) + \
                           (optimistic_new * avg_new_customer_revenue * month)
            
            forecasts['Month'].append(month)
            forecasts['Conservative_MRR'].append(conservative_mrr)
            forecasts['Expected_MRR'].append(expected_mrr)
            forecasts['Optimistic_MRR'].append(optimistic_mrr)
        
        forecast_df = pd.DataFrame(forecasts)
        
        print(f"\nCurrent MRR: ${current_mrr:,.2f}")
        print(f"\nProjected MRR in {months_ahead} months:")
        print(f"  Conservative: ${forecast_df['Conservative_MRR'].iloc[-1]:,.2f}")
        print(f"  Expected:     ${forecast_df['Expected_MRR'].iloc[-1]:,.2f}")
        print(f"  Optimistic:   ${forecast_df['Optimistic_MRR'].iloc[-1]:,.2f}")
        
        return forecast_df
    
    def market_opportunity_analysis(self):
        """Analyze market opportunities by region and service"""
        print("\n" + "="*60)
        print("MARKET OPPORTUNITY ANALYSIS")
        print("="*60)
        
        # Calculate market penetration by region
        regional_analysis = self.df.groupby('Region').agg({
            'customerID': 'count',
            'MonthlyCharges': 'sum',
            'ServiceAvailability': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'Churn': lambda x: (x == 'Yes').mean()
        }).round(3)
        
        regional_analysis.columns = ['Customers', 'MRR', 'Service_Level', 'Churn_Rate']
        
        # Estimate market size (customers per 1000 residents - Bahamas specific)
        bahamas_population = {
            'Nassau': 274400,
            'Freeport': 26910,
            'Abaco': 17224,
            'Eleuthera': 11165,
            'Exuma': 7314,
            'Other Islands': 50000
        }
        
        regional_analysis['Population'] = regional_analysis.index.map(bahamas_population)
        regional_analysis['Penetration_Rate'] = (
            regional_analysis['Customers'] / regional_analysis['Population'] * 100
        ).round(2)
        
        # Calculate addressable market (assuming 70% penetration is realistic max)
        regional_analysis['Addressable_Customers'] = (
            regional_analysis['Population'] * 0.70 - regional_analysis['Customers']
        ).clip(lower=0)
        
        # Revenue opportunity
        avg_arpu = self.df['MonthlyCharges'].mean()
        regional_analysis['Revenue_Opportunity_Annual'] = (
            regional_analysis['Addressable_Customers'] * avg_arpu * 12
        ).round(0)
        
        # Opportunity score (combining low penetration, low churn, high service availability)
        regional_analysis['Opportunity_Score'] = (
            (100 - regional_analysis['Penetration_Rate']) * 0.4 +
            (1 - regional_analysis['Churn_Rate']) * 100 * 0.3 +
            regional_analysis['Service_Level'].map({'Full': 100, 'Partial': 60, 'Limited': 30}) * 0.3
        ).round(1)
        
        print("\nRegional Market Opportunities:")
        print(regional_analysis.sort_values('Opportunity_Score', ascending=False))
        
        return regional_analysis
    
    def retention_campaign_roi_calculator(self):
        """Calculate ROI for retention campaigns targeting high-risk customers"""
        print("\n" + "="*60)
        print("RETENTION CAMPAIGN ROI ANALYSIS")
        print("="*60)
        
        # Identify high-risk customers
        high_risk = self.df[
            (self.df['ChurnRiskCategory'] == 'High') & 
            (self.df['Churn'] == 'No')
        ].copy()
        
        print(f"\nHigh-Risk Customer Base: {len(high_risk):,}")
        
        # Campaign assumptions
        campaign_scenarios = {
            'Discount Offer': {
                'cost_per_customer': 25,
                'expected_retention_lift': 0.25,  # 25% of at-risk stay
                'discount_duration_months': 6,
                'discount_amount': 10
            },
            'Service Upgrade': {
                'cost_per_customer': 50,
                'expected_retention_lift': 0.35,
                'discount_duration_months': 0,
                'discount_amount': 0
            },
            'Loyalty Program': {
                'cost_per_customer': 15,
                'expected_retention_lift': 0.20,
                'discount_duration_months': 12,
                'discount_amount': 5
            }
        }
        
        results = []
        
        for campaign_name, params in campaign_scenarios.items():
            # Calculate metrics
            total_campaign_cost = params['cost_per_customer'] * len(high_risk)
            customers_retained = len(high_risk) * params['expected_retention_lift']
            
            # Revenue saved (12-month value)
            avg_monthly_revenue = high_risk['MonthlyCharges'].mean()
            revenue_saved = customers_retained * avg_monthly_revenue * 12
            
            # Discount cost
            discount_cost = (
                customers_retained * 
                params['discount_amount'] * 
                params['discount_duration_months']
            )
            
            # Net benefit
            net_benefit = revenue_saved - total_campaign_cost - discount_cost
            roi = (net_benefit / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
            
            results.append({
                'Campaign': campaign_name,
                'Target_Customers': len(high_risk),
                'Campaign_Cost': total_campaign_cost,
                'Expected_Retained': int(customers_retained),
                'Revenue_Saved_12M': revenue_saved,
                'Discount_Cost': discount_cost,
                'Net_Benefit': net_benefit,
                'ROI_%': roi
            })
        
        roi_df = pd.DataFrame(results)
        
        print("\nCampaign ROI Comparison:")
        for _, row in roi_df.iterrows():
            print(f"\n{row['Campaign']}:")
            print(f"  Investment:        ${row['Campaign_Cost']:,.0f}")
            print(f"  Customers Retained: {row['Expected_Retained']:,}")
            print(f"  Revenue Saved:     ${row['Revenue_Saved_12M']:,.0f}")
            print(f"  Net Benefit:       ${row['Net_Benefit']:,.0f}")
            print(f"  ROI:               {row['ROI_%']:.1f}%")
        
        return roi_df
    
    def export_insights_for_tableau(self):
        """Export enhanced dataset with predictions for Tableau"""
        print("\n" + "="*60)
        print("EXPORTING ENHANCED DATASET")
        print("="*60)
        
        # Add prediction-based segments
        self.df['ChurnProbability_Segment'] = pd.cut(
            self.df['ChurnProbability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Export
        output_file = '../data/cable_bahamas_with_predictions.csv'
        self.df.to_csv(output_file, index=False)
        print(f"✓ Exported enhanced dataset: {output_file}")
        
        return self.df


def main():
    """Execute all analytics"""
    print("="*60)
    print("CABLE BAHAMAS - ADVANCED ANALYTICS SUITE")
    print("Market Intelligence & Forecasting")
    print("="*60 + "\n")
    
    data_dir='../data/'
    # Initialize
    analytics = CableBahamasPredictiveAnalytics()
    
    # 1. Churn Prediction Model
    model, importance = analytics.build_churn_prediction_model()
    
    # 2. Revenue Forecasting
    forecast = analytics.revenue_forecasting(months_ahead=12)
    
    forecast.to_csv(data_dir + 'revenue_forecast_12m.csv', index=False)
    print("✓ Saved revenue forecast to revenue_forecast_12m.csv")
    
    # 3. Market Opportunity Analysis
    market_opps = analytics.market_opportunity_analysis()
    market_opps.to_csv(data_dir + 'market_opportunities.csv')
    print("✓ Saved market opportunities to market_opportunities.csv")
    
    # 4. Retention Campaign ROI
    campaign_roi = analytics.retention_campaign_roi_calculator()
    campaign_roi.to_csv(data_dir + 'retention_campaign_roi.csv', index=False)
    print("✓ Saved campaign ROI to retention_campaign_roi.csv")
    
    # 5. Export for Tableau
    enhanced_df = analytics.export_insights_for_tableau()
    
    print("\n" + "="*60)
    print("ANALYTICS COMPLETE")
    print("="*60)
    print("\nGenerated Files:")
    print("  • cable_bahamas_with_predictions.csv")
    print("  • revenue_forecast_12m.csv")
    print("  • market_opportunities.csv")
    print("  • retention_campaign_roi.csv")
    print("\nReady for Tableau visualization!")


if __name__ == "__main__":
    main()