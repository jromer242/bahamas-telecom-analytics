import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os, sys

class CableBahamasDataEnrichment:
    """
    Enriches telecom churn dataset with Cable Bahamas-specific features
    for realistic BI analysis and visualization
    """
    
    def __init__(self, data_path):
        """Initialize with path to the Kaggle dataset"""
        self.data_path = data_path
        self.df = None
        
        # Cable Bahamas specific configurations
        self.regions = {
            'Nassau': 0.45,  # Population distribution weights
            'Freeport': 0.25,
            'Abaco': 0.10,
            'Eleuthera': 0.08,
            'Exuma': 0.07,
            'Other Islands': 0.05
        }
        
        self.competitors = ['BTC', 'None']
        
        self.service_packages = {
            'Basic Cable': {'base_price': 49.99, 'channels': 50},
            'Premium Cable': {'base_price': 89.99, 'channels': 150},
            'REV Internet Basic': {'base_price': 59.99, 'speed': '25 Mbps'},
            'REV Internet Premium': {'base_price': 99.99, 'speed': '100 Mbps'},
            'ALIVFibr': {'base_price': 149.99, 'speed': '500 Mbps', 'fiber': True},
            'Triple Play': {'base_price': 139.99, 'bundle': True},
            'Internet + TV Bundle': {'base_price': 119.99, 'bundle': True}
        }
        
    def load_data(self):
        """Load the original Kaggle dataset"""
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        csv_path = os.path.join(self.data_path, csv_files[0])
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} records from {csv_files[0]}")
        return self
    
    def add_geographic_data(self):
        """Add Bahamas-specific geographic information"""
        # Assign regions based on population distribution
        self.df['Region'] = np.random.choice(
            list(self.regions.keys()),
            size=len(self.df),
            p=list(self.regions.values())
        )
        
        # Add island-specific service availability
        # Rural islands have limited service options
        self.df['ServiceAvailability'] = self.df['Region'].map({
            'Nassau': 'Full',
            'Freeport': 'Full',
            'Abaco': 'Partial',
            'Eleuthera': 'Partial',
            'Exuma': 'Limited',
            'Other Islands': 'Limited'
        })
        
        print("✓ Added geographic data (Region, ServiceAvailability)")
        return self
    
    def add_competitive_data(self):
        """Add competitor and market data"""
        # Primary competitor (only BTC competes with Cable Bahamas)
        # BTC has ~10% market share vs Cable Bahamas ~90%
        
        # Assign competitor presence (BTC more common in areas with service gaps)
        def assign_competitor(row):
            if row['ServiceAvailability'] == 'Limited':
                # BTC stronger in underserved areas
                return np.random.choice(['BTC', 'None'], p=[0.40, 0.60])
            elif row['ServiceAvailability'] == 'Partial':
                return np.random.choice(['BTC', 'None'], p=[0.25, 0.75])
            else:  # Full service areas
                return np.random.choice(['BTC', 'None'], p=[0.15, 0.85])
        
        self.df['PrimaryCompetitor'] = self.df.apply(assign_competitor, axis=1)
        
        # BTC competitive offers (more likely for churned customers)
        self.df['ReceivedCompetitiveOffer'] = np.where(
            (self.df['Churn'] == 'Yes') & 
            (self.df['PrimaryCompetitor'] == 'BTC') &
            (np.random.random(len(self.df)) < 0.70),
            'Yes', 'No'
        )
        
        # Market tenure in months (how long Cable Bahamas has been in their area)
        # Cable Bahamas has 90%+ household penetration
        self.df['MarketTenure'] = self.df['Region'].map({
            'Nassau': np.random.randint(60, 120, len(self.df)),
            'Freeport': np.random.randint(48, 96, len(self.df)),
            'Abaco': np.random.randint(24, 60, len(self.df)),
            'Eleuthera': np.random.randint(12, 48, len(self.df)),
            'Exuma': np.random.randint(6, 36, len(self.df)),
            'Other Islands': np.random.randint(3, 24, len(self.df))
        })
        
        print("✓ Added competitive data (BTC competitor info, Market data)")
        return self
    
    def add_service_package_data(self):
        """Map existing services to Cable Bahamas packages (REV/ALIVFibr brands)"""
        def assign_package(row):
            if row['InternetService'] == 'No':
                return 'Basic Cable'
            elif row['InternetService'] == 'Fiber optic':
                # Fiber customers get ALIVFibr
                if row['StreamingTV'] == 'Yes' and row['StreamingMovies'] == 'Yes':
                    return 'Triple Play'
                else:
                    return 'ALIVFibr'
            elif row['InternetService'] == 'DSL':
                # DSL customers on REV Internet
                if row['StreamingTV'] == 'Yes' or row['StreamingMovies'] == 'Yes':
                    return 'Internet + TV Bundle'
                else:
                    return 'REV Internet Premium'
            return 'Basic Cable'
        
        self.df['ServicePackage'] = self.df.apply(assign_package, axis=1)
        
        # Add package pricing in BSD (Bahamian Dollar, 1:1 with USD)
        self.df['PackagePrice_BSD'] = self.df['ServicePackage'].map(
            {k: v['base_price'] for k, v in self.service_packages.items()}
        )
        
        print("✓ Added Cable Bahamas service packages (REV/ALIVFibr) and BSD pricing")
        return self
    
    def add_customer_behavior_metrics(self):
        """Add enhanced customer behavior and engagement metrics"""
        # Customer support interactions
        self.df['SupportTickets_LastYear'] = np.random.poisson(
            lam=self.df['Churn'].map({'Yes': 4.5, 'No': 1.2}),
            size=len(self.df)
        )
        
        # Service outage experienced
        self.df['OutagesExperienced'] = np.random.poisson(
            lam=self.df['Region'].map({
                'Nassau': 1.5,
                'Freeport': 2.0,
                'Abaco': 3.5,
                'Eleuthera': 3.0,
                'Exuma': 4.0,
                'Other Islands': 5.0
            }),
            size=len(self.df)
        )
        
        # Customer satisfaction score (1-10)
        base_satisfaction = np.random.normal(7.5, 1.5, len(self.df))
        churn_penalty = self.df['Churn'].map({'Yes': -2.5, 'No': 0})
        self.df['SatisfactionScore'] = np.clip(
            base_satisfaction + churn_penalty,
            1, 10
        ).round(1)
        
        # Payment reliability score (0-100)
        self.df['PaymentReliabilityScore'] = np.where(
            self.df['PaymentMethod'] == 'Electronic check',
            np.random.uniform(60, 85, len(self.df)),
            np.random.uniform(80, 98, len(self.df))
        ).round(0)
        
        # Online account usage
        self.df['OnlineAccountLogins_LastMonth'] = np.random.poisson(
            lam=self.df['OnlineSecurity'].map({'Yes': 8, 'No': 2, 'No internet service': 0}),
            size=len(self.df)
        )
        
        print("✓ Added customer behavior metrics (Support tickets, Satisfaction, etc.)")
        return self
    
    def add_financial_metrics(self):
        """Add detailed financial and revenue metrics"""
        # Convert TotalCharges to numeric (handling spaces)
        self.df['TotalCharges'] = pd.to_numeric(
            self.df['TotalCharges'], 
            errors='coerce'
        )
        
        # Fill NaN TotalCharges (new customers)
        self.df['TotalCharges'].fillna(
            self.df['MonthlyCharges'], 
            inplace=True
        )
        
        # Customer Lifetime Value (CLV) estimation
        self.df['EstimatedCLV'] = np.where(
            self.df['Churn'] == 'Yes',
            self.df['TotalCharges'],
            self.df['TotalCharges'] + (self.df['MonthlyCharges'] * 24)  # Projected 2 years
        )
        
        # Revenue at Risk (for churned customers)
        self.df['RevenueAtRisk'] = np.where(
            self.df['Churn'] == 'Yes',
            self.df['MonthlyCharges'] * 12,  # Annual value
            0
        )
        
        # Payment delay days (average)
        self.df['AvgPaymentDelayDays'] = np.where(
            self.df['PaymentMethod'] == 'Electronic check',
            np.random.poisson(8, len(self.df)),
            np.random.poisson(2, len(self.df))
        )
        
        # Discount percentage currently applied
        self.df['CurrentDiscount_Percent'] = np.random.choice(
            [0, 5, 10, 15, 20],
            size=len(self.df),
            p=[0.5, 0.2, 0.15, 0.10, 0.05]
        )
        
        print("✓ Added financial metrics (CLV, Revenue at Risk, Payment data)")
        return self
    
    def add_temporal_features(self):
        """Add time-based features for trend analysis"""
        # Simulate signup dates based on tenure
        end_date = datetime.now()
        
        self.df['SignupDate'] = self.df['tenure'].apply(
            lambda x: end_date - timedelta(days=int(x * 30))
        )
        
        # Cohort (year-month of signup)
        self.df['SignupCohort'] = self.df['SignupDate'].dt.to_period('M').astype(str)
        
        # Last service upgrade/downgrade
        self.df['MonthsSinceLastChange'] = np.random.randint(
            0, 
            self.df['tenure'] + 1
        )
        
        # Contract renewal date (for contract customers)
        self.df['ContractRenewalMonth'] = np.where(
            self.df['Contract'] != 'Month-to-month',
            self.df['SignupDate'] + pd.to_timedelta(
                np.random.randint(1, 24, len(self.df)) * 30, unit='D'
            ),
            pd.NaT
        )
        
        print("✓ Added temporal features (Signup dates, Cohorts, Renewal dates)")
        return self
    
    def add_churn_risk_indicators(self):
        """Add calculated churn risk indicators for predictive modeling"""
        # Engagement score (composite)
        engagement_factors = []
        
        if 'OnlineAccountLogins_LastMonth' in self.df.columns:
            engagement_factors.append(self.df['OnlineAccountLogins_LastMonth'] / 10)
        
        if 'SupportTickets_LastYear' in self.df.columns:
            engagement_factors.append(10 - self.df['SupportTickets_LastYear'])
        
        self.df['EngagementScore'] = np.mean(engagement_factors, axis=0).clip(0, 10)
        
        # Churn risk category (rule-based)
        def calculate_risk_category(row):
            risk_score = 0
            
            # Contract type
            if row['Contract'] == 'Month-to-month':
                risk_score += 3
            
            # Payment method
            if row['PaymentMethod'] == 'Electronic check':
                risk_score += 2
            
            # Support tickets
            if 'SupportTickets_LastYear' in row and row['SupportTickets_LastYear'] > 3:
                risk_score += 2
            
            # Tenure
            if row['tenure'] < 6:
                risk_score += 2
            
            # Satisfaction
            if 'SatisfactionScore' in row and row['SatisfactionScore'] < 6:
                risk_score += 3
            
            if risk_score >= 7:
                return 'High'
            elif risk_score >= 4:
                return 'Medium'
            else:
                return 'Low'
        
        self.df['ChurnRiskCategory'] = self.df.apply(calculate_risk_category, axis=1)
        
        print("✓ Added churn risk indicators (Engagement, Risk categories)")
        return self
    
    def create_enriched_dataset(self, output_path='cable_bahamas_enriched.csv'):
        """Execute all enrichment steps and save the result"""
        print("\n" + "="*60)
        print("CABLE BAHAMAS DATA ENRICHMENT PIPELINE")
        print("="*60 + "\n")
        
        (self.load_data()
             .add_geographic_data()
             .add_competitive_data()
             .add_service_package_data()
             .add_customer_behavior_metrics()
             .add_financial_metrics()
             .add_temporal_features()
             .add_churn_risk_indicators())
        
        # Save enriched dataset
        self.df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ ENRICHMENT COMPLETE!")
        print(f"{'='*60}")
        print(f"\nOriginal columns: {len(pd.read_csv(os.path.join(self.data_path, os.listdir(self.data_path)[0])).columns)}")
        print(f"Enriched columns: {len(self.df.columns)}")
        print(f"New columns added: {len(self.df.columns) - len(pd.read_csv(os.path.join(self.data_path, os.listdir(self.data_path)[0])).columns)}")
        print(f"\nTotal records: {len(self.df):,}")
        print(f"Output saved to: {output_path}")
        
        return self.df
    
    def generate_data_dictionary(self, output_path='data_dictionary.csv'):
        """Generate a data dictionary for the enriched dataset"""
        data_dict = {
            'Column': [],
            'Description': [],
            'Data Type': [],
            'Source': []
        }
        
        # Original columns
        original_cols = {
            'customerID': 'Unique customer identifier',
            'gender': 'Customer gender (Male/Female)',
            'SeniorCitizen': 'Whether customer is senior citizen (1=Yes, 0=No)',
            'Partner': 'Whether customer has a partner',
            'Dependents': 'Whether customer has dependents',
            'tenure': 'Number of months customer has stayed',
            'PhoneService': 'Whether customer has phone service',
            'MultipleLines': 'Whether customer has multiple lines',
            'InternetService': 'Type of internet service',
            'OnlineSecurity': 'Whether customer has online security',
            'OnlineBackup': 'Whether customer has online backup',
            'DeviceProtection': 'Whether customer has device protection',
            'TechSupport': 'Whether customer has tech support',
            'StreamingTV': 'Whether customer has streaming TV',
            'StreamingMovies': 'Whether customer has streaming movies',
            'Contract': 'Contract term (Month-to-month, One year, Two year)',
            'PaperlessBilling': 'Whether customer has paperless billing',
            'PaymentMethod': 'Payment method used',
            'MonthlyCharges': 'Monthly charge amount',
            'TotalCharges': 'Total amount charged',
            'Churn': 'Whether customer churned (Yes/No)'
        }
        
        # New columns
        new_cols = {
            'Region': 'Customer location in the Bahamas',
            'ServiceAvailability': 'Level of service availability in customer region',
            'PrimaryCompetitor': 'Competitor presence (BTC only - Cable Bahamas has ~90% market share)',
            'ReceivedCompetitiveOffer': 'Whether customer received competitive offer from BTC',
            'MarketTenure': 'How long Cable Bahamas has operated in region (months)',
            'ServicePackage': 'Cable Bahamas service package (REV/ALIVFibr brands)',
            'PackagePrice_BSD': 'Package price in Bahamian Dollars',
            'SupportTickets_LastYear': 'Number of support tickets in last 12 months',
            'OutagesExperienced': 'Number of service outages experienced',
            'SatisfactionScore': 'Customer satisfaction rating (1-10)',
            'PaymentReliabilityScore': 'Payment reliability score (0-100)',
            'OnlineAccountLogins_LastMonth': 'Online account logins last month',
            'EstimatedCLV': 'Estimated Customer Lifetime Value',
            'RevenueAtRisk': 'Annual revenue at risk (for churned customers)',
            'AvgPaymentDelayDays': 'Average payment delay in days',
            'CurrentDiscount_Percent': 'Current discount percentage applied',
            'SignupDate': 'Customer signup date',
            'SignupCohort': 'Signup cohort (Year-Month)',
            'MonthsSinceLastChange': 'Months since last service change',
            'ContractRenewalMonth': 'Contract renewal date',
            'EngagementScore': 'Calculated customer engagement score',
            'ChurnRiskCategory': 'Churn risk level (Low/Medium/High)'
        }
        
        for col in self.df.columns:
            data_dict['Column'].append(col)
            
            if col in original_cols:
                data_dict['Description'].append(original_cols[col])
                data_dict['Source'].append('Original Kaggle Dataset')
            elif col in new_cols:
                data_dict['Description'].append(new_cols[col])
                data_dict['Source'].append('Cable Bahamas Enrichment')
            else:
                data_dict['Description'].append('N/A')
                data_dict['Source'].append('Unknown')
            
            data_dict['Data Type'].append(str(self.df[col].dtype))
        
        dict_df = pd.DataFrame(data_dict)
        dict_df.to_csv(output_path, index=False)
        print(f"\n✓ Data dictionary saved to: {output_path}")
        
        return dict_df


# Usage Example
if __name__ == "__main__":
    # Initialize enrichment
    import kagglehub, os

    # Re-download (it will use cached version, so it's fast)
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print("Dataset location:", path)

    # Verify the CSV is there
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print("CSV files found:", csv_files)

    # Now use it
    enricher = CableBahamasDataEnrichment(data_path=path)
    
    # Create enriched dataset
    enriched_df = enricher.create_enriched_dataset(
        output_path='data/cable_bahamas_customer_data.csv'
    )
    
    # Generate data dictionary
    data_dict = enricher.generate_data_dictionary(
        output_path='data/cable_bahamas_data_dictionary.csv'
    )
    
    # Display sample of enriched data
    print("\n" + "="*60)
    print("SAMPLE OF ENRICHED DATA")
    print("="*60)
    print(enriched_df.head())
    
    # Display new columns added
    print("\n" + "="*60)
    print("NEW COLUMNS ADDED")
    print("="*60)
    new_columns = [col for col in enriched_df.columns 
                   if col not in pd.read_csv(os.path.join(path, 
                   os.listdir(path)[0])).columns]
    for col in new_columns:
        print(f"  • {col}")