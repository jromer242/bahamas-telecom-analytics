import sqlite3
import pandas as pd
import re

def execute_sql_file(conn, sql_file_path):
    """
    Execute SQL file with CREATE VIEW statements in SQLite.
    Handles SQLite's limitation of one CREATE VIEW per execution.
    """
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    # Split the SQL content into individual statements
    # Remove comments and empty lines
    lines = []
    for line in sql_content.split('\n'):
        # Skip comment lines
        if line.strip().startswith('--'):
            continue
        lines.append(line)
    
    sql_content = '\n'.join(lines)
    
    # Split by semicolon but keep CREATE VIEW statements together
    statements = []
    current_statement = []
    in_create_view = False
    
    for line in sql_content.split('\n'):
        stripped = line.strip()
        
        # Start of CREATE VIEW
        if 'CREATE VIEW' in stripped.upper():
            if current_statement:
                statements.append('\n'.join(current_statement))
                current_statement = []
            in_create_view = True
        
        current_statement.append(line)
        
        # End of statement (semicolon)
        if ';' in line:
            statements.append('\n'.join(current_statement))
            current_statement = []
            in_create_view = False
    
    # Execute each statement individually
    executed_count = 0
    failed_count = 0
    
    for i, statement in enumerate(statements, 1):
        statement = statement.strip()
        if not statement or statement == ';':
            continue
            
        try:
            conn.execute(statement)
            executed_count += 1
            
            # Extract view name for feedback
            if 'CREATE VIEW' in statement.upper():
                match = re.search(r'CREATE VIEW\s+(\w+)', statement, re.IGNORECASE)
                if match:
                    view_name = match.group(1)
                    print(f"✓ Created view: {view_name}")
        except sqlite3.OperationalError as e:
            failed_count += 1
            print(f"✗ Error in statement {i}: {str(e)[:100]}")
            # Print first few lines of problematic statement
            print(f"  Statement preview: {statement[:200]}...")
            continue
    
    conn.commit()
    print(f"\n{'='*60}")
    print(f"Execution Summary:")
    print(f"  Successfully executed: {executed_count}")
    print(f"  Failed: {failed_count}")
    print(f"{'='*60}\n")
    
    return executed_count, failed_count


def main():
    """
    Main execution function
    """
    print("="*60)
    print("CABLE BAHAMAS SQL EXECUTION PIPELINE")
    print("="*60 + "\n")
    
    # 1. Connect to SQLite database
    print("Step 1: Connecting to database...")
    conn = sqlite3.connect('./cable_bahamas.db')
    print("✓ Connected to cable_bahamas.db\n")
    
    # 2. Load enriched CSV data
    print("Step 2: Loading enriched data...")
    try:
        df = pd.read_csv('../data/cable_bahamas_enriched.csv')
        print(f"✓ Loaded {len(df):,} records from cable_bahamas_enriched.csv")
        print(f"  Columns: {len(df.columns)}\n")
    except FileNotFoundError:
        print("✗ Error: cable_bahamas_enriched.csv not found!")
        print("  Please run the enrichment script first.\n")
        conn.close()
        return
    
    # 3. Load data into SQLite
    print("Step 3: Creating table in SQLite...")
    df.to_sql('cable_bahamas_customer_data', conn, if_exists='replace', index=False)
    print("✓ Created table: cable_bahamas_customer_data\n")
    
    # 4. Execute SQL queries
    print("Step 4: Executing SQL views and queries...")
    try:
        executed, failed = execute_sql_file(conn, '../sql/sql_queries.sql')
    except FileNotFoundError:
        print("✗ Error: sql_queries.sql not found!")
        print("  Please ensure the SQL file is in the same directory.\n")
        conn.close()
        return
    
    # 5. Verify views were created
    print("\nStep 5: Verifying created views...")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
    views = cursor.fetchall()
    
    if views:
        print(f"✓ Found {len(views)} views in database:")
        for view in views:
            print(f"    • {view[0]}")
    else:
        print("✗ No views found in database")
        print("  Check SQL execution errors above\n")
        conn.close()
        return
    
    # 6. Export key views for Tableau
    print("\nStep 6: Exporting views to CSV for Tableau...")
    export_views = [
        'vw_executive_kpis',
        'vw_regional_performance', 
        'vw_customer_segments',
        'vw_package_performance',
        'vw_high_risk_retention_targets',
        'vw_tableau_master'
    ]
    
    exported_count = 0
    for view_name in export_views:
        try:
            query = f"SELECT * FROM {view_name}"
            df_view = pd.read_sql(query, conn)
            output_file = f"../{view_name}.csv"
            df_view.to_csv(output_file, index=False)
            print(f"  ✓ Exported {view_name} ({len(df_view):,} rows)")
            exported_count += 1
        except Exception as e:
            print(f"  ✗ Failed to export {view_name}: {str(e)[:80]}")
    
    # 7. Summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"Database: cable_bahamas.db")
    print(f"Base table: cable_bahamas_customer_data ({len(df):,} rows)")
    print(f"Views created: {len(views)}")
    print(f"CSVs exported: {exported_count}")
    print("="*60 + "\n")
    
    # Close connection
    conn.close()
    print("✓ Database connection closed")
    
    # Additional tips
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Import vw_tableau_master.csv into Tableau as your main data source")
    print("2. Use other exported CSVs as supplementary data sources")
    print("3. Create relationships in Tableau using customerID as the key")
    print("4. Start building dashboards!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()