import pandas as pd
from pymongo import MongoClient
import re
from collections import defaultdict
import numpy as np
from datetime import datetime

class DatabaseCleaner:
    def __init__(self, connection_string="mongodb://localhost:27017", db_name="ai_test_db"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.cleaning_report = {}
    
    def analyze_all_collections(self):
        """Analyze all collections for data quality issues"""
        print("🔍 Starting Database Quality Analysis...")
        print("=" * 50)
        
        for collection_name in self.db.list_collection_names():
            print(f"\n📊 Analyzing Collection: {collection_name}")
            print("-" * 30)
            
            collection = self.db[collection_name]
            self.cleaning_report[collection_name] = self.analyze_collection(collection)
        
        self.generate_summary_report()
        return self.cleaning_report
    
    def analyze_collection(self, collection):
        """Analyze a single collection for data quality issues"""
        # Get sample data for analysis
        sample_data = list(collection.find().limit(1000))
        if not sample_data:
            return {"status": "empty", "issues": []}
        
        df = pd.DataFrame(sample_data)
        issues = []
        stats = {}
        
        # 1. Duplicate Detection
        duplicates = self.find_duplicates(df, collection.name)
        if duplicates:
            issues.extend(duplicates)
        
        # 2. Email Validation
        email_issues = self.validate_emails(df)
        if email_issues:
            issues.extend(email_issues)
        
        # 3. Missing Data Analysis
        missing_data = self.analyze_missing_data(df)
        if missing_data:
            issues.extend(missing_data)
        
        # 4. Data Type Inconsistencies
        type_issues = self.check_data_types(df)
        if type_issues:
            issues.extend(type_issues)
        
        # 5. Outlier Detection for Numerical Fields
        outliers = self.detect_outliers(df)
        if outliers:
            issues.extend(outliers)
        
        # Collection Statistics
        stats = {
            "total_records": len(sample_data),
            "total_fields": len(df.columns),
            "issues_found": len(issues)
        }
        
        # Print results for this collection
        self.print_collection_results(collection.name, stats, issues)
        
        return {
            "stats": stats,
            "issues": issues,
            "sample_data": sample_data[:5]  # Keep 5 samples for reference
        }
    
    def find_duplicates(self, df, collection_name):
        """Find duplicate records based on key fields"""
        issues = []
        
        # Define key fields for different collections
        key_fields_map = {
            "customers": ["email"],
            "users": ["email"],
            "products": ["name"],
            "orders": ["customerId", "productId", "orderDate"],
            "payments": ["orderId"]
        }
        
        key_fields = key_fields_map.get(collection_name, [])
        
        for field in key_fields:
            if field in df.columns:
                # Find duplicates based on this field
                duplicates = df[df.duplicated(subset=[field], keep=False)]
                if not duplicates.empty:
                    duplicate_values = duplicates[field].value_counts()
                    issues.append({
                        "type": "duplicates",
                        "field": field,
                        "count": len(duplicates),
                        "examples": duplicate_values.head(3).to_dict(),
                        "severity": "high" if len(duplicates) > 10 else "medium"
                    })
        
        return issues
    
    def validate_emails(self, df):
        """Validate email formats"""
        issues = []
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if 'email' in df.columns:
            # Check for invalid email formats
            invalid_emails = df[~df['email'].str.match(email_pattern, na=False)]
            if not invalid_emails.empty:
                issues.append({
                    "type": "invalid_email_format",
                    "field": "email",
                    "count": len(invalid_emails),
                    "examples": invalid_emails['email'].head(3).tolist(),
                    "severity": "medium"
                })
            
            # Check for email case inconsistencies
            mixed_case_emails = df[df['email'].str.contains(r'[A-Z]', na=False)]
            if not mixed_case_emails.empty:
                issues.append({
                    "type": "email_case_inconsistency",
                    "field": "email",
                    "count": len(mixed_case_emails),
                    "examples": mixed_case_emails['email'].head(3).tolist(),
                    "severity": "low"
                })
        
        return issues
    
    def analyze_missing_data(self, df):
        """Analyze missing or null data"""
        issues = []
        
        for column in df.columns:
            if column == '_id':
                continue
                
            missing_count = df[column].isnull().sum()
            empty_strings = (df[column] == '').sum() if df[column].dtype == 'object' else 0
            total_missing = missing_count + empty_strings
            
            if total_missing > 0:
                missing_percentage = (total_missing / len(df)) * 100
                severity = "high" if missing_percentage > 20 else "medium" if missing_percentage > 5 else "low"
                
                issues.append({
                    "type": "missing_data",
                    "field": column,
                    "count": total_missing,
                    "percentage": round(missing_percentage, 2),
                    "severity": severity
                })
        
        return issues
    
    def check_data_types(self, df):
        """Check for data type inconsistencies"""
        issues = []
        
        # Check phone number formats
        if 'phone' in df.columns:
            phone_formats = df['phone'].astype(str).str.len().value_counts()
            if len(phone_formats) > 3:  # Too many different formats
                issues.append({
                    "type": "inconsistent_phone_format",
                    "field": "phone",
                    "count": len(df),  # Add missing count field
                    "formats_found": phone_formats.to_dict(),
                    "severity": "medium"
                })
        
        # Check for mixed data types in amount fields
        amount_fields = ['amount', 'price', 'total', 'cost']
        for field in amount_fields:
            if field in df.columns:
                # Check if there are string values in numeric fields
                non_numeric = df[field].apply(lambda x: not isinstance(x, (int, float, type(None))))
                if non_numeric.any():
                    issues.append({
                        "type": "mixed_data_types",
                        "field": field,
                        "count": non_numeric.sum(),
                        "examples": df[non_numeric][field].head(3).tolist(),
                        "severity": "high"
                    })
        
        return issues
    
    def detect_outliers(self, df):
        """Detect statistical outliers in numerical fields"""
        issues = []
        
        numerical_fields = df.select_dtypes(include=[np.number]).columns
        
        for field in numerical_fields:
            if field in ['_id']:
                continue
                
            values = df[field].dropna()
            if len(values) < 10:  # Need enough data for outlier detection
                continue
            
            # Use IQR method for outlier detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(values)) * 100
                severity = "high" if outlier_percentage > 10 else "medium" if outlier_percentage > 5 else "low"
                
                issues.append({
                    "type": "statistical_outliers",
                    "field": field,
                    "count": len(outliers),
                    "percentage": round(outlier_percentage, 2),
                    "range": f"{lower_bound:.2f} to {upper_bound:.2f}",
                    "examples": outliers.head(3).tolist(),
                    "severity": severity
                })
        
        return issues
    
    def print_collection_results(self, collection_name, stats, issues):
        """Print results for a single collection"""
        print(f"📈 Records: {stats['total_records']}")
        print(f"🔧 Fields: {stats['total_fields']}")
        print(f"⚠️  Issues Found: {stats['issues_found']}")
        
        if issues:
            print("\n🚨 Data Quality Issues:")
            for issue in issues:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                emoji = severity_emoji.get(issue['severity'], "⚪")
                
                print(f"  {emoji} {issue['type'].replace('_', ' ').title()}")
                print(f"     Field: {issue['field']}")
                print(f"     Count: {issue['count']}")
                
                if 'examples' in issue:
                    print(f"     Examples: {issue['examples']}")
                if 'percentage' in issue:
                    print(f"     Percentage: {issue['percentage']}%")
                print()
        else:
            print("✅ No data quality issues found!")
    
    def generate_summary_report(self):
        """Generate overall database quality summary"""
        print("\n" + "=" * 60)
        print("📋 DATABASE QUALITY SUMMARY REPORT")
        print("=" * 60)
        
        total_collections = len(self.cleaning_report)
        total_issues = sum(len(report['issues']) for report in self.cleaning_report.values())
        total_records = sum(report['stats']['total_records'] for report in self.cleaning_report.values())
        
        print(f"📊 Collections Analyzed: {total_collections}")
        print(f"📈 Total Records: {total_records:,}")
        print(f"⚠️  Total Issues Found: {total_issues}")
        
        # Issue breakdown by severity
        severity_counts = defaultdict(int)
        issue_type_counts = defaultdict(int)
        
        for collection_report in self.cleaning_report.values():
            for issue in collection_report['issues']:
                severity_counts[issue['severity']] += 1
                issue_type_counts[issue['type']] += 1
        
        if severity_counts:
            print(f"\n🚨 Issues by Severity:")
            print(f"   🔴 High: {severity_counts['high']}")
            print(f"   🟡 Medium: {severity_counts['medium']}")
            print(f"   🟢 Low: {severity_counts['low']}")
            
            print(f"\n📋 Most Common Issues:")
            for issue_type, count in sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {issue_type.replace('_', ' ').title()}: {count}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if severity_counts['high'] > 0:
            print("   🔴 Address high-severity issues first (duplicates, data type problems)")
        if severity_counts['medium'] > 0:
            print("   🟡 Review medium-severity issues (format inconsistencies)")
        if total_issues == 0:
            print("   ✅ Your database quality looks good!")
        else:
            print("   📊 Consider implementing data validation rules")
            print("   🔄 Set up regular data quality monitoring")

def main():
    """Run the data cleaning analysis"""
    print("🚀 Database Quality Analyzer")
    print("=" * 30)
    
    # Initialize cleaner
    cleaner = DatabaseCleaner()
    
    # Run analysis
    report = cleaner.analyze_all_collections()
    
    print(f"\n✅ Analysis Complete! Check the detailed report above.")
    
    return report

if __name__ == "__main__":
    main()