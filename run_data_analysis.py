#!/usr/bin/env python3
"""
Quick script to run data quality analysis on ai_test_db
"""

from data_cleaner import DatabaseCleaner
import json

def run_analysis():
    print("🔍 Analyzing ai_test_db for data quality issues...")
    print()
    
    # Create cleaner instance
    cleaner = DatabaseCleaner(
        connection_string="mongodb://localhost:27017",
        db_name="ai_test_db"
    )
    
    # Run the analysis
    report = cleaner.analyze_all_collections()
    
    # Save detailed report to file
    with open('data_quality_report.json', 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_report = {}
        for collection, data in report.items():
            serializable_report[collection] = {
                'stats': data['stats'],
                'issues': data['issues']
                # Skip sample_data as it contains ObjectId which isn't JSON serializable
            }
        json.dump(serializable_report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: data_quality_report.json")
    
    return report

if __name__ == "__main__":
    run_analysis()