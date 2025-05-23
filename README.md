# Random Forest Based User Behavior Analysis

A comprehensive machine learning framework for analyzing behavioral differences between two user groups using Random Forest and statistical methods. Compare any two user cohorts to identify distinguishing patterns, predict outcomes, and generate actionable insights from event-driven data.

## 🎯 What This Tool Does

- **Compares behavioral patterns** between any two user groups or cohorts
- **Identifies distinguishing factors** that separate different user segments
- **Provides actionable insights** for business optimization and strategy
- **Handles activity bias** - normalizes for different user engagement levels
- **Multiple analysis methods** - statistical comparisons, ML predictions, and pattern recognition

## 📊 Key Features

### 1. **Activity Bias Correction**
- Addresses the common problem where different user groups have vastly different activity levels
- Uses proportional analysis to focus on behavioral priorities rather than raw volume
- Prevents high-activity users from skewing insights

### 2. **Four-Perspective Analysis**
- **Raw Frequency Analysis**: Which behaviors one group does more of (volume differences)
- **ML Feature Importance**: Most predictive behaviors for group classification (Random Forest)
- **Proportional Comparison**: What each group prioritizes relative to total activity (behavioral focus)  
- **Difference Detection**: Behaviors that distinguish groups (discriminating factors)

### 3. **Smart Visualizations**
- Color-coded scatter plots showing behavioral priorities
- Outlier filtering for better pattern visibility
- Interactive charts with key events labeled
- Comprehensive statistical summaries

### 4. **Data Quality Checks**
- Automatic detection of export truncation (Excel limits)
- User balance verification
- Activity bias warnings
- Data integrity validation

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Basic Usage

1. **Prepare your data**: Export user event logs separated by group classification
   - `group_a.csv`: Events from first user group (e.g., churned users, low spenders, etc.)
   - `group_b.csv`: Events from second user group (e.g., retained users, high spenders, etc.)

2. **Quick data validation**:
```python
python check_users.py  # Verify data quality first
```

3. **Run full analysis**:
```python
python user_behavior_analyzer.py
```

### Expected Data Format
```csv
user_info_id,event_name
U001,Food_Log_Success
U001,App_Opened
U001,Premium_Feature_Used
U002,Food_Log_Success
```

## 📈 Analysis Outputs

### Generated Files
- `user_behavior_analysis.png` - Four-panel visualization
- `event_comparison_results.csv` - Detailed event differences
- `feature_importance_results.csv` - ML feature rankings
- `logistic_regression_results.csv` - Event-retention associations

### Console Insights
```
📊 RAW FREQUENCY ANALYSIS (Activity Level):
Events that GROUP B does more:
  • Premium_Feature_Used: +12.3 more per user
  • Advanced_Settings_Access: +8.7 more per user

📈 PROPORTIONAL ANALYSIS (Behavioral Preferences):
Events that GROUP B prioritizes more:
  • Goal_Setting: 15.2% vs 8.3% (+6.9pp)
  • Social_Sharing: 12.1% vs 6.4% (+5.7pp)

🚨 DISTINGUISHING SIGNALS (Events Group A prioritizes more):
  • Error_Page_Views: 8.5% vs 2.1% (+6.4pp difference)
  • Support_Contact: 7.2% vs 3.1% (+4.1pp difference)
```

## 🔧 Configuration Options

### Vectorization Methods
```python
# Initialize with your data files
analyzer = UserBehaviorAnalyzer('group_a.csv', 'group_b.csv')

# Recommended: Handles activity bias
results = analyzer.run_full_analysis(vectorization_method='normalized_count')
```

### Customization
- Adjust minimum event frequency thresholds
- Modify outlier detection sensitivity
- Change visualization parameters
- Export additional statistical measures

## 📋 Project Structure

```
user-behavior-analysis/
│
├── user_behavior_analyzer.py      # Main analysis engine
├── check_users.py                 # Data quality validator
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── sample_data/                   # Example datasets
│   ├── sample_group_a.csv
│   └── sample_group_b.csv
│
└── outputs/                       # Generated results
    ├── user_behavior_analysis.png
    ├── event_comparison_results.csv
    ├── feature_importance_results.csv
    └── logistic_regression_results.csv
```

## 🎯 Use Cases

### Product & Marketing Teams
- **Feature adoption analysis**: What features distinguish power users vs casual users?
- **Conversion optimization**: Behavioral differences between converters vs browsers?
- **Segmentation insights**: How do different customer segments behave?

### Data Science & Analytics Teams  
- **Predictive modeling**: Most predictive behavioral features for any binary outcome
- **User segmentation**: Behavioral pattern identification across cohorts
- **A/B testing insights**: Behavioral impact analysis of experiments

### Business Strategy & Operations
- **Customer success**: Identify behaviors of successful vs struggling customers
- **Risk management**: Early warning behavioral signals
- **Process optimization**: Distinguish efficient vs inefficient user flows

## ⚠️ Important Considerations

### Data Requirements
- **Two distinct user groups**: Clear classification criteria for comparison
- **Same time period**: Events should cover equivalent timeframes for fair comparison
- **Clean group classification**: Clear definition and separation of user groups
- **Sufficient data**: At least 50+ users per group for reliable statistical insights

### Common Pitfalls
- **Activity bias**: High-activity users can skew raw count analysis (use normalized methods)
- **Temporal bias**: Ensure both groups have equivalent observation periods
- **Data leakage**: Don't include post-classification events in behavioral analysis
- **Export truncation**: Watch for CSV row limits (1,048,576 rows) in large datasets

## 🔍 Data Quality Checklist

Before running analysis, verify:
- [ ] Both files have similar number of unique users (balanced comparison)
- [ ] Activity levels per user are reasonable (not 20x different between groups)
- [ ] No Excel row limit truncation (exactly 1,048,575 rows)
- [ ] Event names are consistent between files
- [ ] Time periods are equivalent for both groups
- [ ] User group classifications are accurate and mutually exclusive

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter issues:
1. Check the data quality first using `check_users.py`
2. Verify your CSV format matches the expected structure
3. Review the console output for warnings and recommendations
4. Open an issue with sample data and error messages

## 🔬 Technical Details

### Machine Learning Models
- **Random Forest**: Feature importance ranking (handles non-linear relationships)
- **Logistic Regression**: Linear associations with retention (interpretable coefficients)
- **Count Vectorization**: Converts event sequences to numerical features

### Statistical Methods
- **Proportional normalization**: Each user's events sum to 1.0 
- **Min/max frequency filtering**: Removes rare events for stability
- **Outlier detection**: 95th percentile filtering for visualization
- **Cross-validation**: 80/20 train/test split for model evaluation

### Performance Optimizations
- **Lazy loading**: Only loads required columns for speed
- **Memory efficient**: Handles large datasets (1M+ events)
- **Vectorized operations**: Pandas/NumPy for fast computation
- **Smart filtering**: Removes noise while preserving signal

---

**Built for analysts and data scientists who need to understand behavioral differences between any two user groups.**
