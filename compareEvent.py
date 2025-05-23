import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class UserBehaviorAnalyzer:
    def __init__(self, quitters_file, retainers_file):
        """
        Initialize the analyzer with CSV file paths
        
        Args:
            quitters_file (str): Path to CSV file with quitter user logs
            retainers_file (str): Path to CSV file with retainer user logs
        """
        self.quitters_file = '/Users/chengxuanxia/Downloads/quit.csv'
        self.retainers_file = '/Users/chengxuanxia/Downloads/retain.csv'
        self.quitters_data = None
        self.retainers_data = None
        self.combined_data = None
        self.user_sequences = None
        self.vectorized_data = None
        self.vectorizer = None
        
    def load_data(self):
        """Load and prepare the CSV files"""
        print("Loading data...")
        
        # Load CSV files
        print("Loading quitters data...")
        self.quitters_data = pd.read_csv(self.quitters_file)
        print("Loading retainers data...")
        self.retainers_data = pd.read_csv(self.retainers_file)
        
        # Clean data
        self.quitters_data = self.quitters_data.dropna()
        self.retainers_data = self.retainers_data.dropna()
        
        # Add labels
        self.quitters_data['user_type'] = 'quitter'
        self.retainers_data['user_type'] = 'retainer'
        
        # Combine datasets
        self.combined_data = pd.concat([self.quitters_data, self.retainers_data], ignore_index=True)
        
        print(f"Loaded {len(self.quitters_data)} quitter events from {self.quitters_data['user_info_id'].nunique()} unique users")
        print(f"Loaded {len(self.retainers_data)} retainer events from {self.retainers_data['user_info_id'].nunique()} unique users")
        print(f"Total events: {len(self.combined_data)}")
        print(f"Columns: {list(self.combined_data.columns)}")
    
    def explore_data(self):
        """Explore the basic characteristics of the data"""
        print("\n=== DATA EXPLORATION ===")
        
        # Basic statistics
        print(f"Total unique users: {self.combined_data['user_info_id'].nunique()}")
        print(f"  - Quitters: {self.quitters_data['user_info_id'].nunique()}")
        print(f"  - Retainers: {self.retainers_data['user_info_id'].nunique()}")
        print(f"Total events: {len(self.combined_data)}")
        print(f"  - Quitter events: {len(self.quitters_data)}")
        print(f"  - Retainer events: {len(self.retainers_data)}")
        print(f"Unique event types: {self.combined_data['event_name'].nunique()}")
        
        # Events per user statistics
        events_per_user = self.combined_data.groupby(['user_info_id', 'user_type']).size()
        quitter_events_per_user = events_per_user[events_per_user.index.get_level_values('user_type') == 'quitter']
        retainer_events_per_user = events_per_user[events_per_user.index.get_level_values('user_type') == 'retainer']
        
        print(f"\nEvents per user statistics:")
        print(f"  Quitters - Mean: {quitter_events_per_user.mean():.1f}, Median: {quitter_events_per_user.median():.1f}")
        print(f"  Retainers - Mean: {retainer_events_per_user.mean():.1f}, Median: {retainer_events_per_user.median():.1f}")
        
        # Most common events overall
        print("\nTop 15 most common events:")
        top_events = self.combined_data['event_name'].value_counts().head(15)
        for event, count in top_events.items():
            print(f"  {event}: {count:,}")
        
        # Event distribution by user type
        print("\nEvent distribution comparison (Top 10):")
        event_counts = self.combined_data.groupby(['event_name', 'user_type']).size().unstack(fill_value=0)
        event_counts['total'] = event_counts.sum(axis=1)
        event_counts['quitter_pct'] = (event_counts['quitter'] / event_counts['total'] * 100)
        top_events_comparison = event_counts.nlargest(10, 'total')
        
        print(top_events_comparison[['quitter', 'retainer', 'quitter_pct']].to_string())
    
    def create_user_sequences(self):
        """Create event sequences for each user"""
        print("\n=== CREATING USER EVENT SEQUENCES ===")
        
        # Group events by user and create sequences
        print("Grouping events by user...")
        self.user_sequences = {}
        
        for user_id in self.combined_data['user_info_id'].unique():
            user_events = self.combined_data[self.combined_data['user_info_id'] == user_id]['event_name'].tolist()
            self.user_sequences[user_id] = user_events
        
        print(f"Created sequences for {len(self.user_sequences)} users")
        print(f"Average events per user: {np.mean([len(seq) for seq in self.user_sequences.values()]):.1f}")
        print(f"Max events per user: {max([len(seq) for seq in self.user_sequences.values()])}")
        print(f"Min events per user: {min([len(seq) for seq in self.user_sequences.values()])}")
        
        # Show sample sequences
        print("\nSample event sequences:")
        sample_users = list(self.user_sequences.keys())[:3]
        for user_id in sample_users:
            user_type = 'retainer' if user_id in self.retainers_data['user_info_id'].values else 'quitter'
            sequence_preview = ' '.join(self.user_sequences[user_id][:10]) + "..."
            print(f"  {user_type} user {user_id}: {sequence_preview} ({len(self.user_sequences[user_id])} events)")
    
    def vectorize_sequences(self, method='count'):
        """Convert user sequences to numerical vectors"""
        print(f"\n=== VECTORIZING SEQUENCES ({method.upper()}) ===")
        
        if method == 'count':
            # Count vectorizer - frequency of each event
            vectorizer = CountVectorizer(min_df=2, max_df=0.95, token_pattern=r'\b\w+\b')
            
        elif method == 'tfidf':
            # TF-IDF vectorizer - emphasizes distinctive events
            vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, token_pattern=r'\b\w+\b')
            
        elif method == 'normalized_count':
            # Normalized count - each user's events sum to 1.0
            vectorizer = CountVectorizer(min_df=2, max_df=0.95, token_pattern=r'\b\w+\b')
        
        print("Fitting vectorizer...")
        # Fit on all sequences
        sequences = [' '.join(seq) for seq in self.user_sequences.values()]
        matrix = vectorizer.fit_transform(sequences)
        
        print("Converting to DataFrame...")
        feature_names = vectorizer.get_feature_names_out()
        
        # Create DataFrame
        vectorized_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
        
        # CRITICAL: Normalize for count method to address activity bias
        if method == 'normalized_count':
            print("Applying row-wise normalization to remove activity bias...")
            # Normalize each row so all events sum to 1.0 for each user
            row_sums = vectorized_df.sum(axis=1)
            vectorized_df = vectorized_df.div(row_sums, axis=0).fillna(0)
            print("Normalization applied: each user's events now sum to 1.0")
        
        # Add user info
        vectorized_df['user_info_id'] = list(self.user_sequences.keys())
        
        # Add user type
        vectorized_df['user_type'] = vectorized_df['user_info_id'].map(
            lambda x: 'retainer' if x in self.retainers_data['user_info_id'].values else 'quitter'
        )
        
        print(f"Created {len(feature_names)} features from {len(self.user_sequences)} users")
        print(f"Feature matrix shape: {vectorized_df.shape}")
        
        # Calculate sparsity
        non_zero_count = (vectorized_df.select_dtypes(include=[np.number]) != 0).sum().sum()
        total_count = vectorized_df.select_dtypes(include=[np.number]).size
        sparsity_pct = (non_zero_count / total_count) * 100
        print(f"Matrix sparsity: {sparsity_pct:.2f}% non-zero")
        
        self.vectorized_data = vectorized_df
        self.vectorizer = vectorizer
    
    def analyze_event_differences(self):
        """Analyze differences in event patterns between quitters and retainers"""
        print("\n=== ANALYZING EVENT DIFFERENCES ===")
        
        # Separate quitters and retainers
        quitters_vector = self.vectorized_data[self.vectorized_data['user_type'] == 'quitter']
        retainers_vector = self.vectorized_data[self.vectorized_data['user_type'] == 'retainer']
        
        # Calculate mean frequency for each event type (exclude metadata columns)
        event_columns = [col for col in self.vectorized_data.columns 
                        if col not in ['user_type', 'user_info_id', 'event_count']]
        
        quitters_mean = quitters_vector[event_columns].mean()
        retainers_mean = retainers_vector[event_columns].mean()
        
        # Calculate differences
        differences = retainers_mean - quitters_mean
        
        # NORMALIZATION ANALYSIS - Calculate proportional differences
        print("Calculating normalized (proportional) differences...")
        
        # Calculate proportions (each event / total events per user)
        quitters_proportions = quitters_vector[event_columns].div(
            quitters_vector[event_columns].sum(axis=1), axis=0
        ).mean()
        
        retainers_proportions = retainers_vector[event_columns].div(
            retainers_vector[event_columns].sum(axis=1), axis=0
        ).mean()
        
        # Calculate proportional differences
        proportional_differences = retainers_proportions - quitters_proportions
        
        # Create comparison DataFrame with both raw and normalized results
        comparison_df = pd.DataFrame({
            'event': event_columns,
            'quitters_mean': quitters_mean.values,
            'retainers_mean': retainers_mean.values,
            'raw_difference': differences.values,
            'quitters_proportion': quitters_proportions.values,
            'retainers_proportion': retainers_proportions.values,
            'proportional_difference': proportional_differences.values,
            'abs_raw_difference': np.abs(differences.values),
            'abs_proportional_difference': np.abs(proportional_differences.values)
        })
        
        # Sort by absolute raw difference first
        comparison_df = comparison_df.sort_values('abs_raw_difference', ascending=False)
        
        print("Top 15 events by RAW frequency differences:")
        print(comparison_df.head(15)[['event', 'quitters_mean', 'retainers_mean', 'raw_difference']])
        
        # Sort by absolute proportional difference
        comparison_df_prop = comparison_df.sort_values('abs_proportional_difference', ascending=False)
        
        print("\nTop 15 events by PROPORTIONAL differences (normalized by user activity):")
        print(comparison_df_prop.head(15)[['event', 'quitters_proportion', 'retainers_proportion', 'proportional_difference']])
        
        return comparison_df, comparison_df_prop
    
    def feature_importance_analysis(self):
        """Use machine learning to identify important features"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Prepare data (exclude metadata columns)
        event_columns = [col for col in self.vectorized_data.columns 
                        if col not in ['user_type', 'user_info_id', 'event_count']]
        X = self.vectorized_data[event_columns]
        y = (self.vectorized_data['user_type'] == 'retainer').astype(int)  # 1 for retainer, 0 for quitter
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'event': event_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 15 most important events (Random Forest):")
        print(importance_df.head(15))
        
        # Model performance
        y_pred = rf.predict(X_test)
        print(f"\nModel Accuracy: {rf.score(X_test, y_test):.3f}")
        
        # Logistic Regression for interpretable coefficients
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        
        # Get coefficients (log-odds)
        coef_df = pd.DataFrame({
            'event': event_columns,
            'coefficient': lr.coef_[0],
            'abs_coefficient': np.abs(lr.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\nTop 15 events with strongest coefficients (Logistic Regression):")
        print(coef_df.head(15)[['event', 'coefficient']])
        
        return importance_df, coef_df
    
    def visualize_results(self, comparison_df, importance_df):
        """Create visualizations of the results"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top event differences (using raw_difference)
        top_diff = comparison_df.head(10)
        axes[0, 0].barh(range(len(top_diff)), top_diff['raw_difference'], 
                       color=['red' if x < 0 else 'green' for x in top_diff['raw_difference']])
        axes[0, 0].set_yticks(range(len(top_diff)))
        axes[0, 0].set_yticklabels(top_diff['event'], fontsize=8)
        axes[0, 0].set_xlabel('Raw Difference (Retainers - Quitters)')
        axes[0, 0].set_title('Top 10 Event Differences (Raw Counts)')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Feature importance
        top_importance = importance_df.head(10)
        axes[0, 1].barh(range(len(top_importance)), top_importance['importance'])
        axes[0, 1].set_yticks(range(len(top_importance)))
        axes[0, 1].set_yticklabels(top_importance['event'], fontsize=8)
        axes[0, 1].set_xlabel('Feature Importance')
        axes[0, 1].set_title('Top 10 Most Important Events (Random Forest)')
        
        # 3. NORMALIZED Event frequency comparison (proportions)
        print("Creating normalized frequency scatter plot...")
        
        # Use proportional data for the scatter plot
        quitters_proportions = comparison_df['quitters_proportion']
        retainers_proportions = comparison_df['retainers_proportion']
        
        # Convert to percentages for better readability
        quitters_pct = quitters_proportions * 100
        retainers_pct = retainers_proportions * 100
        
        # Debug: Check for duplicates and data quality
        print(f"Total events in comparison_df: {len(comparison_df)}")
        print(f"Unique events: {comparison_df['event'].nunique()}")
        if len(comparison_df) != comparison_df['event'].nunique():
            print("⚠️  WARNING: Duplicate events detected!")
            duplicates = comparison_df[comparison_df['event'].duplicated()]['event'].tolist()
            print(f"Duplicate events: {duplicates}")
        
        # Filter out extremely low-frequency events for better visualization
        # Keep events where at least one group has >0.1% activity
        min_threshold = 0.1
        visible_mask = (quitters_pct >= min_threshold) | (retainers_pct >= min_threshold)
        
        # Remove extreme outliers that skew the plot scale
        # Calculate the 95th percentile for both axes to identify outliers
        q_95th = quitters_pct.quantile(0.95)
        r_95th = retainers_pct.quantile(0.95)
        
        # Keep points that are not extreme outliers
        outlier_mask = (quitters_pct <= q_95th * 2) & (retainers_pct <= r_95th * 2)
        
        # Combine both filters
        final_mask = visible_mask & outlier_mask
        
        visible_quitters = quitters_pct[final_mask]
        visible_retainers = retainers_pct[final_mask]
        visible_events = comparison_df[final_mask]['event']
        
        # Identify what we filtered out
        outliers_removed = comparison_df[visible_mask & ~outlier_mask]
        if len(outliers_removed) > 0:
            print(f"\n🎯 Removed {len(outliers_removed)} extreme outlier(s) for better visualization:")
            for _, row in outliers_removed.iterrows():
                print(f"  {row['event']}: Q={row['quitters_proportion']*100:.2f}%, R={row['retainers_proportion']*100:.2f}%")
        
        print(f"Events above {min_threshold}% threshold: {len(comparison_df[visible_mask])}")
        print(f"Events shown in plot (excluding outliers): {len(visible_events)}")
        print(f"Unique visible events: {visible_events.nunique()}")
        
        # Show some examples of what's being plotted
        print("\nSample of events being plotted:")
        sample_visible = comparison_df[visible_mask].head(10)
        for _, row in sample_visible.iterrows():
            print(f"  {row['event']}: Q={row['quitters_proportion']*100:.2f}%, R={row['retainers_proportion']*100:.2f}%")
        
        # Create scatter plot with better sizing and colors
        # Color points based on which group favors them more
        colors = ['red' if q > r else 'blue' for q, r in zip(visible_quitters, visible_retainers)]
        sizes = [max(30, abs(q-r)*10) for q, r in zip(visible_quitters, visible_retainers)]  # Size by difference magnitude
        
        scatter = axes[1, 0].scatter(visible_quitters, visible_retainers, 
                                   c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (where proportions would be equal)
        max_val = max(visible_quitters.max(), visible_retainers.max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'gray', linestyle='--', alpha=0.5, 
                       label='Equal Proportions')
        
        axes[1, 0].set_xlabel('Quitters: % of Total Activity')
        axes[1, 0].set_ylabel('Retainers: % of Total Activity')
        axes[1, 0].set_title('Event Proportions: Quitters vs Retainers (NORMALIZED)\nRed=Quitter-favored, Blue=Retainer-favored')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Smart annotation: Only label the most interesting points
        # Find events with biggest proportional differences that are visible
        visible_comparison = comparison_df[visible_mask].copy()
        top_interesting = visible_comparison.nlargest(5, 'abs_proportional_difference')
        
        for _, row in top_interesting.iterrows():
            q_pct = row['quitters_proportion'] * 100
            r_pct = row['retainers_proportion'] * 100
            event_name = row['event'][:12] + '...' if len(row['event']) > 12 else row['event']
            
            # Smart label positioning to avoid overlap
            offset_x = 5 if q_pct < max_val * 0.7 else -25
            offset_y = 5 if r_pct < max_val * 0.7 else -10
            
            axes[1, 0].annotate(event_name, (q_pct, r_pct), 
                              xytext=(offset_x, offset_y), textcoords='offset points',
                              fontsize=6, alpha=0.9,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # Add text explanation
        outlier_text = f"(Excluded {len(outliers_removed)} outlier(s))" if len(outliers_removed) > 0 else ""
        axes[1, 0].text(0.02, 0.98, f'Showing {len(visible_events)} events ≥{min_threshold}% activity {outlier_text}\nPoint size = difference magnitude\nLabeled: Top 5 most different events', 
                       transform=axes[1, 0].transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Print a comprehensive list of all visible events for reference
        print(f"\n📋 ALL {len(visible_events)} VISIBLE EVENTS (≥{min_threshold}% activity):")
        visible_detailed = comparison_df[visible_mask].sort_values('abs_proportional_difference', ascending=False)
        
        print("Top 20 most behaviorally different events:")
        for i, (_, row) in enumerate(visible_detailed.head(20).iterrows(), 1):
            q_pct = row['quitters_proportion'] * 100
            r_pct = row['retainers_proportion'] * 100
            diff = row['proportional_difference'] * 100
            favor = "R" if diff > 0 else "Q"
            print(f"{i:2d}. {row['event'][:40]:40} | Q:{q_pct:5.2f}% R:{r_pct:5.2f}% | {favor}-favored")
        
        if len(visible_detailed) > 20:
            print(f"\n... and {len(visible_detailed) - 20} more events with smaller differences.")
            print("\nTo see all events, check the CSV output files generated at the end.")
        
        # 4. CHURN SIGNALS - Events that indicate friction/problems
        print("Analyzing potential churn signals...")
        
        # Find events that quitters do proportionally MORE than retainers
        churn_signals = comparison_df[comparison_df['proportional_difference'] < 0].copy()
        churn_signals['churn_signal_strength'] = abs(churn_signals['proportional_difference'])
        churn_signals = churn_signals.nlargest(10, 'churn_signal_strength')
        
        # Create churn signals visualization
        if len(churn_signals) > 0:
            # Convert to percentages and calculate the "signal strength"
            churn_signals['quitters_pct'] = churn_signals['quitters_proportion'] * 100
            churn_signals['retainers_pct'] = churn_signals['retainers_proportion'] * 100
            churn_signals['signal_strength_pct'] = churn_signals['churn_signal_strength'] * 100
            
            # Create horizontal bar chart showing churn signal strength
            y_pos = range(len(churn_signals))
            axes[1, 1].barh(y_pos, churn_signals['signal_strength_pct'], 
                           color='red', alpha=0.7)
            
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                      for name in churn_signals['event']], fontsize=8)
            axes[1, 1].set_xlabel('Churn Signal Strength (% points)')
            axes[1, 1].set_title('🚨 Potential Churn Signals\n(Events Quitters Do More)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add annotations showing the actual percentages
            for i, (_, row) in enumerate(churn_signals.iterrows()):
                axes[1, 1].annotate(f"Q:{row['quitters_pct']:.1f}% R:{row['retainers_pct']:.1f}%", 
                                   (row['signal_strength_pct'] + 0.1, i),
                                   va='center', fontsize=7, alpha=0.8)
            
            # Print detailed analysis
            print("\n🚨 TOP CHURN SIGNALS (Events quitters prioritize more):")
            for _, row in churn_signals.head(5).iterrows():
                print(f"• {row['event']}: {row['quitters_pct']:.1f}% vs {row['retainers_pct']:.1f}% "
                      f"(+{row['signal_strength_pct']:.1f}pp churn signal)")
                
        else:
            # Fallback if no clear churn signals
            axes[1, 1].text(0.5, 0.5, 'No clear churn signals detected\n(Retainers more active in all areas)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('🚨 Churn Signal Analysis')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('user_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print explanation of the normalized scatter plot
        print("\n📊 NORMALIZED SCATTER PLOT INTERPRETATION:")
        print("• X-axis: What % of quitters' total activity this event represents")
        print("• Y-axis: What % of retainers' total activity this event represents")
        print("• Points ABOVE red line: Retainers prioritize this behavior more")
        print("• Points BELOW red line: Quitters prioritize this behavior more")
        print("• Distance from red line: How different the behavioral priorities are")
    
    def generate_insights(self, comparison_df, comparison_df_prop, importance_df, coef_df):
        """Generate actionable insights from the analysis"""
        print("\n=== KEY INSIGHTS ===")
        
        print("📊 RAW FREQUENCY ANALYSIS (Activity Level):")
        # Events that retainers do more
        retention_events = comparison_df[comparison_df['raw_difference'] > 0].head(5)
        print("Events that RETAINERS do more (absolute frequency):")
        for _, row in retention_events.iterrows():
            print(f"  • {row['event']}: +{row['raw_difference']:.1f} more per user")
        
        # Events that quitters do more
        churn_events = comparison_df[comparison_df['raw_difference'] < 0].head(5)
        print("\nEvents that QUITTERS do more (absolute frequency):")
        for _, row in churn_events.iterrows():
            print(f"  • {row['event']}: {abs(row['raw_difference']):.1f} more per user")
        
        print("\n📈 PROPORTIONAL ANALYSIS (Behavioral Preferences):")
        # Events with higher proportions for retainers
        retention_events_prop = comparison_df_prop[comparison_df_prop['proportional_difference'] > 0].head(5)
        print("Events that RETAINERS prioritize more (% of their activity):")
        for _, row in retention_events_prop.iterrows():
            q_pct = row['quitters_proportion'] * 100
            r_pct = row['retainers_proportion'] * 100
            diff_pct = row['proportional_difference'] * 100
            print(f"  • {row['event']}: {r_pct:.2f}% vs {q_pct:.2f}% (+{diff_pct:.2f}pp)")
        
        # Events with higher proportions for quitters
        churn_events_prop = comparison_df_prop[comparison_df_prop['proportional_difference'] < 0].head(5)
        print("\nEvents that QUITTERS prioritize more (% of their activity):")
        for _, row in churn_events_prop.iterrows():
            q_pct = row['quitters_proportion'] * 100
            r_pct = row['retainers_proportion'] * 100
            diff_pct = abs(row['proportional_difference']) * 100
            print(f"  • {row['event']}: {q_pct:.2f}% vs {r_pct:.2f}% (+{diff_pct:.2f}pp)")
        
        # Most predictive events
        print("\n🤖 MACHINE LEARNING INSIGHTS:")
        print("Most predictive events for retention:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  • {row['event']}: {row['importance']:.3f} importance")
        
        # Strongest positive/negative coefficients
        positive_coef = coef_df[coef_df['coefficient'] > 0].head(3)
        negative_coef = coef_df[coef_df['coefficient'] < 0].head(3)
        
        print("\nEvents most associated with retention (logistic regression):")
        for _, row in positive_coef.iterrows():
            print(f"  • {row['event']}: +{row['coefficient']:.3f} coefficient")
            
        print("\nEvents most associated with churn (logistic regression):")
        for _, row in negative_coef.iterrows():
            print(f"  • {row['event']}: {row['coefficient']:.3f} coefficient")
    
    def run_full_analysis(self, vectorization_method='normalized_count'):
        """Run the complete analysis pipeline"""
        # Load and explore data
        self.load_data()
        self.explore_data()
        
        # Create user sequences
        self.create_user_sequences()
        
        # Vectorize sequences
        self.vectorize_sequences(method=vectorization_method)
        
        # Analyze differences
        comparison_df, comparison_df_prop = self.analyze_event_differences()
        
        # Machine learning analysis
        importance_df, coef_df = self.feature_importance_analysis()
        
        # Visualizations
        self.visualize_results(comparison_df, importance_df)
        
        # Generate insights
        self.generate_insights(comparison_df, comparison_df_prop, importance_df, coef_df)
        
        return {
            'comparison_df': comparison_df,
            'comparison_df_prop': comparison_df_prop,
            'importance_df': importance_df,
            'coef_df': coef_df,
            'vectorized_data': self.vectorized_data
        }

if __name__ == "__main__":
    print("Starting analysis with your data...")
    
    # Run analysis with different methods to compare
    analyzer = UserBehaviorAnalyzer('quitters.csv', 'retainers.csv')
    
    print("\n" + "="*50)
    print("ANALYSIS 1: NORMALIZED COUNT (Recommended)")
    print("="*50)
    results_normalized = analyzer.run_full_analysis(vectorization_method='normalized_count')
    
    print("\n" + "="*50)
    print("ANALYSIS 2: TF-IDF (Alternative)")
    print("="*50)
    results_tfidf = analyzer.run_full_analysis(vectorization_method='tfidf')
    
    print("\n" + "="*50)
    print("ANALYSIS 3: RAW COUNT (Potentially Biased)")
    print("="*50)
    results_raw = analyzer.run_full_analysis(vectorization_method='count')
    
    print("\n" + "="*50)
    print("COMPARISON OF METHODS")
    print("="*50)
    print("Compare the feature importance rankings across methods.")
    print("If all methods agree → robust finding")
    print("If they differ significantly → activity bias detected")