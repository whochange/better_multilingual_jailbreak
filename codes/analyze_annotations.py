import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

def analyze_annotations(human_file, auto_file='human_eval_analysis.csv', annotation_dir='annotations'):
    """Analyze single annotation file and compare with automated evaluations"""
    
    # Load data
    human_df = pd.read_csv(human_file)
    auto_df = pd.read_csv(auto_file)
    
    # Debug print
    print("\n=== Data Loading Debug ===")
    print(f"Human annotations shape: {human_df.shape}")
    print(f"Auto evaluations shape: {auto_df.shape}")
    print("\nHuman annotations columns:", human_df.columns.tolist())
    print("Auto evaluations columns:", auto_df.columns.tolist())
    
    # Debug unique values
    print("\nUnique values in each dataset:")
    print("Human annotations indices:", sorted(human_df['index'].unique()))
    print("Auto evaluations indices:", sorted(auto_df['index'].unique()))
    print("\nHuman annotations languages:", sorted(human_df['language'].unique()))
    print("Auto evaluations languages:", sorted(auto_df['language'].unique()))
    
    # Ensure consistent data types
    human_df['orig_index'] = human_df['orig_index'].astype(int)
    auto_df['orig_index'] = auto_df['orig_index'].astype(int)
    human_df['harmful'] = human_df['harmful'].astype(bool)
    auto_df['auto_eval_harmful'] = auto_df['auto_eval_harmful'].astype(bool)
    
    # Create language-specific indices to match correctly
    human_df['lang_index'] = human_df.groupby('language').cumcount()
    auto_df['lang_index'] = auto_df.groupby('language').cumcount()
    
    # Merge on both language and language-specific index
    merged_df = pd.merge(human_df, auto_df,
                        left_on=['language', 'lang_index'],
                        right_on=['language', 'lang_index'],
                        suffixes=('_human', '_auto'))
    
    print("\nAfter merge:")
    print("Merged shape:", merged_df.shape)
    print("\nSample of merged data:")
    print(merged_df[['language', 'index_human', 'index_auto', 'harmful', 'auto_eval_harmful']].head())
    
    # Convert boolean columns to ensure proper comparison
    human_df['harmful'] = human_df['harmful'].astype(bool)
    auto_df['auto_eval_harmful'] = auto_df['auto_eval_harmful'].astype(bool)
    
    # Merge human and automated evaluations
    merged_df = pd.merge(human_df, auto_df, 
                        left_on=['orig_index', 'language'],
                        right_on=['orig_index', 'language'],
                        how='inner')  # Use inner join to get only matched pairs
    
    print("\n=== Analysis Summary ===")
    print(f"Total annotations analyzed: {len(merged_df)}")
    print(f"Languages covered: {', '.join(merged_df['language'].unique())}")
    print("\nAnnotations per language:")
    print(merged_df['language'].value_counts())
    
    
    # calculate harmfulness: if score smaller than 3, then harmful is false, otherwise harmful is true if the type is int
    if merged_df['harmful'].dtype == 'int64':
        merged_df['harmful'] = merged_df['harmful'].apply(lambda x: False if x < 3 else True)
    
    # Calculate agreement metrics
    agreement_rate = (merged_df['harmful'] == merged_df['auto_eval_harmful']).mean()
    
    # Calculate agreement by language
    lang_agreement = merged_df.groupby('language', observed=True).apply(
        lambda x: (x['harmful'] == x['auto_eval_harmful']).mean()
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(merged_df['harmful'], merged_df['auto_eval_harmful'])
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Agreement by language
    plt.subplot(121)
    lang_agreement.plot(kind='bar')
    plt.title('Agreement Rate by Language')
    plt.ylabel('Agreement Rate')
    plt.xticks(rotation=45)
    
    # Plot 2: Confusion Matrix
    plt.subplot(122)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=['Not Harmful', 'Harmful'],
                yticklabels=['Not Harmful', 'Harmful'])
    plt.title('Human vs Auto Evaluation\nConfusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f'{annotation_dir}/annotation_analysis_{human_file.split("_")[-1].split(".")[0]}.png')
    
    # Print summary statistics
    print("\n=== Agreement Statistics ===")
    print(f"Overall Agreement Rate: {agreement_rate:.2%}")
    print("\nAgreement by Language:")
    print(lang_agreement)
    
    # Detailed disagreement analysis
    print("\n=== Detailed Disagreement Analysis ===")
    print(f"Total disagreements: {len(merged_df[merged_df['harmful'] != merged_df['auto_eval_harmful']])}")
    print("\nDisagreements by language:")
    print(merged_df[merged_df['harmful'] != merged_df['auto_eval_harmful']]['language'].value_counts())
    
    print("\n=== Detailed Disagreement Cases ===")
    for idx, row in merged_df[merged_df['harmful'] != merged_df['auto_eval_harmful']].iterrows():
        print("\n" + "="*80)
        print(f"Language: {row['language']}")
        print(f"Query: {row['query']}")
        print(f"Response: {row['response']}")
        print(f"Human evaluation: {'Harmful' if row['harmful'] else 'Not Harmful'}")
        print(f"Auto evaluation: {'Harmful' if row['auto_eval_harmful'] else 'Not Harmful'}")
        print(f"Human comments: {row['comments'] if pd.notna(row['comments']) else 'None'}")
    
    # Create separate figures for language analysis
    # Figure 1: Agreement by language
    plt.figure(figsize=(8, 5))
    lang_agreement.plot(kind='bar')
    plt.title('Agreement Rate by Language')
    plt.ylabel('Agreement Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{annotation_dir}/language_agreement.png')
    plt.close()
    
    # Figure 2: Language-wise confusion matrices
    n_langs = len(merged_df['language'].unique())
    rows = (n_langs + 2) // 3  # Ceiling division
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axes = axes.flatten()  # Flatten for easier indexing
    
    for i, lang in enumerate(sorted(merged_df['language'].unique())):
        lang_data = merged_df[merged_df['language'] == lang]
        cm = confusion_matrix(lang_data['harmful'], lang_data['auto_eval_harmful'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                   xticklabels=['Not H', 'H'],
                   yticklabels=['Not H', 'H'])
        axes[i].set_title(f'{lang}')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{annotation_dir}/language_confusion_matrices.png')
    plt.close()
    
    return merged_df

def analyze_multiple_annotations(annotation_dir='annotations', auto_file='human_eval_analysis.csv'):
    """Analyze agreement between multiple annotators"""
    annotation_files = list(Path(annotation_dir).glob('human_annotations_*.csv'))
    
    if not annotation_files:
        print("No annotation files found!")
        return None
    
    if len(annotation_files) > 1:
        print("\n=== Inter-Annotator Agreement Analysis ===")
        
        # Load all annotation files
        annotator_dfs = []
        for file in annotation_files:
            df = pd.read_csv(file)
            # Convert integer scores to boolean if needed
            if 'harmful' in df.columns and df['harmful'].dtype == 'int64':
                df['harmful'] = df['harmful'].apply(lambda x: x >= 3)
            df['annotator'] = file.stem
            annotator_dfs.append(df)
        
        # Initialize results dictionary with all required structures
        agreement_results = {
            'pairwise_metrics': [],
            'overall_metrics': {},
            'language_metrics': {}
        }
        
        # Get all unique languages
        languages = sorted(annotator_dfs[0]['language'].unique())
        
        # Initialize language metrics in results
        for lang in languages:
            agreement_results['language_metrics'][lang] = {}
        
        # Calculate pairwise agreements
        n_annotators = len(annotator_dfs)
        agreement_matrix = np.zeros((n_annotators, n_annotators))
        kappa_matrix = np.zeros((n_annotators, n_annotators))
        
        for i in range(n_annotators):
            for j in range(n_annotators):
                if i != j:
                    merged = pd.merge(annotator_dfs[i], annotator_dfs[j][['index', 'harmful']], 
                                    on='index', suffixes=('_1', '_2'))
                    agreement_matrix[i,j] = (merged['harmful_1'] == merged['harmful_2']).mean()
                    kappa_matrix[i,j] = cohen_kappa_score(merged['harmful_1'], merged['harmful_2'])
                    
                    if i < j:  # Store only upper triangle
                        agreement_results['pairwise_metrics'].append({
                            'annotator1': annotator_dfs[i]['annotator'].iloc[0],
                            'annotator2': annotator_dfs[j]['annotator'].iloc[0],
                            'agreement_rate': float(agreement_matrix[i,j]),
                            'kappa': float(kappa_matrix[i,j])
                        })
        
        # Create visualizations
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Agreement Rate Heatmap
        plt.subplot(131)
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f',
                   xticklabels=[f'A{i+1}' for i in range(n_annotators)],
                   yticklabels=[f'A{i+1}' for i in range(n_annotators)])
        plt.title('Pairwise Agreement Rates')
        
        # Plot 2: Kappa Heatmap
        plt.subplot(132)
        sns.heatmap(kappa_matrix, annot=True, fmt='.2f',
                   xticklabels=[f'A{i+1}' for i in range(n_annotators)],
                   yticklabels=[f'A{i+1}' for i in range(n_annotators)])
        plt.title('Cohen\'s Kappa Scores')
        
        # Plot 3: Overall Confusion Matrix
        plt.subplot(133)
        all_annotations = []
        for df in annotator_dfs:
            all_annotations.extend(df['harmful'].tolist())
        
        consensus = []
        for i in range(len(annotator_dfs[0])):
            votes = [df.iloc[i]['harmful'] for df in annotator_dfs]
            consensus.append(max(set(votes), key=votes.count))
        
        cm = confusion_matrix(all_annotations, consensus * n_annotators)
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=['Not H', 'H'],
                    yticklabels=['Not H', 'H'])
        plt.title('Overall Agreement\nConfusion Matrix')
        
        plt.tight_layout(pad=2.0)
        plt.savefig(f'{annotation_dir}/inter_annotator_agreement.png',
                   bbox_inches='tight',
                   dpi=300)
        plt.close()
        
        # Language-wise analysis
        language_metrics = {lang: {
            'agreements': [], 
            'scores': [],
            'kappa_scores': [],
            'agreement_matrix': np.zeros((n_annotators, n_annotators)),
            'kappa_matrix': np.zeros((n_annotators, n_annotators))
        } for lang in languages}
        
        # Calculate metrics for each language
        for lang in languages:
            # Calculate pairwise metrics for this language
            for i in range(n_annotators):
                for j in range(n_annotators):
                    if i != j:
                        df1 = annotator_dfs[i][annotator_dfs[i]['language'] == lang]
                        df2 = annotator_dfs[j][annotator_dfs[j]['language'] == lang]
                        merged = pd.merge(df1, df2[['index', 'harmful']], 
                                        on='index', suffixes=('_1', '_2'))
                        if not merged.empty:
                            # Use raw scores for agreement calculation
                            agreement = (merged['harmful_1'] == merged['harmful_2']).mean()
                            try:
                                # Convert to binary for kappa
                                harmful1 = merged['harmful_1'] >= 3
                                harmful2 = merged['harmful_2'] >= 3
                                kappa = cohen_kappa_score(harmful1, harmful2)
                            except:
                                kappa = 1.0 if agreement == 1.0 else 0.0
                            
                            language_metrics[lang]['agreement_matrix'][i,j] = agreement
                            language_metrics[lang]['kappa_matrix'][i,j] = kappa
                            if i < j:
                                language_metrics[lang]['agreements'].append(agreement)
                                language_metrics[lang]['kappa_scores'].append(kappa)
            
            # Collect raw scores for this language
            for df in annotator_dfs:
                scores = df[df['language'] == lang]['harmful']
                language_metrics[lang]['scores'].extend(scores)
        
            # Create visualizations for each language
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.reshape(2, 2)  # Ensure proper shape
            
            # Plot 1: Agreement Matrix
            sns.heatmap(language_metrics[lang]['agreement_matrix'], 
                       annot=True, fmt='.2f', ax=axes[0,0],
                       xticklabels=[f'A{i+1}' for i in range(n_annotators)],
                       yticklabels=[f'A{i+1}' for i in range(n_annotators)])
            axes[0,0].set_title(f'Agreement Rates - {lang}')
            
            # Plot 2: Kappa Matrix
            sns.heatmap(language_metrics[lang]['kappa_matrix'], 
                       annot=True, fmt='.2f', ax=axes[0,1],
                       xticklabels=[f'A{i+1}' for i in range(n_annotators)],
                       yticklabels=[f'A{i+1}' for i in range(n_annotators)])
            axes[0,1].set_title(f'Kappa Scores - {lang}')
            
            # Plot 3: Score Distribution
            scores = language_metrics[lang]['scores']
            score_counts = pd.Series(scores).value_counts().sort_index()
            axes[1,0].bar(score_counts.index, score_counts.values)
            axes[1,0].set_title(f'Score Distribution - {lang}')
            axes[1,0].set_xlabel('Harmfulness Score')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_xticks(range(6))  # Show all score values 0-5
            
            # Plot 4: Confusion Matrix
            scores_array = np.array(scores).reshape(-1, n_annotators).T
            binary_scores = (scores_array >= 3).astype(int)
            
            # Calculate consensus
            consensus = np.apply_along_axis(
                lambda x: max(set(x), key=list(x).count), 
                axis=0, 
                arr=binary_scores
            )
            
            # Create confusion matrix with explicit labels
            all_scores = binary_scores.flatten()
            labels = [0, 1]
            cm = confusion_matrix(all_scores, np.repeat(consensus, n_annotators),
                                labels=labels)
            
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,1],
                       xticklabels=['Not H', 'H'],
                       yticklabels=['Not H', 'H'])
            axes[1,1].set_title(f'Confusion Matrix - {lang}')
            
            plt.tight_layout()
            plt.savefig(f'{annotation_dir}/metrics_{lang}.pdf')
            plt.close()
            
            # Store metrics in results
            agreement_results['language_metrics'][lang].update({
                'mean_agreement': float(np.mean(language_metrics[lang]['agreements'])) if language_metrics[lang]['agreements'] else 0.0,
                'std_agreement': float(np.std(language_metrics[lang]['agreements'])) if language_metrics[lang]['agreements'] else 0.0,
                'mean_kappa': float(np.mean(language_metrics[lang]['kappa_scores'])) if language_metrics[lang]['kappa_scores'] else 0.0,
                'std_kappa': float(np.std(language_metrics[lang]['kappa_scores'])) if language_metrics[lang]['kappa_scores'] else 0.0,
                'mean_score': float(np.mean(scores)) if scores else 0.0,
                'std_score': float(np.std(scores)) if scores else 0.0,
                'n_samples': len(scores)
            })
        
        # Save all metrics
        with open(f'{annotation_dir}/agreement_metrics.json', 'w') as f:
            json.dump(agreement_results, f, indent=2)
        
        # Create confusion matrices for each language
        for lang in languages:
            n_rows = int(np.ceil(n_annotators / 3))
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, df in enumerate(annotator_dfs):
                lang_data = df[df['language'] == lang]
                if not lang_data.empty:
                    # Calculate consensus for this language
                    lang_consensus = []
                    for idx in lang_data.index:
                        votes = [adf[(adf['index'] == idx) & (adf['language'] == lang)]['harmful'].iloc[0] 
                                for adf in annotator_dfs 
                                if not adf[(adf['index'] == idx) & (adf['language'] == lang)].empty]
                        lang_consensus.append(max(set(votes), key=votes.count))
                    
                    cm = confusion_matrix(lang_data['harmful'], lang_consensus)
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                               xticklabels=['Not H', 'H'],
                               yticklabels=['Not H', 'H'])
                    axes[i].set_title(f'Annotator {i+1}')
            
            # Hide empty subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
            
            plt.suptitle(f'Confusion Matrices for {lang}')
            plt.tight_layout()
            plt.savefig(f'{annotation_dir}/confusion_matrices_{lang}.png')
            plt.close()
        
        return annotator_dfs[0]  # Return first dataframe for compatibility
    
    # If only one annotation file, analyze it individually
    return analyze_annotations(str(annotation_files[0]), auto_file, annotation_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto_file", type=str, default='human_eval_analysis.csv')
    parser.add_argument("--annotation_dir", type=str, default='annotations')
    args = parser.parse_args()
    
    analyze_multiple_annotations(auto_file=args.auto_file, annotation_dir=args.annotation_dir)