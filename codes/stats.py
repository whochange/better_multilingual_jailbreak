import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentAnalyzer:
    def __init__(self, experiment_dir: str):
        self.exp_dir = Path(experiment_dir)
        self.stats = {
            'models': {},
            'languages': {},
            'overall': {
                'total_queries': 0,
                'success_rate': 0,
                'harmful_requests': 0,
                'harmful_responses': 0,
                'response_refusals': 0,
                'harmful_requests_en': 0,
                'harmful_responses_en': 0,
                'response_refusals_en': 0
            }
        }
    
    def analyze_model_results(self, model_dir: Path) -> Dict:
        """Analyze results for a single model"""
        logging.info(f"Analyzing model directory: {model_dir}")
        
        model_stats = {
            'total_queries': 0,
            'by_language': {},
            'safety_metrics': {
                'harmful_requests': 0,
                'harmful_responses': 0,
                'response_refusals': 0,
                'harmful_requests_en': 0,
                'harmful_responses_en': 0,
                'response_refusals_en': 0,
                'ref_harmful_sum_en': 0,
            }
        }
        
        # Process each language's results
        result_files = list(model_dir.glob("eval_results_*.json"))
        logging.info(f"Found {len(result_files)} result files")
        
        for result_file in result_files:
            # Get language code order by en zh it vi HRL ar ko th MRL bn sw jv LRL Avg.
            lang = result_file.stem.split('_')[-1] 
            logging.info(f"Processing language: {lang}")
            
            with open(result_file) as f:
                results = json.load(f)
            
            logging.info(f"Loaded {len(results)} results for {lang}")
            
            lang_stats = {
                'queries': len(results),
                'harmful_requests': 0,
                'harmful_responses': 0,
                'response_refusals': 0,
                'harmful_requests_en': 0,
                'harmful_responses_en': 0,
                'response_refusals_en': 0,
                'ref_harmful_sum_en': 0,
            }
            
            # Process each query
            for item in results:
                eval_results = item.get('evaluation', {})
                lang_stats['harmful_requests'] += int(eval_results.get('harmful_request', False))
                lang_stats['harmful_responses'] += int(eval_results.get('harmful_response', False))
                lang_stats['response_refusals'] += int(eval_results.get('response_refusal', False))
            
            logging.info(f"Language {lang} stats: {lang_stats}")
            
            # Process each query's translation
            for item in results:
                eval_results = item.get('evaluation_en', {})
                lang_stats['harmful_requests_en'] += int(eval_results.get('harmful_request', False))
                lang_stats['harmful_responses_en'] += int(eval_results.get('harmful_response', False))
                lang_stats['response_refusals_en'] += int(eval_results.get('response_refusal', False))
                lang_stats['ref_harmful_sum_en'] += int(eval_results.get('response_refusal', False)) + int(eval_results.get('harmful_response', False))
                
            model_stats['by_language'][lang] = lang_stats
            model_stats['total_queries'] += lang_stats['queries']
            model_stats['safety_metrics']['harmful_requests'] += lang_stats['harmful_requests']
            model_stats['safety_metrics']['harmful_responses'] += lang_stats['harmful_responses']
            model_stats['safety_metrics']['response_refusals'] += lang_stats['response_refusals']
            model_stats['safety_metrics']['harmful_requests_en'] += lang_stats['harmful_requests_en']
            model_stats['safety_metrics']['harmful_responses_en'] += lang_stats['harmful_responses_en']
            model_stats['safety_metrics']['response_refusals_en'] += lang_stats['response_refusals_en']
            model_stats['safety_metrics']['ref_harmful_sum_en'] += lang_stats['ref_harmful_sum_en']
        
        logging.info(f"Final model stats: {model_stats}")
        
        # save model_stats to a json file
        with open(model_dir / "model_stats.json", "w") as f:
            json.dump(model_stats, f, indent=2)
        
        return model_stats
    
    def analyze_experiment(self):
        """Analyze entire experiment directory"""
        logging.info(f"Analyzing experiment in: {self.exp_dir}")
        
        for model_dir in self.exp_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                logging.info(f"Processing model: {model_dir.name}")
                self.stats['models'][model_dir.name] = self.analyze_model_results(model_dir)
        
        # Calculate overall stats
        total_queries = sum(m['total_queries'] for m in self.stats['models'].values())
        self.stats['overall']['total_queries'] = total_queries
        
        # Save stats
        output_file = self.exp_dir / "experiment_stats.json"
        with open(output_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        logging.info(f"Saved stats to: {output_file}")
        return self.stats
    
    def generate_plots(self, output_dir: str = None):
        """Generate visualization plots"""
        if output_dir is None:
            output_dir = self.exp_dir / "plots"
        Path(output_dir).mkdir(exist_ok=True)
        
        logging.info("Preparing data for plotting")
        
        # Prepare data for plotting
        model_data = []
        for model_name, stats in self.stats['models'].items():
            safety = stats['safety_metrics']
            total = stats['total_queries']
            if total > 0:  # Avoid division by zero
                model_data.append({
                    'Model': model_name,  # Changed from 'model' to 'Model'
                  #  'Harmful Requests (%)': safety['harmful_requests'] / total * 100,
                  #  'Harmful Responses (%)': safety['harmful_responses'] / total * 100,
                   # 'Refusal Rate (%)': safety['response_refusals'] / total * 100,
                    'Harmful Requests (%) En': safety['harmful_requests_en'] / total * 100,
                    'Harmful Responses (%) En': safety['harmful_responses_en'] / total * 100,
                    'Refusal Rate (%) En': safety['response_refusals_en'] / total * 100,
                    'Refusal Harmful Sum (%) En': safety['ref_harmful_sum_en'] / total * 100,    
                })
        
        logging.info(f"Prepared plot data: {model_data}")
        
        df = pd.DataFrame(model_data)
        logging.info(f"DataFrame columns: {df.columns}")
        
        # Plot safety metrics by model
        plt.figure(figsize=(12, 6))
        df.plot(x='Model', 
               y=['Harmful Requests (%) En', 'Harmful Responses (%) En', 'Refusal Rate (%) En', 'Refusal Harmful Sum (%) En'],
               kind='bar')
        plt.title('Safety Metrics by Model')
        plt.xlabel('Model')
        plt.ylabel('Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/safety_metrics.png")
        plt.close()
        
        logging.info(f"Generated main plot at {output_dir}/safety_metrics.png")
        
        # Generate language-specific plots
        self._generate_language_plots(output_dir)
    
    def _generate_language_plots(self, output_dir: str):
        """Generate language-specific plots for each model"""
        # Set global font size
        plt.rcParams.update({'font.size': 14})
        
        for model_name, stats in self.stats['models'].items():
            lang_data = []
            lang_order = ['en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']
            
            for lang in lang_order:
                lang_stats = stats['by_language'].get(lang, {})
                total = lang_stats.get('queries', 0)
                if total > 0:
                    lang_data.append({
                        'Language': lang,
                        'Harmful Responses (%) En': lang_stats['harmful_responses_en'] / total * 100,
                        'Refusal Rate (%) En': lang_stats['response_refusals_en'] / total * 100
                    })
            
            if lang_data:
                df_lang = pd.DataFrame(lang_data)
                df_lang['Language'] = pd.Categorical(df_lang['Language'], 
                                                   categories=lang_order, 
                                                   ordered=True)
                df_lang = df_lang.sort_values('Language')
                
                # Create figure
                fig, ax1 = plt.subplots(figsize=(10, 7))
                
                # Plot bars
                x = np.arange(len(df_lang['Language']))
                width = 0.35
                ax1.bar(x - width/2, df_lang['Harmful Responses (%) En'], 
                       width, label='Harmful Responses', color='lightcoral')
                ax1.bar(x + width/2, df_lang['Refusal Rate (%) En'], 
                       width, label='Refusal Rate', color='royalblue')
                
                # Add vertical dotted lines between language groups
                ax1.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)  # HRL-MRL boundary
                ax1.axvline(x=6.5, color='gray', linestyle=':', alpha=0.5)  # MRL-LRL boundary
                
                # Calculate group means
                hrl_harmful = df_lang['Harmful Responses (%) En'][:4].mean()
                mrl_harmful = df_lang['Harmful Responses (%) En'][4:7].mean()
                lrl_harmful = df_lang['Harmful Responses (%) En'][7:].mean()
                
                hrl_refusal = df_lang['Refusal Rate (%) En'][:4].mean()
                mrl_refusal = df_lang['Refusal Rate (%) En'][4:7].mean()
                lrl_refusal = df_lang['Refusal Rate (%) En'][7:].mean()
                
                # Plot trend lines with points
                # Centers of each group (average position of languages in group)
                trend_x = [1.5, 5.0, 8.0]  # Adjusted for exact center of each group
                
                # Plot lines
                ax1.plot(trend_x, [hrl_harmful, mrl_harmful, lrl_harmful], 
                        '-', color='red', alpha=0.7, label='Harmful Trend')
                ax1.plot(trend_x, [hrl_refusal, mrl_refusal, lrl_refusal], 
                        '-', color='blue', alpha=0.7, label='Refusal Trend')
                
                # Add points at means
                ax1.scatter(trend_x, [hrl_harmful, mrl_harmful, lrl_harmful], 
                           color='red', s=40, zorder=5)
                ax1.scatter(trend_x, [hrl_refusal, mrl_refusal, lrl_refusal], 
                           color='blue', s=40, zorder=5)
                
                # Customize plot
                ax1.set_xlabel('Language', fontsize=18)
                ax1.set_ylabel('Rate (%)', fontsize=16)
                ax1.set_title(f'Safety Metrics by Language - {model_name}', fontsize=18)
                ax1.set_xticks(x)
                ax1.set_xticklabels(df_lang['Language'], rotation=45, fontsize=14)
                ax1.tick_params(axis='y', labelsize=14)
                ax1.legend(fontsize=14)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/safety_metrics_{model_name}.pdf", 
                           format='pdf', bbox_inches='tight')
                plt.close()
                
                logging.info(f"Generated language plot for {model_name}")

def analyze_experiment_dir(experiment_dir: str):
    """Analyze a specific experiment directory"""
    logging.info(f"Starting analysis of: {experiment_dir}")
    analyzer = ExperimentAnalyzer(experiment_dir)
    stats = analyzer.analyze_experiment()
    analyzer.generate_plots()
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('experiment_dir', help='Directory containing experiment results')
    
    args = parser.parse_args()
    analyze_experiment_dir(args.experiment_dir)