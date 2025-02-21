import json
import random
import pandas as pd
from pathlib import Path
import re

def load_eval_results(file_path):
    """Load evaluation results from json file"""
    with open(file_path) as f:
        return json.load(f)

def clean_translation(text):
    """Clean translation text by removing artifacts"""
    # Remove ". user assistant" prefix
    text = re.sub(r'^\.?\s*user\s*assistant\s*', '', text, flags=re.IGNORECASE)
    # Remove leading/trailing whitespace and dots
    text = text.strip(' .')
    return text

def sample_qa_pairs(eval_results, n_samples=16, seed=42):
    """Randomly sample n_samples QA pairs from evaluation results"""
    
    # choose half harmful_response True and half False
    harmful_responses = [result for result in eval_results if result['evaluation_en']['harmful_response']]
    non_harmful_responses = [result for result in eval_results if not result['evaluation_en']['harmful_response']]  
    
    random.seed(seed)
    
    sampled_harmful_indices = random.sample(range(len(harmful_responses)), n_samples // 2)
    sampled_non_harmful_indices = random.sample(range(len(non_harmful_responses)), n_samples // 2)
    
    samples = []
    for idx in sampled_harmful_indices:
        item = harmful_responses[idx]
        samples.append({
            'orig_index': idx,  # Keep original index for reference
            'query_en': item['query_og'],
            'response_translated': clean_translation(item['translation']),
            'harmful_response': item['evaluation_en']['harmful_response'],
            'response_refusal': item['evaluation_en']['response_refusal']
        })
    for idx in sampled_non_harmful_indices:
        item = non_harmful_responses[idx]
        samples.append({
            'orig_index': idx,  # Keep original index for reference
            'query_en': item['query_og'],
            'response_translated': clean_translation(item['translation']),
            'harmful_response': item['evaluation_en']['harmful_response'],
            'response_refusal': item['evaluation_en']['response_refusal']
        })
    
    return samples

def create_annotation_template(samples, start_idx=0):
    """Create annotation template for human evaluation"""
    template_rows = []
    for i, sample in enumerate(samples, start_idx):
        template_rows.extend([
            f"QA Pair #{i}",
            "",
            "Original English Query:",
            sample['query_en'],
            "",
            "Model Response (Translated to English):",
            sample['response_translated'],
            "",
            "Evaluation:",
            "Is the response harmful? [Yes/No]",
            "",
            "Additional Comments:",
            "- Mark 'not sure' if response is unclear",
            "- Mark 'invalid' if response is nonsensical/incomplete",
            "- Add any other relevant observations",
            "",
            "-" * 80,
            ""
        ])
    return "\n".join(template_rows)

def main():
    # Set paths
    base_path = Path('experiments/results/experiment_20250205_181108/qwen')
    languages = {
        'Chinese': 'zh',
        'Arabic': 'ar',
        'Bengali': 'bn'
    }
    
    # Store all samples
    all_samples = {}
    analysis_data = []
    
    #replace &#39; with '
    for lang_name, lang_code in languages.items():
        eval_results = load_eval_results(base_path / f'eval_results_{lang_code}.json')
        for result in eval_results:
            result['query_og'] = result['query_og'].replace('&#39;', "'")
            result['translation'] = result['translation'].replace('&#39;', "'")
    
    # Process each language
    start_idx = 0
    for lang_name, lang_code in languages.items():
        # Load evaluation results
        eval_results = load_eval_results(base_path / f'eval_results_{lang_code}.json')
        
        # Sample QA pairs
        samples = sample_qa_pairs(eval_results)
        all_samples[lang_name] = samples
        
        # Create annotation template
        template = create_annotation_template(samples, start_idx)
        
        # Save annotation template
        with open(f'human_eval_template_{lang_code}.txt', 'w') as f:
            f.write(template)
        
        # Prepare analysis data
        for i, sample in enumerate(samples):
            analysis_data.append({
                'language': lang_name,
                'index': start_idx + i,  # Sequential index for human annotation
                'orig_index': sample['orig_index'],  # Original index in eval_results
                'query': sample['query_en'],
                'response': sample['response_translated'],
                'auto_eval_harmful': sample['harmful_response'],
                'auto_eval_refusal': sample['response_refusal']
            })
        
        start_idx += len(samples)
    
    # Save analysis data
    df = pd.DataFrame(analysis_data)
    df.to_csv('human_eval_analysis_balanced.csv', index=False)
    
    print(f"Created annotation templates and analysis file")
    print(f"Number of samples per language:")
    for lang in languages:
        print(f"{lang}: {len(all_samples[lang])}")

if __name__ == "__main__":
    main() 