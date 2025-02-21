# Better Evaluation on Multilingual Jailbreak

This repository evaluates safe behavior across multiple languages.

## Repository Structure

- `codes/`: Core implementation containing all necessary code for generating responses, computing statistics, and annotations
- `results/`: 
    - `annotation/`: Contains analysis plots and inter-annotator agreement metrics
    - `generation/`: Contains safety metric visualizations 
    - `experiments/`: Sample runs across different models

## Models Evaluated

Tested across 10 languages (ar, bn, en, it, jv, ko, sw, th, vi, zh):
- aya-101
- aya-expanse  
- llama-2
- llama-3
- qwen
- seallm

## Dataset

To obtain MultiJail dataset, please refer to the original paper of https://arxiv.org/pdf/2310.06474