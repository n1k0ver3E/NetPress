import numpy as np
import pandas as pd
import jsonlines
import random
import os
import time
import matplotlib.pyplot as plt
import json

import argparse
from scipy import stats
import math

# anexample of how to use main.py with input args
# Example usage:
# python eval_with_sem_err.py --sampling_method random

plt.rcParams['font.size'] = '20'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
def main():
    parser = argparse.ArgumentParser(description='Evaluate semantic error detection performance')
    parser.add_argument('--sampling_method', type=str, choices=['first', 'random'], default='first',
                        help='Method to sample: "first" takes first N samples, "random" takes random N samples')
    args = parser.parse_args()

    input_path = ["logs/AzureGPT4Agent_few_shot_semantic/new_gpt4o_few_shot_semantic.jsonl", 
                  "logs/AzureGPT4Agent_few_shot_semantic/extra_50_gpt4o_few_shot_semantic.jsonl",
                  "logs/AzureGPT4Agent_cot/new_gpt4o_cot.jsonl",
                  "logs/AzureGPT4Agent_cot/extra_50_gpt4o_cot.jsonl",
                  "logs/Qwen2.5-72B-Instruct_cot/new_qwen_cot_50.jsonl",
                  "logs/Qwen2.5-72B-Instruct_cot/extra_50_qwen_cot.jsonl",
                  "logs/Qwen2.5-72B-Instruct_few_shot_semantic/new_qwen_few_shot_semantic_50.jsonl",
                  "logs/Qwen2.5-72B-Instruct_few_shot_semantic/extra_50_qwen_few_shot_semantic.jsonl"]
    
    # Define which files to merge
    cot_merge_files = [
        "logs/AzureGPT4Agent_cot/new_gpt4o_cot.jsonl",
        "logs/AzureGPT4Agent_cot/extra_50_gpt4o_cot.jsonl"
    ]
    cot_merge_name = "cot_GPT"

    few_shot_merge_files = [
        "logs/AzureGPT4Agent_few_shot_semantic/new_gpt4o_few_shot_semantic.jsonl",
        "logs/AzureGPT4Agent_few_shot_semantic/extra_50_gpt4o_few_shot_semantic.jsonl"
    ]
    few_shot_merge_name = "few_shot_GPT"
    
    # Add Qwen merge file definitions
    qwen_cot_merge_files = [
        "logs/Qwen2.5-72B-Instruct_cot/new_qwen_cot_50.jsonl",
        "logs/Qwen2.5-72B-Instruct_cot/extra_50_qwen_cot.jsonl"
    ]
    qwen_cot_merge_name = "cot_Qwen"
    
    qwen_few_shot_merge_files = [
        "logs/Qwen2.5-72B-Instruct_few_shot_semantic/new_qwen_few_shot_semantic_50.jsonl",
        "logs/Qwen2.5-72B-Instruct_few_shot_semantic/extra_50_qwen_few_shot_semantic.jsonl"
    ]
    qwen_few_shot_merge_name = "few_shot_Qwen"
    
    # Load data from each file
    file_results = {}
    for path in input_path:
        print(f"Processing file: {path}")
        results = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                results.append(obj)
        file_results[path] = results
    
    # Process results, merging specified files
    all_results = []
    processed_files = set()
    
    # Handle the CoT merged files
    if set(cot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in cot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(cot_merge_files),
            'name': cot_merge_name,
            'grouped_results': grouped_results
        })

    # Handle the Few Shot merged files
    if set(few_shot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in few_shot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(few_shot_merge_files),
            'name': few_shot_merge_name,
            'grouped_results': grouped_results
        })
        
    # Handle the Qwen CoT merged files
    if set(qwen_cot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in qwen_cot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(qwen_cot_merge_files),
            'name': qwen_cot_merge_name,
            'grouped_results': grouped_results
        })

    # Handle the Qwen Few Shot merged files
    if set(qwen_few_shot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in qwen_few_shot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(qwen_few_shot_merge_files),
            'name': qwen_few_shot_merge_name,
            'grouped_results': grouped_results
        })
    
    # Then process any remaining files
    for path in input_path:
        if path not in processed_files:
            results = file_results[path]
            
            # Group the results by task label
            grouped_results = {}
            for result in results:
                task_label = result["Label"]
                if task_label not in grouped_results:
                    grouped_results[task_label] = []
                grouped_results[task_label].append(result)
            
            all_results.append({
                'path': path,
                'name': os.path.basename(path).split('.')[0],
                'grouped_results': grouped_results
            })

    sample_sizes = [5, 100]
    
    for sample_size in sample_sizes:
        print(f"\n=== STATISTICS FOR {args.sampling_method.upper()} {sample_size} SAMPLES PER LABEL ===")
        
        safety_data = []
        correctness_data = []
        
        for result_set in all_results:
            grouped_results = result_set['grouped_results']
            name = result_set['name']
            
            # Calculate stats for each task label
            task_labels = list(grouped_results.keys())
            
            # Correctness stats
            print(f"\nCorrectness Pass Rates for {name}:")
            
            # Calculate overall statistics for correctness
            all_binary_outcomes = []
            for task_label in task_labels[1:]:
                # Sample based on the specified method
                if args.sampling_method == 'first':
                    samples = grouped_results[task_label][:sample_size]
                else:  # random sampling
                    all_samples = grouped_results[task_label]
                    # If we have enough samples, take a random sample, otherwise take all
                    if len(all_samples) > sample_size:
                        samples = random.sample(all_samples, sample_size)
                    else:
                        samples = all_samples
                        
                binary_outcomes = [1 if result["Result-Correctness"] == "Pass" else 0 for result in samples]
                all_binary_outcomes.extend(binary_outcomes)
            
            # Calculate overall pass rate and error margin for correctness
            total_samples = len(all_binary_outcomes)
            correctness_pass_rate = (sum(all_binary_outcomes) / total_samples) * 100
            correctness_sem = stats.sem(all_binary_outcomes, ddof=0) * 100
            correctness_error_margin = 1.96 * correctness_sem
            
            print(f"Overall Correctness: {correctness_pass_rate:.2f}% ±{correctness_error_margin:.2f}% (n={total_samples})")

            # Safety stats
            print(f"\nSafety Pass Rates for {name}:")
            
            # Calculate overall safety statistics
            all_binary_outcomes = []
            for task_label in task_labels[1:]:
                # Sample based on the specified method
                if args.sampling_method == 'first':
                    samples = grouped_results[task_label][:sample_size]
                else:  # random sampling
                    all_samples = grouped_results[task_label]
                    # If we have enough samples, take a random sample, otherwise take all
                    if len(all_samples) > sample_size:
                        samples = random.sample(all_samples, sample_size)
                    else:
                        samples = all_samples
                        
                binary_outcomes = [1 if result["Result-Safety"] == "Pass" else 0 for result in samples]
                all_binary_outcomes.extend(binary_outcomes)
            
            total_samples = len(all_binary_outcomes)
            safety_pass_rate = (sum(all_binary_outcomes) / total_samples) * 100
            safety_sem = stats.sem(all_binary_outcomes, ddof=0) * 100
            safety_error_margin = 1.96 * safety_sem
            
            print(f"Overall Safety: {safety_pass_rate:.2f}% ±{safety_error_margin:.2f}% (n={total_samples})")
            
            # Store data for plotting
            safety_data.append({
                'name': name,
                'pass_rate': safety_pass_rate,
                'error_margin': safety_error_margin
            })
            
            correctness_data.append({
                'name': name,
                'pass_rate': correctness_pass_rate,
                'error_margin': correctness_error_margin
            })
        
        # Create a new figure for each sample size with professional styling
        plt.figure(figsize=(7, 6), dpi=300)
        
        # Professional color palette - Scientific color scheme
        colors = ['#0073C2', '#EFC000', '#868686', '#CD534C', '#7AA6DC', '#003C67']

        # Plot the scatter points with error bars
        for j in range(len(safety_data)):
            s_data = safety_data[j]
            c_data = correctness_data[j]
            
            plt.errorbar(
                s_data['pass_rate'] / 100,  # Convert to decimal
                c_data['pass_rate'] / 100,  # Convert to decimal
                xerr=s_data['error_margin'] / 100,  # Convert to decimal
                yerr=c_data['error_margin'] / 100,  # Convert to decimal
                fmt='o',
                color=colors[j % len(colors)],
                markersize=8,
                markeredgewidth=1.5,
                markeredgecolor='white',
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                label=s_data['name']
            )

        # Customize grid
        plt.grid(True, linestyle='--', alpha=0.3, which='major')
        plt.gca().set_axisbelow(True)  # Place grid behind points

        # Set labels
        plt.xlabel('Safety Rate')
        plt.ylabel('Success Rate')
        # plt.title(f'N = {sample_size}')  # Comment out title as in file_utils.py

        # Set axis ranges with padding
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)

        # Add legend with improved styling
        legend = plt.legend(loc='lower right',
                          frameon=True,
                          fancybox=False,
                          edgecolor='black')

        plt.tight_layout()
        # Save individual figure
        plt.savefig(f'figs/malt_sampling_{args.sampling_method}_N{sample_size}.png', 
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.2)
        plt.close()

    print(f"Individual figures saved as figs/malt_sampling_{args.sampling_method}_N[sample_size].png")

# run the main function
if __name__ == "__main__":
    main()