import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.dirname(file_dir)))
from solid_step_helper import (
    solid_step_add_node_to_graph, 
    solid_step_remove_node_from_graph,
    solid_step_counting_query,
    solid_step_list_child_nodes,
    solid_step_update_node_value,
    solid_step_rank_child_nodes
)
import networkx as nx
import jsonlines
import json
import re
import copy
import inspect
from collections import defaultdict


def strip_comments(code):
    """
    Remove comments from the code.
    """
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    
    # Remove multi-line comments
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    return code


def extract_finetune_data(input_path):
    """
    Formats the data in the JSONL file to the format expected for fine-tuning Qwen. Also returns info on number
    of queries per sub-category (type of action). Here's an example of the expected output:

    [
        {
            "id": "identity_0",
            "conversations": [
                {
                    "from": "user",
                    "value": "Remove ju1.s4.s2c1 from the graph. Rank direct child nodes of ju1.s4 in the updated graph based on physical_capacity_bps attribute. Return a list of tuple, each tuple has node name and its total physical capacity."
                },
                {
                    "from": "assistant",
                    "value": "def process_graph(graph_data):\n            # Create a copy of the graph before modifications\n            graph_copy = copy.deepcopy(graph_data)\n            \n            # Remove the specified node\n            node_to_remove = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4.s2c1':\n                    node_to_remove = node[0]\n                    break\n            \n            if node_to_remove:\n                graph_copy.remove_node(node_to_remove)\n            \n            # Find direct child nodes of ju1.s4\n            parent_node = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4':\n                    parent_node = node[0]\n                    break\n            \n            child_nodes = []\n            if parent_node:\n                for successor in graph_copy.successors(parent_node):\n                    child_nodes.append(successor)\n            \n            # Calculate total physical capacity for each child node\n            def calculate_total_capacity(node):\n                total_capacity = 0\n                for successor in nx.descendants(graph_copy, node):\n                    if 'EK_PORT' in graph_copy.nodes[successor]['type']:\n                        total_capacity += graph_copy.nodes[successor].get('physical_capacity_bps', 0)\n                return total_capacity\n            \n            child_capacities = []\n            for child in child_nodes:\n                child_name = graph_copy.nodes[child]['name']\n                total_capacity = calculate_total_capacity(child)\n                child_capacities.append((child_name, total_capacity))\n            \n            # Sort the child nodes based on total physical capacity\n            child_capacities.sort(key=lambda x: x[1], reverse=True)\n            \n            # Return the result as a list of tuples\n            return_object = {\n                'type': 'list',\n                'data': child_capacities\n            }\n            \n            return return_object"
                }
            ]
        },
    ]
    """
    # Dictionary to track items per sub-category
    subcategory_items = defaultdict(list)
    subcategory_counts = defaultdict(int)
    all_results = []
    global_count = 0
    
    # First pass: Read the JSONL file and identify all sub-categories
    with jsonlines.open(input_path) as reader:
        for item in reader:
            messages = item["messages"]

            question = None
            answer = None
            task_label = None

            for message in messages:
                if "question" in message:
                    question = message.get("question", "")
                elif "answer" in message:
                    answer = message.get("answer", "")
                elif "task_label" in message:
                    task_label = message.get("task_label", "")
            
            # Extract the sub-category (everything after "capacity planning, level-?")
            parts = task_label.split(", ")
            if len(parts) > 2:
                subcategory = parts[2]  # Get the third part of the label
                    
                # Create the conversation format
                conversation_item = {
                    "id": f"identity_{global_count}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": question
                        },
                        {
                            "from": "assistant",
                            "value": answer
                        }
                    ]
                }
                
                # Add to the subcategory items
                subcategory_items[subcategory].append(conversation_item)
                subcategory_counts[subcategory] += 1
                global_count += 1
    
    # Combine all subcategory items into one list
    for subcategory, items in subcategory_items.items():
        all_results.extend(items)
    
    return all_results, dict(subcategory_counts)


def prepend_function_definitions(answer_code):
    """
    Include all solid_step_* function definitions before the answer code.
    """

    # List of all the possible atomic actions.
    SOLID_STEP_FUNCTIONS = [
        solid_step_add_node_to_graph, 
        solid_step_remove_node_from_graph,
        solid_step_counting_query,
        solid_step_list_child_nodes,
        solid_step_update_node_value,
        solid_step_rank_child_nodes
    ]

    # Mapping function names to their source code implementations.
    function_definitions = {
        rf'{func.__name__}\((.*?)\)': inspect.getsource(func) for func in SOLID_STEP_FUNCTIONS
    }
    
    modified_code = answer_code
    func_defs_to_add = []
    # Whenever an atomic function is called, we need to include its function definition.
    for pattern, definition in function_definitions.items():
        # Find all occurrences of the function call
        match = re.search(pattern, modified_code)
        
        # Process each match from end to start to avoid index issues
        if match is not None:
            func_defs_to_add.append(definition)
    
    # Concatenate the function definitions and the original code.
    func_defs_to_add.append(modified_code)
    modified_code = "".join(func_defs_to_add)
    
    return modified_code


def process_example(example):
    """Process a single example to convert solid_step functions"""
    # Check if example is in the messages format
    if "messages" in example:
        messages = example["messages"]
        # Find question and answer in messages
        question = None
        answer = None
        task_label = None
        
        for message in messages:
            if "question" in message:
                question = message["question"]
            elif "answer" in message:
                answer = message["answer"]
            elif "task_label" in message:
                task_label = message["task_label"]
        
        if not answer or 'ground_truth_process_graph' not in answer:
            return example
        
        # Extract the function body
        match = re.search(r'ground_truth_process_graph\(.*?\):(.*?)(?=return return_object|$)', answer, re.DOTALL)
        
        # Skip if no answer function found.
        if not match:
            return example
        
        
        # Include the updated graph key in the answer (similar to evaluation format).
        pattern = r"return_object = {'type': ('.*'), 'data': (\S+)}"
        ret_match = re.search(pattern, answer)
        new_answer = re.sub(pattern, f"return_object = {{'type': {ret_match.group(1)}, 'data': {ret_match.group(2)}, \
                            'updated_graph': nx.readwrite.json_graph.node_link_data(graph_data)}}", answer)

        # Change answer to match the expected format of the MALT evaluation.
        new_answer = new_answer.replace("def ground_truth_process_graph", "def process_graph")

        # Add necessary function definitions to the answer.
        new_answer = strip_comments(prepend_function_definitions(answer))
        
        # Create a new example with the converted answer
        new_example = copy.deepcopy(example)
        new_example["messages"] = []
        
        if question:
            new_example["messages"].append({"question": question})
        if new_answer:
            new_example["messages"].append({"answer": new_answer})
        if task_label:
            new_example["messages"].append({"task_label": task_label})
        
        return new_example
    else:
        # For direct question/answer format
        question = example.get('question', '')
        answer = example.get('answer', '')
        
        # Skip if no answer or not ground_truth_process_graph found
        if not answer or 'ground_truth_process_graph' not in answer:
            return example
        
        # Extract the function body
        match = re.search(r'ground_truth_process_graph\(.*?\):(.*?)(?=return return_object|$)', answer, re.DOTALL)
        
        # Skip if no answer function found.
        if not match:
            return example
        
        # Add necessary function definitions to the answer.
        new_answer = strip_comments(prepend_function_definitions(answer))
        
        # Create a new example with the converted answer
        new_example = copy.deepcopy(example)
        new_example['answer'] = new_answer
        
        return new_example


def process_jsonlines_file(input_file, output_file):
    """Process a JSONL file containing examples"""
    with jsonlines.open(input_file, 'r') as f_in, jsonlines.open(output_file, 'w') as f_out:
        for example in f_in:
            processed_example = process_example(example)
            f_out.write(processed_example)


def process_json_file(input_file, output_file):
    """Process a JSON file containing examples"""
    with open(input_file, 'r') as f_in:
        data = json.load(f_in)
    
    if isinstance(data, list):
        processed_data = [process_example(example) for example in data]
    else:
        processed_data = process_example(data)
    
    with open(output_file, 'w') as f_out:
        json.dump(processed_data, f_out, indent=2)


if __name__ == "__main__":    
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Preprocess query answers to also include function definitions for solid_step functions.
    preprocessed_path = output_file.replace('.json', '_preproc.json')
    
    # Determine file type by extension
    if input_file.endswith('.jsonl'):
        process_jsonlines_file(input_file, preprocessed_path)
    else:
        process_json_file(input_file, preprocessed_path)
    
    print(f"Preprocessed examples with solid step (atomic actions) function definitions saved to {preprocessed_path}")

    # Convert preprocessed queries to format expected for fine tuning Qwen.
    finetune_data, subcategory_stats = extract_finetune_data(preprocessed_path)

    # Write the output to a file
    with open(output_file, "w") as f:
        json.dump(finetune_data, f, indent=2)

    # Print statistics
    print(f"Extracted {len(finetune_data)} total items for fine-tuning to {preprocessed_path}")
    print("Items per subcategory:")
    for subcategory, count in subcategory_stats.items():
        print(f"  - {subcategory}: {count} items")

    print(f"Final processed examples in Qwen finetune format saved to {output_file}")