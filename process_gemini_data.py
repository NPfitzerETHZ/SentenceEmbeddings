
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer

device = 'cpu'
json_data_file = "gemini_patch_dataset_grid.json"

def collect_train_test_data_gaussian(llm, json_path, train_ratio, test_ratio):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Shuffle the data for randomized splitting
    np.random.shuffle(data)

    data_length = len(data) - 1
    train_size = int(train_ratio * data_length)
    test_size = int(test_ratio*data_length)

    # Split into train and test
    train_data = data[:train_size]
    test_data = data[train_size : train_size + test_size]

    # Helper function to convert raw data to dict
    def process_dataset(dataset):
        responses = [entry["gemini_response"] for entry in dataset]
        embeddings = llm.encode(responses, device=device)
        goals = [[*entry["normalized_start"], *entry["normalized_std"]] for entry in dataset]

        task_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        goal_tensor = torch.tensor(goals, dtype=torch.float32, device=device)

        return {
            "task_embedding": task_embeddings,
            "goal": goal_tensor
        }

    train_dict = process_dataset(train_data)
    test_dict = process_dataset(test_data)

    return train_dict, test_dict

def collect_train_test_data_grid(llm, json_path, train_ratio, test_ratio):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Shuffle the data for randomized splitting
    np.random.shuffle(data)

    data_length = len(data) - 1
    train_size = int(train_ratio * data_length)
    test_size = int(test_ratio*data_length)

    # Split into train and test
    train_data = data[:train_size]
    test_data = data[train_size : train_size + test_size]

    # Helper function to convert raw data to dict
    def process_dataset(dataset):
        responses = [entry["gemini_response"] for entry in dataset]
        embeddings = llm.encode(responses, device=device)
        goals = [[*entry["grid"]] for entry in dataset]

        task_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        goal_tensor = torch.tensor(goals, dtype=torch.float32, device=device)

        return {
            "task_embedding": task_embeddings,
            "goal": goal_tensor
        }

    train_dict = process_dataset(train_data)
    test_dict = process_dataset(test_data)

    return train_dict, test_dict

def collect_train_test_data_grid_texts(json_path, train_ratio, test_ratio, device):
    # Load JSON data
    with open(json_path, 'r') as f:
            data = [json.loads(line) for line in f] 

    # Shuffle the data for randomized splitting
    np.random.shuffle(data)

    data_length = len(data) - 1
    train_size = int(train_ratio * data_length)
    test_size = int(test_ratio*data_length)

    # Split into train and test
    train_data = data[:train_size]
    test_data = data[train_size : train_size + test_size]
    
    def process_dataset(dataset):
        responses = [entry["gemini_response"] for entry in dataset]
        goals = [[*entry["grid"]] for entry in dataset]
        goal_tensor = torch.tensor(goals, dtype=torch.float32, device=device)

        return {
            "task_text": responses,
            "goal": goal_tensor
        }

    train_dict = process_dataset(train_data)
    test_dict = process_dataset(test_data)

    return train_dict, test_dict

def collect_train_test_data_from_embeddings(json_path, train_ratio, test_ratio, device='cpu'):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]  # Handles newline-delimited JSON

    # Shuffle the data for randomized splitting
    np.random.shuffle(data)

    data_length = len(data)
    train_size = int(train_ratio * data_length)
    test_size = int(test_ratio * data_length)

    # Split into train and test
    train_data = data[:train_size]
    test_data = data[train_size : train_size + test_size]

    # Helper function to convert raw data to tensors
    def process_dataset(dataset):
        embeddings = [entry["embedding"] for entry in dataset]
        goals = [[*entry["grid"]] for entry in dataset]

        task_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        goal_tensor = torch.tensor(goals, dtype=torch.float32, device=device)

        return {
            "task_embedding": task_embeddings,
            "goal": goal_tensor
        }

    train_dict = process_dataset(train_data)
    test_dict = process_dataset(test_data)

    return train_dict, test_dict



