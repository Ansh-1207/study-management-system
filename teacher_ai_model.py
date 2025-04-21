import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(file_path, file_type="json"):
    if file_type == "json":
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"Data successfully loaded from JSON file: {file_path}")
            return pd.DataFrame(data)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            return None
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def preprocess_data(data):
    if data is None:
        return None, None, None, None
    assert data is not None, "Error: Input data to preprocess_data cannot be None."
    try:
        concept_performance_df = pd.json_normalize(data['concept_performance'])
        y = concept_performance_df.fillna(0).values

        prior_knowledge_df = pd.json_normalize(data['prior_knowledge']) # Extract prior_knowledge FIRST

        data = data.drop(['student_id', 'concept_performance', 'prior_knowledge', 'learning_style_preference'], axis=1, errors='ignore')
        X_df = pd.concat([data.reset_index(drop=True), prior_knowledge_df], axis=1) # Keep as DataFrame
        X_df = X_df.fillna(X_df.mean())
        feature_names = X_df.columns.tolist() # Extract column names while it's a DataFrame
        X = X_df.values # Convert to NumPy array after extracting feature names
        assert X.shape[1] > 0, "Error: No features available after preprocessing."
        assert y.shape[0] > 0 and y.shape[1] > 0, "Error: No target values available after preprocessing."

        print("Data preprocessed to predict performance across all concepts.")
        return X, y, feature_names, concept_performance_df.columns.tolist() # Return NumPy array for X
    except KeyError as e:
        raise KeyError(f"Error: Could not find necessary columns in the data: {e}")
    except Exception as e:
        raise Exception(f"An error occurred during preprocessing: {e}")

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    try:
        train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=random_state)
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        print("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test, train_idx, test_idx
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return None, None, None, None, None, None

def select_features(X_train, y_train, X_test, new_student_data, feature_names, num_features_to_select=20):
    try:
        # Aggregate the target variables (concept performance) - using the mean here for feature selection
        y_train_aggregated = np.mean(y_train, axis=1)

        selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
        X_train_selected = selector.fit_transform(X_train, y_train_aggregated)
        X_test_selected = selector.transform(X_test)
        new_student_data_selected = selector.transform(new_student_data)
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        print(f"Top {num_features_to_select} features selected based on aggregated target.")
        print(f"Selected feature indices: {selected_indices}")
        print(f"Selected feature names: {selected_feature_names}")
        return X_train_selected, X_test_selected, new_student_data_selected, selected_feature_names, selector # Return the selector
    except Exception as e:
        print(f"Error during feature selection: {e}")
        return None, None, None, None, None

def preprocess_new_student(new_student_raw, feature_names):
    print("--- Inside preprocess_new_student ---")
    print("Columns in new_student_raw:", new_student_raw.columns)
    print("Data in new_student_raw:")
    print(new_student_raw)
    if 'prior_knowledge' in new_student_raw.columns:
        try:
            prior_knowledge_dict = new_student_raw['prior_knowledge'].iloc[0]
            prior_knowledge_df = pd.json_normalize([prior_knowledge_dict])
            new_student_df = new_student_raw.drop(['student_id', 'concept_performance', 'prior_knowledge', 'learning_style_preference'], axis=1, errors='ignore').reset_index(drop=True)
            new_student_data_df = pd.concat([new_student_df, prior_knowledge_df], axis=1)
            new_student_processed = new_student_data_df[feature_names].fillna(new_student_data_df[feature_names].mean()).values.reshape(1, -1) # Explicit reshape to (1, num_features)
            return new_student_processed
        except Exception as e:
            print(f"Error during new student preprocessing: {e}")
            return None
    else:
        print("Error: 'prior_knowledge' column not found in new_student_raw.")
        return None

def create_concept_vocabulary(data):
    concept_vocabulary = set()
    for index, row in data.iterrows():
        prior_knowledge = row.get('prior_knowledge', {})
        if isinstance(prior_knowledge, dict):
            concept_vocabulary.update(prior_knowledge.keys())
    concept_list = sorted(list(concept_vocabulary))
    concept_to_index = {concept: index for index, concept in enumerate(concept_list)}
    index_to_concept = {index: concept for concept, index in concept_to_index.items()}
    return concept_to_index, index_to_concept

def get_concept_embeddings_bert(data, tokenizer, model):
    concept_to_embedding = {}
    concept_to_index, index_to_concept = create_concept_vocabulary(data)
    for concept in concept_to_index:
        inputs = tokenizer(concept, return_tensors="pt", truncation=True, max_length=128).to(device) # Move input to GPU
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the [CLS] token embedding as the representation of the concept
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0) # Keep it on the GPU as a PyTorch tensor
        concept_to_embedding[concept] = cls_embedding
    return concept_to_embedding

class TransformerRegressionModel(nn.Module):
    def __init__(self, bert_model_name, num_numerical_features, concept_embedding_dim, performance_embedding_dim, num_transformer_layers, num_attention_heads, transformer_hidden_dim, output_dim):
        super(TransformerRegressionModel, self).__init__()
        self.bert_model_name = bert_model_name
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_embedding_dim = self.bert_model.config.hidden_size
        self.fc_concept = nn.Linear(self.bert_embedding_dim, 128) # Process aggregated concept embedding
        self.numerical_fc = nn.Linear(num_numerical_features, 64)
        self.relu = nn.ReLU()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=128 + 64, nhead=num_attention_heads, dim_feedforward=transformer_hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_transformer_layers, enable_nested_tensor=False)
        self.fc_out = nn.Linear(128 + 64, output_dim) # Output from combined features

    def forward(self, aggregated_concept_embedding, numerical_features):
        # Process aggregated concept embedding
        processed_concept = self.relu(self.fc_concept(aggregated_concept_embedding))

        # Process numerical features
        numerical_output = self.relu(self.numerical_fc(numerical_features))

        # Combine processed concept and numerical features
        combined_features = torch.cat((processed_concept, numerical_output), dim=-1).unsqueeze(1) # Add sequence length of 1

        # Pass through Transformer
        transformer_output = self.transformer_encoder(combined_features)
        transformer_output = transformer_output.squeeze(1)

        # Output layer
        output = self.fc_out(transformer_output) # Shape: (batch_size, output_dim)

        return output

class BertRegressionDataset(Dataset):
    def __init__(self, X_numerical, y, original_data, concept_to_embedding, numerical_feature_names, tokenizer, bert_model):
        self.X_numerical = X_numerical
        self.y = y
        self.original_data = original_data
        self.concept_to_embedding = concept_to_embedding
        self.numerical_feature_names = numerical_feature_names
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def __len__(self):
        return len(self.X_numerical)

    def __getitem__(self, idx):
        numerical_features = torch.tensor(self.X_numerical[idx], dtype=torch.float32).to(device)

        prior_knowledge_dict = self.original_data.iloc[idx]['prior_knowledge']
        if isinstance(prior_knowledge_dict, str):
            prior_knowledge_dict = json.loads(prior_knowledge_dict.replace("'", "\""))

        concept_names = list(prior_knowledge_dict.keys())
        concept_performances = torch.tensor(list(prior_knowledge_dict.values()), dtype=torch.float32).to(device)

        # Get BERT embeddings for concept names
        inputs = self.tokenizer(concept_names, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            bert_outputs = self.bert_model(**inputs)
        concept_embeddings = bert_outputs.last_hidden_state[:, 0, :]

        # Aggregate concept embeddings (mean pooling)
        if concept_embeddings.numel() > 0:
            aggregated_concept_embedding = torch.mean(concept_embeddings, dim=0)
        else:
            aggregated_concept_embedding = torch.zeros(self.bert_model.config.hidden_size, device=device)

        targets = torch.tensor(self.y[idx], dtype=torch.float32).to(device) # No unsqueeze here

        return aggregated_concept_embedding, numerical_features, targets

def train_pytorch_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (aggregated_concept_embedding, numerical_features, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(aggregated_concept_embedding, numerical_features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def evaluate_pytorch_model(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (aggregated_concept_embedding, numerical_features, targets) in enumerate(dataloader):
            outputs = model(aggregated_concept_embedding, numerical_features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Evaluation Loss: {avg_loss:.4f}')
    return avg_loss

def predict_for_new_student_pytorch(model, new_student_data):
    # Adapt for the new model input
    raise NotImplementedError("predict_for_new_student_pytorch needs adaptation")

def tune_hyperparameters_pytorch(X_train, y_train, input_dim, output_dim, param_grid):
    # Adapt for the new model and its hyperparameters
    raise NotImplementedError("tune_hyperparameters_pytorch needs adaptation")

if __name__ == "__main__":
    # 1. Load data
    file_path = 'synthetic_student_performance_dataset_100K.json'
    file_type = "json"
    data = load_data(file_path, file_type)
    if data is None:
        exit()
    new_student_raw = data.iloc[[0]].copy()

    try:
        # 2. Preprocess data for all concepts
        X, y, feature_names, concept_names = preprocess_data(data.copy())
        if X is None or y is None:
            exit()

        # 3. Split data
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_data(X, y)
        if X_train is None:
            exit()

        # 4. Standardize Target Variables
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        # 5. Apply PCA to Target Variables
        n_components_y = min(y_train_scaled.shape[1], 50)
        pca_y = PCA(n_components=n_components_y)
        y_train_pca = pca_y.fit_transform(y_train_scaled)
        y_test_pca = pca_y.transform(y_test_scaled)
        output_dim = n_components_y

        # 6. Select features based on aggregated target
        new_student_processed = preprocess_new_student(new_student_raw, feature_names)
        if new_student_processed is None:
            exit()
        X_train_selected, X_test_selected, new_student_data_selected, selected_feature_names, selector = select_features(
            X_train, y_train, X_test, new_student_processed, feature_names
        )
        if X_train_selected is None or new_student_data_selected is None or selector is None:
            exit()
        input_dim_numerical = X_train_selected.shape[1]

        # 7. Initialize tokenizer and BERT model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

        # 8. Initialize the TransformerRegressionModel (adjust input dimensions)
        embedding_dim = 128
        num_transformer_layers = 1 # Reduced layers for simpler aggregation
        num_attention_heads = 8
        transformer_hidden_dim = 256
        model = TransformerRegressionModel(
            'bert-base-uncased',
            input_dim_numerical,
            768,
            embedding_dim,
            num_transformer_layers,
            num_attention_heads,
            transformer_hidden_dim,
            output_dim
        ).to(device)

        # 9. Create a BertRegressionDataset
        train_dataset = BertRegressionDataset(X_train_selected, y_train_pca, data.iloc[train_idx].copy(), {}, selected_feature_names, tokenizer, bert_model)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        test_dataset = BertRegressionDataset(X_test_selected, y_test_pca, data.iloc[test_idx].copy(), {}, selected_feature_names, tokenizer, bert_model)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # 10. Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 11. Train the new model
        num_epochs = 10
        train_pytorch_model(model, train_dataloader, criterion, optimizer, num_epochs)

        # 12. Evaluate the new model
        evaluate_pytorch_model(model, test_dataloader)

        # 13. Predict for a new student (adaptation needed)
        # new_student_processed_for_prediction = preprocess_new_student(new_student_raw, selected_feature_names)
        # if new_student_processed_for_prediction is not None:
        #     # Need to adapt how new student data is fed into the Transformer model
        #     pass

    except Exception as e:
        print(f"An error occurred in the main block: {e}")