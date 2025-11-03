"""
Module 3b: Hybrid Temporal Network Learning - LSTM Component
Embed-LSTM model for complex pattern prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging
import config


class TimeSeriesDataset(Dataset):
    """Time series dataset for LSTM"""
    def __init__(self, data, cluster_ids, look_back=10):
        self.data = data
        self.cluster_ids = cluster_ids
        self.look_back = look_back

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, index):
        x = self.data[index:index+self.look_back]
        y = self.data[index+self.look_back]
        cluster_id = self.cluster_ids[index+self.look_back]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y]), torch.LongTensor([cluster_id])


class LSTMModel(nn.Module):
    """LSTM model with cluster embeddings"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_clusters):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_clusters = num_clusters
        
        self.embedding = nn.Embedding(num_clusters, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, cluster_id):
        cluster_embedding = self.embedding(cluster_id)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        lstm_out = out[:, -1, :]
        
        combined = torch.cat((lstm_out, cluster_embedding), dim=1)
        hidden = self.fc(combined)
        out = self.output(hidden)

        return out


def load_cluster_data(file_path: Path):
    """
    Load and split cluster data
    
    Returns:
        train_df: Training DataFrame
        test_df: Testing DataFrame
    """
    df = pd.read_csv(file_path, usecols=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'], utc=False)
    
    # Extract cluster ID
    cluster_name = file_path.stem.split('_')[0]
    cluster_id = int(cluster_name.split('-')[1])
    df['cluster_id'] = cluster_id

    # Split train/test
    train_end = pd.to_datetime(config.TRAIN_END_DATE, utc=False)
    train_df = df[df['ds'] <= train_end]
    test_df = df[df['ds'] > train_end]
    
    return train_df, test_df


def prepare_lstm_data(train_df, test_df, look_back=None):
    """
    Prepare data for LSTM training
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        scaler: Fitted scaler for inverse transform
    """
    if look_back is None:
        look_back = config.LSTM_LOOK_BACK
    
    # Normalize per cluster
    unique_clusters = np.unique(train_df['cluster_id'].values)
    train_data_normalized = []
    test_data_normalized = []
    scalers = {}
    
    for cluster in unique_clusters:
        scaler = MinMaxScaler()
        train_cluster = train_df[train_df['cluster_id'] == cluster]['y'].values.reshape(-1, 1)
        test_cluster = test_df[test_df['cluster_id'] == cluster]['y'].values.reshape(-1, 1)
        
        scaler.fit(train_cluster)
        train_normalized = scaler.transform(train_cluster).flatten()
        test_normalized = scaler.transform(test_cluster).flatten()
        
        train_data_normalized.append((train_normalized, cluster))
        test_data_normalized.append((test_normalized, cluster))
        scalers[cluster] = scaler

    train_data_normalized.sort(key=lambda x: x[1])
    test_data_normalized.sort(key=lambda x: x[1])
    train_data = np.array([x[0] for x in train_data_normalized])
    test_data = np.array([x[0] for x in test_data_normalized])
    
    train_cluster_ids = train_df['cluster_id'].values
    test_cluster_ids = test_df['cluster_id'].values

    train_dataset = TimeSeriesDataset(train_data, train_cluster_ids, look_back)
    test_dataset = TimeSeriesDataset(test_data, test_cluster_ids, look_back)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.LSTM_BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.LSTM_BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    return train_loader, test_loader, scalers


def train_lstm_model(model, train_loader, criterion, optimizer, num_epochs, logger=None):
    """Train LSTM model"""
    model.to(config.DEVICE)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for X_batch, y_batch, cluster_ids in train_loader:
            X_batch = X_batch.to(config.DEVICE, non_blocking=True)
            y_batch = y_batch.to(config.DEVICE, non_blocking=True)
            cluster_ids = cluster_ids.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            y_pred = model(X_batch, cluster_ids)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')
        if logger:
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')


def predict_lstm(model, test_loader, scalers, target_cluster_id=None):
    """
    Make predictions using trained LSTM model
    
    Returns:
        predictions: Numpy array of predictions
    """
    model.eval()
    predictions = []
    
    with torch.inference_mode():
        for X_batch, _, cluster_ids in test_loader:
            X_batch = X_batch.to(config.DEVICE)
            cluster_ids = cluster_ids.to(config.DEVICE)
            
            y_pred = model(X_batch, cluster_ids)
            
            for pred, cluster_id in zip(y_pred.cpu().numpy(), cluster_ids.cpu().numpy()):
                if target_cluster_id is None or cluster_id[0] == target_cluster_id:
                    predictions.append((pred[0], cluster_id[0]))
    
    if not predictions:
        return np.array([])
    
    # Apply inverse transform per cluster
    final_predictions = []
    for pred, cluster_id in predictions:
        if cluster_id in scalers:
            pred_rescaled = scalers[cluster_id].inverse_transform([[pred]])[0, 0]
            final_predictions.append(pred_rescaled)
        else:
            final_predictions.append(pred)
    
    final_predictions = np.array(final_predictions)
    final_predictions = np.round(final_predictions)
    
    return final_predictions


def save_lstm_model(model, save_path=None):
    """Save LSTM model"""
    if save_path is None:
        save_path = config.LSTM_MODEL_PATH
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_lstm_model(model, load_path=None):
    """Load LSTM model"""
    if load_path is None:
        load_path = config.LSTM_MODEL_PATH
    
    model.load_state_dict(torch.load(load_path, map_location=config.DEVICE))
    model.eval()
    return model
