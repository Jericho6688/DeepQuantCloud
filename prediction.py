import torch
import torch.nn as nn
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import yaml
import sys
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


DB_PORT = "5432"
DB_HOST = os.environ.get("DB_HOST")  # "timescaledb"
NEW_DB_NAME = os.environ.get("NEW_DB_NAME")
NEW_DB_USER = os.environ.get("NEW_DB_USER")
NEW_DB_PASSWORD = os.environ.get("NEW_DB_PASSWORD")



# Define LSTM Model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, expected_length):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.expected_length = expected_length

    def forward(self, x):
        if x.shape[1] != self.expected_length:
            raise ValueError(f"Input sequence length must be {self.expected_length}, but got {x.shape[1]}")
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def load_config(config_path='prediction.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

def get_database_connection(db_config):
    """Create a database connection using SQLAlchemy."""
    try:
        engine = create_engine(f"postgresql://{NEW_DB_USER}:{NEW_DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{NEW_DB_NAME}")
        return engine
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        sys.exit(1)

def get_latest_data(engine, schema, table, n, feature_columns, order_column='datetime'):
    """Get the latest n data records from the specified table (assuming data is ordered by time)."""
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT {', '.join(feature_columns)}, {order_column}
                FROM "{schema}"."{table}"
                ORDER BY {order_column} ASC
                LIMIT {n}
            """)
            df = pd.read_sql(query, conn)

        # Handle missing values (choose the appropriate handling method according to your needs) Here it is assumed that there are no missing values, you can modify it according to the actual situation
        #df.fillna(method='ffill', inplace=True)

        # Set the time column as the index
        df = df.set_index(order_column)

        return df

    except Exception as e:
        print(f"Error getting data from the database: {e}")
        sys.exit(1)

def load_model(model_path, input_dim, hidden_dim, output_dim, expected_length, device):
    """Load the pre-trained LSTM model."""
    try:
        model = LSTMModel(input_dim, hidden_dim, output_dim, expected_length)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

def preprocess_data(df, feature_columns):
    """Preprocess data to match the model's input requirements, including data normalization."""
    scaler = MinMaxScaler()
    data = df[feature_columns].values
    data = scaler.fit_transform(data)
    data = data.astype(np.float32)
    return data, scaler

def predict(model, data, device):
    """Use the LSTM model to make predictions and calculate probabilities."""
    try:
        inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label_mapping = {0: -1, 1: 0, 2: 1}
        prob_dict = {label_mapping[i]: float(probabilities[i]) for i in range(len(probabilities))}
        return prob_dict
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)




def visualize_prediction(probabilities, schema, table, filename="prediction_graph.png"):
    """Generate a prediction image and display percentages, using a dark color consistent with the background template, and a moderate size."""
    labels = ['Increase', 'Flat', 'Decrease']
    percentages = [probabilities[1], probabilities[0], probabilities[-1]]

    # Use colors that are more coordinated with the background
    colors = ['#00FF00', '#0000FF', '#FF0000']  # Brighter green, blue, and red
    background_color = '#000000'  # Black background
    text_color = '#00FFFF'  # Cyan text

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=background_color)  # Adjust the figure size to 7x5 inches
    ax.set_facecolor(background_color) # Set the plotting area background color

    bars = ax.bar(labels, percentages, color=colors)

    plt.ylabel('Probability', color=text_color)
    plt.title(f'Real-time  {schema.upper()}.{table.upper()}  Prediction', color=text_color, fontsize=11) # Adjust title font size

    # Set coordinate axis colors
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_color(text_color)
    ax.spines['right'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    ax.tick_params(axis='x', colors=text_color, labelsize=9) # Adjust x-axis tick label font size
    ax.tick_params(axis='y', colors=text_color, labelsize=9) # Adjust y-axis tick label font size


    # Add percentage labels, and set color and font size
    for bar, v in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2%}', ha='center', va='bottom', color=text_color, fontsize=9)

    plt.ylim(0, 1.1)
    plt.tight_layout() # Adjust layout to avoid label overlap
    plt.savefig(filename)
    plt.show()


def main():
    # Load configuration file
    config = load_config()

    # Extract database configuration and prediction configuration
    db_config = config.get('database', {})
    prediction_config = config.get('prediction', {})

    # Check configuration file integrity
    required_keys = ['schema', 'table', 'feature_columns', 'order_column', 'data_length', 'model_path', 'input_dim', 'hidden_dim', 'output_dim', 'expected_length', 'device']
    if not all(key in prediction_config for key in required_keys):
        print("Missing some prediction-related parameters in the configuration file.")
        sys.exit(1)

    # Get parameters
    schema = prediction_config['schema']
    table = prediction_config['table']
    feature_columns = prediction_config['feature_columns']
    order_column = prediction_config['order_column']
    data_length = prediction_config['data_length']
    model_path = prediction_config['model_path']
    input_dim = prediction_config['input_dim']
    hidden_dim = prediction_config['hidden_dim']
    output_dim = prediction_config['output_dim']
    expected_length = prediction_config['expected_length']
    device = prediction_config['device']

    # Create database connection
    engine = get_database_connection(db_config)

    # Get data
    df = get_latest_data(engine, schema, table, data_length, feature_columns, order_column)
    if len(df) < data_length:
        print(f"The data obtained from the database is less than the expected {data_length} records.")
        sys.exit(1)

    # Preprocess data
    data_inputs, scaler = preprocess_data(df, feature_columns)

    # Load model
    model = load_model(model_path, input_dim, hidden_dim, output_dim, expected_length, device)

    # Data length check
    if data_inputs.shape[0] != expected_length:
        print(f"Data length mismatch: Expected {expected_length}, but got {data_inputs.shape[0]}.")
        sys.exit(1)

    # Make predictions
    probabilities = predict(model, data_inputs, device)

    # Find the predicted label
    predicted_label = max(probabilities, key=probabilities.get)

    # Output results
    print(f"Predicted label: {predicted_label}")
    print("Corresponding probabilities:")
    for label, prob in sorted(probabilities.items()):
        print(f"Label {label}: {prob:.4f}")

    # Visualize prediction results
    visualize_prediction(probabilities, schema, table)

if __name__ == "__main__":
    main()