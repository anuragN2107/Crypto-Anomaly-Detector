import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import plotly.graph_objects as go

# ---------------------------------------------------------
# STYLING ENGINE: VIBRANT NEON TRADING WEB PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="CryptoPulse Pro", layout="wide")

st.markdown("""
    <style>
        /* Base Background Canvas */
        .stApp {
            background: linear-gradient(135deg, #0d0e15 0%, #171926 100%);
            color: #f1f3f9;
        }
        /* Top Navigation Header Accent Banner */
        .main-header {
            background: linear-gradient(90deg, #ff007f, #7928ca, #00dfd8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
        }
        .sub-header {
            text-align: center;
            color: #8a90af;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        /* Dashboard Container Cards */
        div[data-testid="metric-container"] {
            background-color: #1e2235 !important;
            border: 1px solid #2e3450 !important;
            padding: 15px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        /* Metric Label Text Overrides */
        div[data-testid="stMetricLabel"] {
            color: #a0aec0 !important;
            font-size: 0.95rem !important;
        }
        /* Sidebar Polish */
        .css-1cdg65q, .stSidebar {
            background-color: #11131c !important;
            border-right: 1px solid #25293c;
        }
    </style>
""", unsafe_allow_html=True)

# Re-declare Model Class Definition for instantiation
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.encoder_lstm = nn.LSTM(input_size=no_features, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.output_linear = nn.Linear(embedding_dim, no_features)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        repeat_hidden = hidden.permute(1, 0, 2).repeat(1, self.seq_len, 1)
        x_decoded, _ = self.decoder_lstm(repeat_hidden)
        return self.output_linear(x_decoded)

# Load Trained Weights and Scalar
@st.cache_resource
def load_assets():
    scaler = joblib.load('data_scaler.pkl')
    model = LSTMAutoencoder(seq_len=30, no_features=2)
    model.load_state_dict(torch.load('lstm_anomaly_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return scaler, model

try:
    scaler, model = load_assets()
except Exception as e:
    st.error("Model state files missing. Please ensure weights are in the directory.")

# ---------------------------------------------------------
# RENDER LAYOUT
# ---------------------------------------------------------
st.markdown('<p class="main-header">⚡ CRYPTOPULSE ANOMALY DETECTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">High-Frequency Deep Learning Surveillance Engine via LSTM Autoencoders</p>', unsafe_allow_html=True)

# Sidebar Interactive Controls
st.sidebar.markdown("## 🎛️ Control Panel")
sensitivity = st.sidebar.slider("Anomaly Threshold Sensitivity", min_value=0.005, max_value=0.10, value=0.025, step=0.005)
feed_speed = st.sidebar.slider("Live Market Refresh Rate (Seconds)", min_value=0.2, max_value=2.0, value=0.5, step=0.1)

start_btn = st.sidebar.button("▶️ Launch Stream")

# Metrics Display Row placeholders
m1, m2, m3 = st.columns(3)
metric_p = m1.empty()
metric_v = m2.empty()
metric_s = m3.empty()

chart_placeholder = st.empty()

# Streaming execution Loop logic
if start_btn:
    # Initialize background running mock live state variables
    current_price = 50000.0
    history_buffer = []
    
    # Loop representing live telemetry sequence windows
    for tick in range(100):
        time.sleep(feed_speed)
        
        # Inject standard random walk or spikes randomly
        rand_draw = np.random.rand()
        if rand_draw > 0.96: # Inject sudden spike anomaly
            price_change = np.random.normal(-0.08, 0.02)
            volume_t = np.random.randint(1500, 3000)
        else:
            price_change = np.random.normal(0, 0.002)
            volume_t = np.random.randint(50, 200)
            
        current_price *= (1 + price_change)
        history_buffer.append([current_price, volume_t])
        
        if len(history_buffer) < 30:
            metric_p.metric("Latest Market Price", f"${current_price:,.2f}", "Buffering Sequence...")
            continue
            
        # Retain strictly the moving target window size
        current_window = history_buffer[-30:]
        scaled_window = scaler.transform(current_window)
        
        # Convert array to tensor format
        input_tensor = torch.tensor([scaled_window], dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed_output = model(input_tensor)
            # Calculate standard reconstruction error metric
            loss = torch.mean((input_tensor - reconstructed_output) ** 2).item()
            
        is_anomaly = loss > sensitivity
        
        # Visual State Updates depending on severity evaluation
        if is_anomaly:
            status_text = "🚨 ANOMALY FLAGGED"
            status_color = "#FF007F" 
        else:
            status_text = "✅ STABLE OPERATION"
            status_color = "#00DFD8"
            
        metric_p.metric("Latest Market Price", f"${current_price:,.2f}", f"{price_change*100:.3f}%")
        metric_v.metric("Tick Trading Volume", f"{volume_t} Contracts")
        metric_s.metric("Engine Health State", status_text, f"Loss Score: {loss:.5f}", delta_color="inverse" if is_anomaly else "normal")
        
        # Build vibrant analytical plot
        plot_df = pd.DataFrame(current_window, columns=['Price', 'Volume'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=plot_df['Price'], mode='lines+markers', name='Price Line', line=dict(color=status_color, width=3)))
        fig.update_layout(
            title=f"Sequence Visualizer Window (Current Reconstruction Loss: {loss:.5f})",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#f1f3f9",
            margin=dict(l=20, r=20, t=40, b=20),
            height=380
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
else:
    st.info("System Ready. Click 'Launch Stream' on the Control Panel sidebar to begin active data logging.")