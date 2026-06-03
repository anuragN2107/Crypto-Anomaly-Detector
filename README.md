# ⚡ CryptoPulse: Real-Time Cryptocurrency Anomaly Detector

An end-to-end, high-frequency cryptocurrency anomaly detection engine powered by a deep sequential **LSTM Autoencoder** neural network. This repository contains the core application architecture containerized with Docker and optimized for seamless deployment to Hugging Face Spaces.

🔗 **Live Deployment Dashboard:** [CryptoPulse on Hugging Face Spaces](https://huggingface.co/spaces/anuragN2107/crypto-anomaly-detector)

---

## 🎯 Business Problem & Objective
High-Frequency Trading (HFT) environments lose millions of dollars annually to swift microstructural market changes—such as flash crashes, volume pumps, and wash trading. Traditional lagging indicators (RSI, MACD) evaluate static past boundaries and fail to parse real-time sequence dependencies.

**CryptoPulse** solves this by establishing a continuous, deep learning data surveillance pipeline. By compressing incoming sequential signals and looking for variations in the **Reconstruction Loss**, the architecture proactively flags structural anomalies in seconds without needing manually configured, rigid threshold rules.

---

## 🛠️ Tools & Technologies Used

The project is built using a highly modular, decoupled stack spanning data processing, deep learning, containerization, and cloud deployment infrastructure:

### 🧠 Machine Learning & Core Logic
* **PyTorch (`torch.nn`):** Primary deep learning framework used to model and evaluate the gated Recurrent Neural Network (LSTM) layers.
* **Scikit-Learn (`MinMaxScaler`):** Utilized to normalize incoming highly volatile price and volume signals down into stable bounds ($[0, 1]$).
* **NumPy & Pandas:** Drives fast array manipulations and maintains the sliding lookback observation windows.
* **Joblib:** Handles clean, deterministic serialization of data preprocessing assets.

### 📊 Frontend & Visualization Engine
* **Streamlit:** Powers the dynamic, interactive web dashboard, handling system states and input widgets natively.
* **Plotly Graph Objects:** Generates the high-frequency vector charts to provide real-time visual telemetry without causing screen refresh stutter.
* **HTML5/CSS3 Custom Injection:** Bypasses native dashboard UI styles to inject a vibrant, neon-accented financial trading room layout.

### 📦 DevOps & Cloud Deployment
* **Docker:** Standardizes the isolated app footprint, ensuring consistent builds from a minimal Debian Linux core base.
* **Hugging Face Spaces:** Cloud host infrastructure utilizing basic CPU computational slices to run public-facing container web apps.

---

## 📂 Repository Architecture

```text
crypto-anomaly-detector/
├── app.py                  # Core Streamlit app script & CSS styling engine
├── Dockerfile              # Container manifest optimized for Hugging Face Spaces
├── requirements.txt        # Exact python package dependency registry
├── .gitignore              # Strict version-control exclusion rules
└── README.md               # Complete project roadmap & overview documentation
