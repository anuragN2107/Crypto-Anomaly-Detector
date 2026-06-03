# Step 1: Initialize base optimized stable execution layer
FROM python:3.10-slim

# Step 2: System runtime package installs (Cleaned for Debian Trixie)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Establish isolation scope paths
WORKDIR /workspace

# Step 4: Isolate dependency footprint caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5: Copy processing scripts and model structures 
COPY . .

# Step 6: Expose official cloud platform endpoint routing target
EXPOSE 7860

# Step 7: Override Streamlit default configs to comply with target hosting specs
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]