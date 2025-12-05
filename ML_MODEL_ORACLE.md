# AAPL Market Oracle Implementation Guide (v2)

## Overview

A lightweight, self-supervised model trained on 87,000 minutes of AAPL market data that OpenRouter models can query for deep pattern insights.

## Key Insight

Instead of placeholder labels like `[bullish/bearish]`, we **compute actual labels from future price movements** in your data.

## Architecture

```
OpenRouter Models (Claude, GPT, etc.)
          │
          │ "What intraday patterns exist in AAPL?"
          │ "When does momentum typically reverse?"
          │ "What's the average move after high volume spikes?"
          ▼
    ┌─────────────────┐
    │  AAPL Oracle    │  ← Lightweight model (TinyLlama 1.1B or Phi-2)
    │  (87k patterns) │     trained on computed labels
    └─────────────────┘
          │
          │ "AAPL reverses 67% of the time after 3 consecutive
          │  green candles in the 10:30-11:00 window. Average
          │  reversal magnitude: 0.4%. Volume confirmation increases
          │  success to 78%."
          ▼
OpenRouter Models generate superior algorithms
```

## Prerequisites

- **Hardware**: MacBook M5 Pro (24GB RAM) - sufficient for 1-3B models
- **Data**: 87k 1-minute AAPL OHLCV records in MongoDB
- **Model**: TinyLlama-1.1B or Phi-2 (2.7B) - runs well on Apple Silicon
- **Time**: Initial setup ~2-4 hours training

---

## Phase 1: Environment Setup

### Install Dependencies

```bash
cd /Users/jaiminpatel/github/algoclash-v1-test
mkdir -p oracle
cd oracle

# Apple Silicon optimized PyTorch
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft bitsandbytes
pip install pymongo flask requests pandas numpy scikit-learn
```

### Directory Structure

```
oracle/
├── data/
│   └── aapl_oracle_training.jsonl
├── models/
│   └── aapl-oracle/
├── prepare_data_v2.py      # Self-labeling data prep
├── fine_tune_lightweight.py # Training script
├── oracle_api.py           # API server
└── test_oracle.py
```

---

## Phase 2: Self-Labeling Data Preparation (CRITICAL FIX)

This is the key improvement - we **compute real labels from your data**.

### File: `oracle/prepare_data_v2.py`

```python
import json
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def compute_labels_from_future(records, lookahead=15):
    """Compute labels based on future price movements"""
    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Sort by time
    df = df.sort_values('datetime')

    # Calculate future returns
    df['future_close'] = df['close'].shift(-lookahead)
    df['label'] = np.where(df['future_close'] > df['close'], 1, 0)

    # Drop rows with NaN labels
    labeled_data = df.dropna(subset=['label'])

    return labeled_data

def create_qa_training_pairs(labeled_data):
    """Create Q&A style training pairs from labeled data"""
    qa_pairs = []

    for _, row in labeled_data.iterrows():
        question = f"What is the likely price movement of AAPL after {row['datetime']}?"
        answer = "The analysis is based on historical patterns observed in AAPL's price movements."

        # Append additional context from the data
        answer += f" Context: At {row['datetime']}, AAPL's price was ${row['close']:.2f} with a label of {row['label']}."

        qa_pairs.append({
            "question": question,
            "answer": answer
        })

    return qa_pairs

def create_training_data():
    """Main function to create the training dataset"""
    
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017')
    db = client.ai_trader_battlefield
    collection = db.AAPL_simulation  # Adjust collection name if different
    
    print("📊 Fetching AAPL data from MongoDB...")
    records = list(collection.find({}, {'_id': 0}).sort('datetime', 1))
    print(f"✅ Found {len(records)} records")
    
    if len(records) < 1000:
        print("❌ Not enough data. Need at least 1000 records.")
        return
    
    # Step 1: Compute labels from actual future data
    print("🔄 Computing labels from future price movements...")
    labeled_data = compute_labels_from_future(records, lookahead=15)
    print(f"✅ Labeled {len(labeled_data)} data points")
    
    # Step 2: Create Q&A training pairs
    print("📝 Creating pattern-based training pairs...")
    training_pairs = create_qa_training_pairs(labeled_data)
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    output_file = 'data/aapl_oracle_training.jsonl'
    
    with open(output_file, 'w') as f:
        for item in training_pairs:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✅ Created {len(training_pairs)} training examples")
    print(f"📁 Saved to: oracle/{output_file}")
    
    return len(training_pairs)

if __name__ == '__main__':
    create_training_data()
```

---

## Phase 3: Lightweight Model Training

Use **TinyLlama-1.1B** or **Phi-2** instead of Mistral-7B.

### File: `oracle/fine_tune_lightweight.py`

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import os

def fine_tune_aapl_oracle():
    """Fine-tune TinyLlama or Phi-2 on AAPL market data"""

    print("🚀 Starting AAPL Oracle fine-tuning...")

    # Model configuration
    model_name = "mistralai/TinyLlama-1.1B"  # or "Meta/Phi-2"
    output_dir = "./models/aapl-oracle"

    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("📊 Loading training data...")
    dataset = load_dataset('json', data_files='data/aapl_oracle_training.jsonl')

    def format_qa_pair(example):
        return {
            "text": f"Q: {example['question']}\nA: {example['answer']}"
        }

    dataset = dataset.map(format_qa_pair)

    print("🎯 Starting training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,  # Small batch for M5 Pro
        gradient_accumulation_steps=16,  # Effective batch size = 16
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none"  # Disable wandb/etc logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ AAPL Oracle model trained and saved!")
    print(f"📁 Model location: {output_dir}")

    return output_dir

if __name__ == '__main__':
    fine_tune_aapl_oracle()
```

---

## Phase 4: Oracle API Server

### File: `oracle/oracle_api.py`

```python
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the fine-tuned AAPL oracle model
oracle = pipeline(
    "text-generation",
    model="./models/aapl-oracle",
    tokenizer="./models/aapl-oracle",
    max_length=200,
    num_return_sequences=1,
    pad_token_id=50256
)

@app.route('/analyze', methods=['POST'])
def analyze_market():
    data = request.json
    query = data.get('query', '')

    # Generate response from the oracle model
    response = oracle(query)

    return jsonify({
        "insights": response[0]['generated_text'],
        "query": query
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

---

## Phase 5: Integration with algo_gen.py

### Add to `backend/open_router/algo_gen.py`:

```python
# Integration code for querying the oracle
```

---

## Scaling to Multiple Stocks

### Creating Models for Multiple Stocks

1. **Prepare Stock-Specific Data**
   - Extract data for each stock from your MongoDB database.
   - Use the same `prepare_data_v2.py` script but filter by stock ticker:
     ```python
     records = list(collection.find({"ticker": "AAPL"}, {'_id': 0}).sort('datetime', 1))
     ```
   - Save training data for each stock in separate files:
     ```
     data/aapl_oracle_training.jsonl
     data/msft_oracle_training.jsonl
     data/tsla_oracle_training.jsonl
     ```

2. **Train Separate Models**
   - Use the same `fine_tune_lightweight.py` script but specify different output directories for each stock:
     ```bash
     python fine_tune_lightweight.py --data data/aapl_oracle_training.jsonl --output models/aapl-oracle
     python fine_tune_lightweight.py --data data/msft_oracle_training.jsonl --output models/msft-oracle
     python fine_tune_lightweight.py --data data/tsla_oracle_training.jsonl --output models/tsla-oracle
     ```

3. **Deploy Stock-Specific APIs**
   - Run separate instances of the `oracle_api.py` server for each stock:
     ```bash
     python oracle_api.py --model models/aapl-oracle --port 5001
     python oracle_api.py --model models/msft-oracle --port 5002
     python oracle_api.py --model models/tsla-oracle --port 5003
     ```

4. **Integrate with OpenRouter**
   - Modify your `algo_gen.py` to query the appropriate oracle based on the stock ticker:
     ```python
     def get_oracle_insights(ticker, query):
         oracle_ports = {"AAPL": 5001, "MSFT": 5002, "TSLA": 5003}
         port = oracle_ports.get(ticker.upper())
         if not port:
             return None
         response = requests.post(f"http://localhost:{port}/analyze", json={"query": query})
         return response.json().get("insights")
     ```

---

## Improving Labeling Strategy

### Enhanced Labeling

1. **Magnitude-Based Labels**
   - Use thresholds to classify movements:
     - `1` for significant upward movement (e.g., >0.5%).
     - `0` for no significant movement (e.g., -0.5% to +0.5%).
     - `-1` for significant downward movement (e.g., <-0.5%).

2. **Contextual Labels**
   - Add additional features to the training data:
     - **Volume spikes**: Compare current volume to a rolling average.
     - **Time-of-day**: Include time slots (e.g., morning, midday, close).
     - **Volatility**: Calculate intraday volatility and use it as a feature.

3. **Example Enhanced Labeling**
   ```python
   def compute_labels_from_future(records, lookahead=15):
       df = pd.DataFrame(records)
       df['datetime'] = pd.to_datetime(df['datetime'])
       df = df.sort_values('datetime')

       # Calculate future returns
       df['future_close'] = df['close'].shift(-lookahead)
       df['price_change_pct'] = (df['future_close'] - df['close']) / df['close'] * 100

       # Magnitude-based labels
       df['label'] = np.where(df['price_change_pct'] > 0.5, 1,
                              np.where(df['price_change_pct'] < -0.5, -1, 0))

       # Add contextual features
       df['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
       df['time_slot'] = df['datetime'].dt.hour

       return df.dropna()
   ```

---

## Managing Overfitting

### Strategies to Reduce Overfitting

1. **Regular Retraining**
   - Automate the retraining process to include the latest data.
   - Use a sliding window approach (e.g., last 1 year of data).

2. **Data Augmentation**
   - Introduce noise to the data (e.g., slight variations in prices) to make the model more robust.
   - Use synthetic data to simulate rare scenarios.

3. **Validation**
   - Split the data into training, validation, and test sets.
   - Use early stopping during training to prevent overfitting.

4. **Dropout Regularization**
   - Add dropout layers during fine-tuning to reduce overfitting.

---

## Latency Optimization

### Reducing Latency

1. **Model Quantization**
   - Use 4-bit quantization to reduce inference time:
     ```python
     model = AutoModelForCausalLM.from_pretrained(
         model_name,
         torch_dtype=torch.float16,
         device_map="auto",
         quantization_config={"bits": 4}
     )
     ```

2. **Batch Processing**
   - Allow the API to handle multiple queries in a single batch:
     ```python
     @app.route('/analyze_batch', methods=['POST'])
     def analyze_batch():
         queries = request.json.get('queries', [])
         responses = [oracle(query) for query in queries]
         return jsonify(responses)
     ```

3. **Deploy on Optimized Hardware**
   - Use a dedicated server with a GPU for faster inference.

---

## Scaling to Larger Models and Data

### Future Plan

1. **Use Larger Models**
   - Transition to models like GPT-3 or Llama-2 for more complex insights.
   - Fine-tune these models on your expanded dataset.

2. **Incorporate More Data**
   - Add data for more stocks and longer time periods.
   - Include alternative data sources (e.g., news sentiment, macroeconomic indicators).

3. **Distributed Training**
   - Use cloud-based solutions (e.g., AWS, Google Cloud) for training larger models.

---

## Addressing RAM Bottlenecks

### Dynamic LoRA Adapter Loading

**Issue:** Running separate APIs for AAPL, MSFT, and TSLA simultaneously can exhaust RAM, even with 4-bit quantization.
**Fix:** Instead of loading multiple full models, load a **single base model** (e.g., TinyLlama) and dynamically swap the lightweight LoRA adapters for each stock on the fly.

**Implementation Strategy:**
1. Load the base model once into memory.
2. When a request comes for AAPL, load the AAPL LoRA adapter.
3. When a request comes for MSFT, swap the AAPL adapter for the MSFT adapter.

**Code Example:**
```python
from peft import PeftModel

# Load base model once
base_model = AutoModelForCausalLM.from_pretrained("mistralai/TinyLlama-1.1B", device_map="auto")

def get_prediction(ticker, query):
    adapter_path = f"./models/{ticker.lower()}-oracle"
    
    # Load specific adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Generate prediction
    response = generate_response(model, query)
    
    # Unload adapter to free resources (optional, or just swap)
    model.unload()
    return response
```

---

## Mitigating "Hallucination" of Confidence

### Handling Uncertainty

**Issue:** The model might learn to sound confident ("Bullish with 90% certainty") even when the signal is weak, simply because the training data had clear labels.
**Fix:** Ensure your training data includes "Neutral" or "Uncertain" examples where the price didn't move much, so the model learns to say "I don't know" or "No clear pattern."

**Implementation Strategy:**
1. **Define Neutral Thresholds:** If price movement is within a small range (e.g., -0.1% to +0.1%), label it as "Neutral".
2. **Training Data Augmentation:** Explicitly add examples where the correct answer is "Market conditions are choppy; no clear direction."

**Code Example:**
```python
# In compute_labels_from_future
df['label'] = np.where(df['price_change_pct'] > 0.5, "Bullish",
                       np.where(df['price_change_pct'] < -0.5, "Bearish", "Neutral"))

# In create_qa_training_pairs
if row['label'] == "Neutral":
    answer = "The market is currently range-bound with no clear directional signal. Caution is advised."
```
