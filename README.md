## ClaimVerify: AI-Powered Fact-Verification System

### 🔍 Project Overview

**ClaimVerify** is an intelligent, end-to-end **fact-verification system** that evaluates the credibility of user-submitted claims by combining Offline verified fact-check databases, Semantic retrieval and a Transformer-based classification model.

The system performs:

1. **Data Preprocessing** – Cleans and merges labeled claims from **PolitiFact** and **Snopes**.
2. **Semantic Embedding Generation** – Encodes claims using **MiniLM**.
3. **Efficient Retrieval** – Finds semantically similar past claims using a **FAISS index**.
4. **Transformer Classification** – Uses a fine-tuned **RoBERTa classifier** to predict a truth label.
5. **Explainability and UI** – Provides interpretability and user interaction through a **Streamlit app**.

---

### 🧱 System Architecture and Pipeline

The end-to-end workflow is:

**Data → Preprocessing → Retrieval → Classification → Interface**

#### 🔸 Components

* **Data Layer**
  Sources: PolitiFact and Snopes datasets merged into a unified `merged_factcheck_datasetcleaned.csv`.
  Processed embeddings and FAISS indices stored under `/data/processed/`.

* **Retrieval Layer**
  Encodes input claims and searches for nearest neighbors in the FAISS vector index.
  Output: similar verified claims and verdict metadata.

* **Model Layer**
  Fine-tuned **RoBERTa** classifier with **temperature-scaled calibration** for reliability.
  Outputs a confidence-weighted verdict (e.g., *True*, *Mostly True*, *False*).

* **Interface Layer**
  A **Streamlit** app that allows user interaction, visualization of retrieved evidence, and model explanations.

📊 The detailed architecture diagram as of Project Deliverable 2 is as follows:

<img width="830" height="939" alt="Deliverable2Architecture" src="https://github.com/user-attachments/assets/e673a5dc-6078-4793-b059-222af65645df" />


---

### 🗂 Repository Structure

```bash
UF-EEE6778-Fall25-TermProject/
│
├── data/
│   ├── raw/                      # Original PolitiFact & Snopes data
│   └── processed/
│       ├── merged_factcheck_datasetcleaned.csv
│       ├── embeddings/           # Numpy/Pickle embeddings & metadata
│       └── faiss_index/          # FAISS index and metadata CSV
│
├── models/
│   └── classifier/
│       └── roberta_finetuned/    # Model weights, tokenizer, calibration files
│
├── notebooks/
│   ├── data_preprocessingandmerge.ipynb
│   ├── OfflineRetrievalSystem.ipynb
│   ├── ClassifierTraining.ipynb
│   └── EDA_MergedDataset.ipynb
│
├── src/
│   ├── data_preprocess.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── model.py
│   ├── inference_pipeline.py
│   ├── explainability.py
│   └── __init__.py
│
├── ui/
│   ├── streamlit_app.py
│   ├── assets/
│   │   ├── dark_theme.css
│   │   └── light_theme.css
│   ├── InitialUI.png
│   └── ResultUI.png
│
├── results/
│   ├── EDAResults/               # Exploratory Data Analysis visuals
│   ├── UI/                       # Interface screenshots
│   └── training_logs/            # Model metrics and plots
│
├── Architecture/
│   ├── architecture_diagram.png
│   └── Deliverable2Architecture.png
│
├── DeliverableReports/
│   ├── ProjectDeliverablleReport1.pdf
│   └── ProjectDeliverableReport2.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

### ⚙️ Installation & Setup

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/UF-EEE6778-Fall25-TermProject.git
cd UF-EEE6778-Fall25-TermProject
```

#### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🧪 Model Training and Evaluation

Run the training notebooks sequentially:

1. **Data Preprocessing**

   ```bash
   jupyter notebook notebooks/data_preprocessingandmerge.ipynb
   ```

   * Cleans and merges datasets
   * Generates `merged_factcheck_dataset.csv`

2. **Retrieval Setup**

   ```bash
   jupyter notebook notebooks/OfflineRetrievalSystem.ipynb
   ```

   * Builds Sentence-BERT embeddings
   * Creates FAISS index for claim retrieval

3. **Classifier Training**

   ```bash
   jupyter notebook notebooks/ClassifierTraining.ipynb
   ```

   * Fine-tunes RoBERTa model
   * Outputs model metrics and saves weights to `/models/classifier/roberta_finetuned/`

4. **Evaluation**

   * Metrics, confusion matrix, and sample predictions saved to `/results/`

---

### 🖥️ Running the Streamlit Interface

To launch the prototype user interface:

```bash
streamlit run ui/streamlit_app.py
```

**Features available now (Deliverable 2):**

* Enter a textual claim
* Retrieve top-k similar verified claims
* Display corresponding verdicts and sources
* Show model’s predicted label with confidence
* Present dark/light themes for user preference

**Example Outputs:**
These are the sample UI Outputs :
<img width="682" height="963" alt="ResultUI" src="https://github.com/user-attachments/assets/da729dcf-a4eb-4713-ba3b-a676616a2faf" />


---

### 📊 Results Summary

|             Metric | Value (Prototype) | Notes                    |
| -----------------: | ----------------: | ------------------------ |
|           Accuracy |             ~78 % | On merged dataset        |
|           F1-Score |              0.75 | Macro-averaged           |
|  Calibration Error |             < 0.1 | Post temperature scaling |
| Retrieval Recall@3 |              0.82 | FAISS-based retrieval    |

---

### 🧩 Key Files and Artifacts

| File / Folder                               | Purpose                              |
| ------------------------------------------- | ------------------------------------ |
| `src/retrieval.py`                          | FAISS retrieval engine               |
| `src/inference_pipeline.py`                 | End-to-end claim-to-verdict pipeline |
| `src/explainability.py`                     | Captum/SHAP interpretability         |
| `models/classifier/roberta_finetuned/`      | Fine-tuned RoBERTa model             |
| `ui/streamlit_app.py`                       | User interface implementation        |
| `Architecture/Deliverable2Architecture.png` | System pipeline diagram              |

---

### 📦 Requirements

For required dependencies refer `requirements.txt`

---

### 🧑‍💻 Author

**Sai Satwik Yarapothini**
M.S. Applied Data Science
University of Florida
📧 [saisatwi.yarapot@ufl.edu](mailto:saisatwi.yarapot@ufl.edu)

---

### 🚀 Planned Features — Deliverable 3 (Final System)

| Category                             | Planned Addition               | Description                                                |
| ------------------------------------ | ------------------------------ | ---------------------------------------------------------- |
| **Web Search Integration**           | Online retrieval module        | Incorporate Google Fact-Check API / live news verification |
| **Enhanced Explainability**          | LIME + Attention visualization | Show why the model classified a claim a certain way        |
| **Confidence Calibration Dashboard** | Visualization of uncertainty   | Display prediction reliability to end users                |
| **Performance Optimization**         | Model pruning / batching       | Faster inference for real-time response                    |
| **Expanded UI Functionality**        | Upload batch claims            | CSV input + summary statistics                             |
| **Comprehensive Report**             | Deliverable 3 report           | Full evaluation metrics and deployment notes               |
