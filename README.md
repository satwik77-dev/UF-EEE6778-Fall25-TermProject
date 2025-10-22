## 🧠 ClaimVerify: AI-Powered Fact Verification System

---

### 🔍 Project Overview

**ClaimVerify** is an intelligent **Fact-verification system** that combines offline fact-check databases with real-time web search capabilities.
The goal is to analyze a user-provided claim and determine its credibility by retrieving similar fact-checked statements and generating an interpretive model verdict.

The system integrates:

* **Offline structured datasets** (PolitiFact & Snopes) for high-quality, labeled claims.
* **Transformer-based semantic embeddings (Sentence-BERT)** for similarity detection.
* **FAISS retrieval** for efficient search.
* **Streamlit interface** for intuitive, interactive claim analysis.

---

### 🧩 Repository Structure

```bash
UF-EEE6778-Fall25-TermProject/
│
├── data/
│   ├── raw/                     # Original datasets (PolitiFact, Snopes)
│   └── processed/               # Cleaned & merged offline database
│
├── notebooks/
│   └── setup.ipynb              # Environment verification & dataset loading
│
├── src/
│   ├── preprocess.py            # Text cleaning & harmonization
│   ├── embeddings.py            # Sentence-BERT embedding functions
│   ├── retrieval.py             # FAISS search + API fallback logic
│   ├── model.py                 # Model training & inference
│   └── explainability.py        # SHAP / interpretability visuals
│
├── ui/
│   └── streamlit_app.py         # Placeholder for Streamlit interface
│
├── results/
│   ├── EDAResults/              # Plots from exploratory analysis
│   └── SampleUI/                # Example outputs / predictions
│
├── docs/
│   └── architecture_diagram.png # Planned architecture diagram
│
├── DeliverableReports/
│   └── Deliverable1Report.pdf   # Technical Blueprint Report
│
├── requirements.txt             # Dependency list
├── .gitignore                   # Ignore unnecessary files
└── README.md                    # You are here!
```

---

### ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/UF-EEE6778-Fall25-TermProject.git
   cd UF-EEE6778-Fall25-TermProject
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate     # For macOS/Linux
   venv\Scripts\activate        # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

### 🧪 Running the Notebook

To verify the setup and preview early exploration:

```bash
jupyter notebook notebooks/setup.ipynb
```

The notebook performs:

* Environment check (package imports)
* Dataset loading from `/data/raw/`
* Basic exploration (counts, verdict distribution, sample plots)

---

### 🗂 Dataset Information

| Dataset               | Source                                        | Type           | Records | Label Categories                                                 |
| --------------------- | --------------------------------------------- | -------------- | ------- | ---------------------------------------------------------------- |
| PolitiFact Fact-Check | Kaggle (https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset?resource=download&select=politifact_factcheck_data.json) | Tabular (JSON) | ~21,000 | true, mostly true, half true, mostly false, false, pants-on-fire |
| Snopes Fact-News      | Kaggle (https://www.kaggle.com/datasets/ambityga/snopes-factnews-data/data?select=snopeswithsum.csv)         | Tabular (CSV)  | ~10,000 | true, false, miscaptioned, mixture                               |

The final **offline database** (`data/processed/offline_db.csv`) merges these two datasets for structured retrieval.
When a claim is not found offline, the system can query the **Google Custom Search API** for live fact-checking articles.

---

### 🖥️ Planned Interface

The Streamlit app (`ui/streamlit_app.py`) will:

* Accept a **claim text input**
* Retrieve **similar existing claims**
* Display **model verdict + interpretability insights**
* Offer a **user-friendly confidence visualization**

---

### Reproducibility Notes

* All required datasets (except large originals) are included in `data/`
* Scripts in `/src` are modular and can be run independently
* Results and EDA plots are stored under `/results/`
* `setup.ipynb` runs without modification on any standard Python 3.9+ environment

---

### 🧑‍💻 Author

**Sai Satwik Yarapothini**
M.S. Applied Data Science
University of Florida
📧 *[saisatwi.yarapot@ufl.edu](mailto:saisatwi.yarapot@ufl.edu)*
