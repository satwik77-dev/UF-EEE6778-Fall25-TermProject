# ClaimVerify: AI-Powered Fact Verification System (Deliverable 3)

### 🔍 Project Overview

**ClaimVerify** is an AI-driven pipeline designed to assess the credibility of user-submitted claims by combining:

- An **offline expert-verified fact-check database** (PolitiFact + Snopes)
- **Semantic retrieval** using MiniLM embeddings and a FAISS index
- A **fine-tuned RoBERTa v2 classifier** with calibrated probabilities
- A **hybrid retrieval fallback** (offline + Google stub)
- A redesigned **Streamlit interface** with improved usability and transparency

Deliverable 3 introduces a more robust, consistent, and user-centered system, focusing on **pipeline stability**, **uncertainty handling**, **explainability**, and **interaction quality**.

---

## 🧱 Updated System Architecture

The full end-to-end workflow now follows:

**User Claim → Preprocessing → Offline Retrieval → Optional Online Fallback → Classification → Calibration → Explainability → UI Display**

Key updates in Deliverable 3:

- Unified preprocessing for embeddings + classifier  
- Temperature scaling v2 for better-calibrated confidence  
- Integrated Gradients explanation heatmaps  
- Hybrid offline + online-stub retrieval  
- UI warnings for low similarity  
- Color-coded prediction bars based on thresholds  

📌 This architecture is illustrated here:

**Fig. 1 — Deliverable 3 System Architecture**  
<img width="436" height="823" alt="Deliverable3Architecture" src="https://github.com/user-attachments/assets/7db93553-6f99-4e26-80e3-72b212acf519" />


---

## 🗂 Repository Structure (Deliverable 3)

```bash
UF-EEE6778-Fall25-TermProject/
│
├── data/
│   ├── raw/                                  # Original PolitiFact & Snopes datasets
│   └── processed/
│       ├── merged_factcheck_datasetcleaned.csv
│       ├── embeddings/                       # MiniLM embeddings + metadata
│       └── faiss_index/                      # FAISS index + mapping files
│
├── models/
│   └── classifier/
│       └── roberta_finetuned_v2/             # Deliverable 3 trained model + calibration
│
├── notebooks/
│   ├── ClassifierTraining_Deliverable3.ipynb
│   ├── ClassifierTrainingDeliverable2.ipynb
│   ├── HybridRetrieval_Deliverable3.ipynb
│   ├── InferencePipeline_Deliverable3.ipynb
│   ├── OfflineRetrievalSystemDeliverable2.ipynb
│   ├── data_preprocessingandmerge.ipynb
│   ├── EDA_MergedDataset.ipynb
│   └── setup.ipynb
│
├── src/
│   ├── data_preprocess.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── model.py
│   ├── inference_pipeline.py                 # (Legacy) Deliverable 2 pipeline
│   ├── explainability.py
│   └── __init__.py
│
├── ui/
│   ├── InferencePipeline_Deliverable3.py     # NEW pipeline for UI
│   ├── Streamlit_UI_Deliverable3.py          # NEW improved UI
│   ├── assets/
│   │   ├── dark_theme.css
│   │   └── light_theme.css
│   └── UI Results/
│       ├── Deliverable2/
│       └── Deliverable3/                     # Deliverable 3 UI output screenshots
│
├── results/
│   ├── EDAResults/                           # Distribution plots, summaries
│   ├── Deliverable3/                                   # Screenshots used in report
│
├── Architecture/
│   ├── Deliverable3Architecture
│   ├── Deliverable2Architecture
│   └── Deliverable1Architecture
│
├── DeliverableReports/
│   ├── ProjectDeliverableReport1.pdf
│   ├── ProjectDeliverableReport2.pdf
│   └── ProjectDeliverableReport3.pdf         # Deliverable 3 Report
│
├── requirements.txt
├── .gitignore
└── README.md
````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/UF-EEE6778-Fall25-TermProject.git
cd UF-EEE6778-Fall25-TermProject
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Updated Deliverable 3 Pipeline

### **1. Run preprocessing (if needed)**

```bash
jupyter notebook notebooks/data_preprocessingandmerge.ipynb
```

### **2. Build/Load FAISS Retrieval System**

```bash
jupyter notebook notebooks/HybridRetrieval_Deliverable3.ipynb
```

### **3. Train or load the Deliverable 3 classifier**

```bash
jupyter notebook notebooks/ClassifierTraining_Deliverable3.ipynb
```

### **4. Run the new inference pipeline**

```bash
jupyter notebook notebooks/InferencePipeline_Deliverable3.ipynb
```

---

## 🖥️ Running the Deliverable 3 Streamlit UI

```bash
streamlit run ui/Streamlit_UI_Deliverable3.py
```

### New UI Enhancements (D3)

* Light theme with consistent typography
* Color-coded verdict bars (True = green, False = red, Uncertain = yellow)
* Warning banner for **low similarity**
* Token-level **explainability heatmap**
* Scrollable evidence cards
* Cleaner layout + improved spacing

📌 Example UI Output for Deliverable 3:

<img width="1908" height="991" alt="StreamlitUI_Deliverable3" src="https://github.com/user-attachments/assets/66a8a51c-29b3-45e2-b2b0-1385a3254d79" />
<img width="1892" height="979" alt="ui_sample_deliverable3" src="https://github.com/user-attachments/assets/4ba582a7-cedf-4610-889a-c77185fb2fca" />


---

## 📊 Deliverable 3 Performance Summary

| Metric              | Deliverable 2 | Deliverable 3 | Change         |
| ------------------- | ------------- | ------------- | -------------- |
| **Accuracy**        | 0.61          | 0.54          | –0.07          |
| **Macro F1**        | 0.49          | 0.5057        | ↑              |
| **Macro Precision** | 0.51          | 0.50          | –0.01          |
| **Macro Recall**    | 0.50          | 0.50          | ~              |
| **Brier Score**     | 0.1706        | 0.18          | Slightly bad |

📌 Full comparison table:
<img width="661" height="359" alt="comparison_table" src="https://github.com/user-attachments/assets/76a39703-a4d7-46d0-8d6e-63d48bfed7a3" />

📌 Deliverable 3 confusion matrix:
<img width="445" height="386" alt="confusion_matrix_D3" src="https://github.com/user-attachments/assets/8cedf87a-abfe-4b41-b087-c63003e55971" />


---

## 🧩 Key Files

| File                                      | Description                           |
| ----------------------------------------- | ------------------------------------- |
| `ui/Streamlit_UI_Deliverable3.py`         | Updated UI with new design + warnings |
| `ui/InferencePipeline_Deliverable3.py`    | Final inference logic used by UI      |
| `models/classifier/roberta_finetuned_v2/` | Final trained + calibrated model      |
| `src/explainability.py`                   | Integrated Gradients implementation   |
| `docs/architecture_deliverable3.png`      | Final architecture diagram            |

---

## ⚠️ Known Issues & Limitations

* **Uncertain** class still underperforms (dataset ambiguity)
* Online fallback currently uses **Google stub**, not full API
* Integrated Gradients is computationally expensive on CPU/MPS
* Retrieval similarity < 0.70 may produce weaker evidence

---

## 👤 Author

**Sai Satwik Yarapothini**
M.S. Applied Data Science, University of Florida
📧 [saisatwi.yarapot@ufl.edu](mailto:saisatwi.yarapot@ufl.edu)

---

## 🚀 Planned Work for Final Deliverable (D4)

* Full research-grade IEEE paper (10–12 pages)
* Final integration of live online fact-check API
* Better handling of the **Uncertain** class
* Robust user-study and usability evaluation
* Improved explanation consistency and potential LIME fallback
* Extended bias, fairness, and RAI analysis
