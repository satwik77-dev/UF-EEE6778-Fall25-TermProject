ClaimVerify: AI-Powered Fact Verification System

🔍 Project Overview:

ClaimVerify is an AI-powered fact verification system designed to assist users in assessing the credibility of natural-language claims through a transparent, evidence-driven, and uncertainty-aware pipeline. Rather than acting as an automated arbiter of truth, the system is explicitly designed as a decision-support tool that grounds predictions in verifiable evidence and clearly communicates model confidence and uncertainty.

- The system integrates multiple complementary components to address the challenges of real-world misinformation detection:

a. Offline expert-verified fact-check databases, constructed from PolitiFact and Snopes

b. Semantic retrieval using MiniLM sentence embeddings and a FAISS similarity index

c. A fine-tuned RoBERTa-based classifier with temperature-scaled probability calibration

d. A hybrid retrieval fallback mechanism using the Google Custom Search API when offline evidence is insufficient

e. Token-level explainability via Integrated Gradients to improve transparency

f. A user-centered Streamlit interface designed to emphasize interpretability, evidence traceability, and uncertainty awareness

By combining retrieval, classification, calibration, and explainability into a single end-to-end system, ClaimVerify aims to bridge the gap between research prototypes and deployable fact-verification tools suitable for real-world use.

🧱 System Architecture:

ClaimVerify follows a modular, hybrid system architecture that integrates offline semantic retrieval with online fallback search, calibrated neural inference, and interpretable output generation.

<img width="1326" height="568" alt="ClaimVerifyFinalArchitecture" src="https://github.com/user-attachments/assets/f52cecf5-8eb2-4d85-b6e2-54176342a6ea" />


The end-to-end pipeline proceeds as follows:

User Claim → Text Preprocessing → Offline Semantic Retrieval → Conditional Online Fallback → Classification → Probability Calibration → Explainability → UI Presentation

Key architectural characteristics include:

- Unified preprocessing across retrieval and classification stages to ensure consistency

- Near-perfect Recall@k performance in offline semantic retrieval

- Explicit confidence calibration to mitigate overconfident predictions

- Clearly defined uncertainty thresholds that defer ambiguous claims

- Evidence-grounded explanations that support user trust and transparency

This architecture enables ClaimVerify to handle both previously fact-checked claims and emerging or novel claims in a controlled and interpretable manner.

🗂 Repository Structure
```bash
UF-EEE6778-Fall25-TermProject/
│
├── data/
│   ├── raw/                                   # Original PolitiFact & Snopes datasets
│   └── processed/
│       ├── merged_factcheck_dataset_cleaned.csv  # Final unified offline dataset
│       └── faiss_index/                       # FAISS index + metadata files
│
├── models/
│   └── classifier/
│       └── roberta_finetuned_v2/              # Final trained RoBERTa model + calibration
│           ├── pytorch_model.bin
│           ├── config.json
│           ├── tokenizer.json
│           ├── label_mapping.pkl
│           └── temperature_scaling.pt
│
├── notebooks/
│   ├── data_preprocessing_and_merge.ipynb     # Dataset cleaning & merging
│   ├── EDA_MergedDataset.ipynb                 # Exploratory data analysis
│   ├── ClassifierTraining_Deliverable3.ipynb  # Classifier training & calibration
│   ├── HybridRetrieval_Deliverable3.ipynb     # FAISS + MiniLM retrieval evaluation
│   ├── InferencePipeline_Deliverable3.ipynb   # End-to-end pipeline testing
│   └── google_custom_search_config.png        # Google Custom Search API configuration
│
├── src/                                       # Modular backend implementation
│   ├── __init__.py
│   ├── preprocessing.py                      # Text normalization & cleaning
│   ├── embeddings.py                         # MiniLM embedding loader (safe CPU)
│   ├── retrieval.py                          # FAISS-based semantic retrieval
│   ├── model.py                              # Calibrated RoBERTa classifier
│   ├── explainability.py                     # Integrated Gradients explanations
│   └── inference_pipeline.py                 # Hybrid inference orchestration
│
├── ui/
│   ├── ClaimVerify_FinalSystem.py             # Final all-in-one Streamlit application code
│   └── UI Results/
│       └── FinalResults/                     # Final UI screenshots (report & demo)
│           ├── ui_main_dashboard.png
│           ├── ui_evidence_cards.png
│           ├── ui_token_importance.png
│           ├── ui_uncertainty_fallback.png
│           └── ui_totaloutput.png
│
├── results/
│   └── Final_confusion_matrix.png            # Final test-set confusion matrix
│
├── Architecture/
│   └── ClaimVerify_FinalArchitecture.png     # Final system architecture diagram
│
├── DeliverableReports/
│   └── Final_IEEE_Report.pdf                 # IEEE-format final report
│
├── requirements.txt                          # Project dependencies
├── .gitignore
└── README.md                                 # Project documentation
```

⚙️ Installation & Setup:

1️⃣ Clone the Repository:
git clone https://github.com/satwik77-dev/UF-EEE6778-Fall25-TermProject

cd UF-EEE6778-Fall25-TermProject

2️⃣ Create and Activate a Virtual Environment:

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

3️⃣ Install Dependencies:

pip install -r requirements.txt

🖥️ Running the Streamlit Application:

streamlit run ui/ClaimVerify_FinalSystem.py


The application launches a browser-based interface that allows users to:

a. Enter a natural-language claim

b. View retrieved offline or online evidence

c. Inspect calibrated confidence scores

d. Analyze token-level explanation heatmaps

e. Identify when hybrid fallback retrieval is triggered
<img width="1899" height="414" alt="ui_main_dashboard" src="https://github.com/user-attachments/assets/d57a0d70-5d91-4f19-87e2-7004618fb579" />


- Sample Interface Output:

Example Results

1. Offline Retrieval (Unified Dataset)
<img width="1898" height="700" alt="ui_totaloutput" src="https://github.com/user-attachments/assets/1ddb468e-83c7-4c90-bb7a-1f88f3663a8a" />


2. Google Search Fallback (Low Similarity / Uncertain Claims)
<img width="1900" height="694" alt="ui_uncertainty_fallback" src="https://github.com/user-attachments/assets/08bd6658-f853-4cf4-83d0-fbed16b293cf" />


📊 Performance Summary (Final System):

The final ClaimVerify system achieves balanced multi-class performance on a three-class fact-verification task while prioritizing calibrated confidence and conservative decision-making.



<img width="290" height="251" alt="Screenshot 2025-12-11 at 21 08 25" src="https://github.com/user-attachments/assets/cd9e4ab2-abd4-428f-951f-480f86f1fbb3" />
<img width="289" height="79" alt="Screenshot 2025-12-11 at 21 08 12" src="https://github.com/user-attachments/assets/9afdd4a0-a54e-44b5-b0f7-8e045bdd8859" />
<img width="262" height="111" alt="Screenshot 2025-12-11 at 21 08 02" src="https://github.com/user-attachments/assets/7c05a7c0-6db2-4ede-ab47-a5f15e212660" />


Key highlights include:

- Overall accuracy of 54% on the test set

- Macro F1-score of 0.49, indicating balanced performance across classes

- Weighted F1-score of 0.56, reflecting strong misinformation detection

- Calibrated Brier Score of 0.1989, demonstrating improved probability reliability

- Near-perfect Recall@k for offline semantic retrieval

These results emphasize reliability and interpretability over inflated accuracy, which is essential for responsible deployment.

🧩 Key Components:
ClaimVerify has several key components that forms the system :
- MiniLM + FAISS	High-speed semantic evidence retrieval
- RoBERTa	Claim classification
- Temperature Scaling	Probability calibration
- Integrated Gradients	Token-level explainability
- Google Custom Search	Hybrid fallback retrieval
- Streamlit	User-facing interface
  
⚠️ Known Limitations:

- The Uncertain class remains challenging due to inherent label ambiguity

- Integrated Gradients explanations are computationally expensive on CPU

- Live Google API usage is rate-limited with 100 free queries/day

- Performance depends on coverage and quality of existing fact-check databases

🔮 Future Improvements:

- Targeting 80–85% accuracy through ensemble and evidence-aware architectures

- Retrieval-aware and multi-hop reasoning models

- Expanded multilingual support

- Formal user studies for interface validation

- More efficient explanation methods

- Continuous and automated web-index updates

👤 Author

Sai Satwik Yarapothini
M.S. Applied Data Science
University of Florida
📧 saisatwi.yarapot@ufl.edu

📄 License

This project was developed for academic purposes as part of the University of Florida Applied Machine Learning II coursework and is intended for research and educational use.
