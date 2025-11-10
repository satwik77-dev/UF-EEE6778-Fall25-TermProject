# Unified Dataset Builder - PolitiFact + Snopes

#Importing Required Libraries
import pandas as pd
import json
from pathlib import Path

def build_unified_factcheck_dataset():
    """Preprocess and merge PolitiFact + Snopes datasets into a unified schema."""

    politifact_path = Path('/Users/satwik/Documents/GitHub/UF-EEE6778-Fall25-TermProject/data/raw/politifact_factcheck_data.json')
    snopes_path = Path('/Users/satwik/Documents/GitHub/UF-EEE6778-Fall25-TermProject/data/raw/snopeswithsum.csv')
    output_path = Path('/Users/satwik/Documents/GitHub/UF-EEE6778-Fall25-TermProject/data/processed/merged_factcheck_dataset.csv')

    # Load PolitiFact Dataset
    politifact_data = []
    try:
        with open(politifact_path, "r") as f:
            politifact_data = json.load(f)
    except json.JSONDecodeError:
        with open(politifact_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    politifact_data.append(json.loads(line))
    df_politifact = pd.DataFrame(politifact_data)

    # Load Snopes
    df_snopes = pd.read_csv(snopes_path)

    # PolitiFact Processing
    df_politifact = df_politifact.rename(columns={
        "statement": "claim_text",
        "verdict": "verdict_original",
        "statement_originator": "originator",
        "factcheck_analysis_link": "url"
    })
    df_politifact["dataset_source"] = "PolitiFact"
    df_politifact["summary"] = None
    df_politifact["claim_id"] = [f"P{i:05d}" for i in range(1, len(df_politifact)+1)]

    mapping_pf = {
        "true": "Likely True",
        "mostly-true": "Likely True",
        "half-true": "Uncertain",
        "mostly-false": "Likely False",
        "false": "Likely False",
        "pants-fire": "Likely False"
    }
    df_politifact["verdict_mapped"] = df_politifact["verdict_original"].str.lower().map(mapping_pf)

    df_politifact = df_politifact[
        ["claim_id", "claim_text", "verdict_original", "verdict_mapped",
         "summary", "originator", "url", "dataset_source"]
    ]

    # Snopes Processing
    df_snopes = df_snopes.rename(columns={
        "claim": "claim_text",
        "rate": "verdict_original",
        "summary": "summary"
    })
    df_snopes["originator"] = None
    df_snopes["url"] = None
    df_snopes["dataset_source"] = "Snopes"
    df_snopes["claim_id"] = [f"S{i:05d}" for i in range(1, len(df_snopes)+1)]

    mapping_snopes = {
        "true": "Likely True",
        "mostly true": "Likely True",
        "false": "Likely False",
        "mostly false": "Likely False",
        "mixture": "Uncertain",
        "unproven": "Uncertain",
        "miscaptioned": "Uncertain",
        "legend": "Uncertain",
        "outdated": "Uncertain",
        "scam": "Likely False",
        "satire": "Uncertain",
        "research in progress": "Uncertain",
        "correct attribution": "Likely True",
        "misattributed": "Likely False"
    }
    df_snopes["verdict_mapped"] = df_snopes["verdict_original"].str.lower().map(mapping_snopes)

    df_snopes = df_snopes[
        ["claim_id", "claim_text", "verdict_original", "verdict_mapped",
         "summary", "originator", "url", "dataset_source"]
    ]

    # Merge and Save Dataset
    df_merged = pd.concat([df_politifact, df_snopes], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f" Unified dataset saved to: {output_path}")
    print(f"Total records: {len(df_merged)}")
    print(df_merged['verdict_mapped'].value_counts(dropna=False))

    return df_merged
