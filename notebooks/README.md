# Notebooks

Analysis notebooks accompanying the platform:

| Notebook | Purpose |
| --- | --- |
| `legal_bert_finetuning.ipynb` | Fine-tune Legal-BERT on the CUAD dataset for the 15-category risk classifier and report per-category F1 (matching the benchmark table in the root README). |
| `ragas_evaluation.ipynb` | Evaluate RAG retrieval faithfulness / answer relevancy over the CUAD test split with the RAGAS framework. |
| `risk_analysis.ipynb` | Exploratory analysis of clause risk distribution, monetary exposure, and Neo4j obligation-chain exploration. |

To run them:

```bash
venv/Scripts/pip install -r requirements-cloud.txt jupyter
jupyter lab
```

The classifier and pipeline used in these notebooks are the same modules under
[`src/`](../src) — import them directly, e.g.
`from src.pipeline import ContractIntelligencePipeline`.
