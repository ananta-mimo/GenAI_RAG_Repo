# Local RAG Pipeline: Chat with ML/Credit-Risk Papers

A fully local Retrieval-Augmented Generation (RAG) system that answers questions
about a small corpus of open-access research papers (SHAP, XGBoost, and
explainable AI in credit risk), with no cloud APIs, no subscriptions, and no
internet dependency at inference time.

Built as a portfolio project to demonstrate an understanding of the full RAG
stack: chunking, embeddings, vector search, generation, and evaluation, running
entirely on a laptop with no dedicated GPU.

## Why this project

Most public RAG tutorials wrap an OpenAI API call around a vector store and
call it done. This project intentionally avoids any hosted LLM: every
component, embedding, retrieval, and generation, runs locally on CPU. The goal
was to understand each piece well enough to build it by hand before reaching
for a framework like LangChain, and to build a lightweight evaluation process
rather than stopping at "it runs."

## Architecture

```
PDFs (data/)
   |
   v
Text extraction (pypdf)
   |
   v
Chunking (~500 chars, 50-char overlap)
   |
   v
Embedding (sentence-transformers: all-MiniLM-L6-v2)
   |
   v
Vector index (FAISS, IndexFlatL2)
   |
   v
Query embedding  ---->  similarity search  ---->  top-k chunks retrieved
                                                          |
                                                          v
                                          Prompt assembly (context + question)
                                                          |
                                                          v
                                          Local LLM generation (Phi-3-mini via Ollama)
                                                          |
                                                          v
                                          Answer + faithfulness score
```

## Stack

| Component | Tool | Notes |
|---|---|---|
| PDF parsing | `pypdf` | Extracts raw text per page |
| Chunking | Custom function | Fixed-size character chunks with overlap |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384-dim vectors, CPU-only |
| Vector store | `faiss-cpu` (`IndexFlatL2`) | Exact nearest-neighbor search |
| LLM | Phi-3-mini via Ollama | Runs fully offline, no API key |
| Evaluation | Custom word-overlap faithfulness score | Heuristic, not a learned metric |

**Hardware used:** Dell Inspiron 5406, Intel Core i5-1135G7, 16GB RAM, Intel
Iris Xe integrated graphics (no dedicated GPU). All inference runs on CPU.

## Setup

```bash
git clone <this-repo-url>
cd GenAI_RAG_Repo
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install sentence-transformers faiss-cpu ollama pypdf
```

Install [Ollama](https://ollama.com/download), then pull the model:

```bash
ollama pull phi3
```

Place source PDFs in a `data/` folder, then run:

```bash
python minimal_local_rag.py
```

## Corpus

Three open-access arXiv papers, chosen to overlap with a companion credit-risk
modeling project in this portfolio:

- Lundberg & Lee, *Consistent feature attribution for tree ensembles* (the
  original SHAP paper)
- Hadji Misheva et al., *Explainable AI in Credit Risk Management* (SHAP/LIME
  applied to the Lending Club dataset)
- Mayer, *SHAP for additively modeled features in a boosted trees model*

221 chunks were produced from these three PDFs at a 500-character chunk size
with 50-character overlap.

## Evaluation

Five questions were run against the pipeline, each scored with a custom
word-overlap faithfulness metric (fraction of the generated answer's key words
found in the retrieved context, after lightweight stemming to reduce false
mismatches from plurals/verb tense).

| Question | Faithfulness |
|---|---|
| How is SHAP used to explain credit risk model predictions? | 0.51–0.70 |
| What is the difference between interpretability and explainability? | 0.66–0.76 |
| How does SHAP relate to cooperative game theory? | 0.50–0.62 |
| What is a SHAP dependence plot used for? | 0.70 |
| How does LIME differ from SHAP? | 0.45–0.61 |

Scores vary slightly across runs due to the LLM's non-deterministic sampling.

### Faithfulness metric: known limitations

This is a lexical heuristic, not a semantic one, and it has two systematic
blind spots, confirmed while building it:

- **Paraphrasing is penalized as "unsupported."** Synonyms and reworded
  phrasing lower the score even when the meaning is fully grounded in the
  source.
- **PDF extraction artifacts suppress real matches.** Two-column academic PDF
  layouts introduce broken words (e.g. `de-pendence` split across a line
  break, or `featurej` from a glued citation marker), which the regex-based
  tokenizer treats as different words entirely.

A stemming step was added to reduce false negatives from plurals and verb
tense (e.g. `decompose`/`decomposing`), which raised scores meaningfully. An
earlier hand-rolled stemmer over-stemmed some words inconsistently
(`decomposing` → `decompos` while `decompose` stayed unstemmed); this is
documented as a known NLP failure mode, over-stemming, and is fixable by
switching to a tested implementation like `nltk`'s Porter Stemmer.

### A real generation error the eval set caught

For "How does LIME differ from SHAP?", retrieval pulled the correct passage,
but the generated answer swapped which method a stated limitation belongs to:
the source text attributes exponential runtime to SHAP (from computing exact
Shapley values over feature subsets) and the probabilistic-models-only
constraint to LIME. The model's answer reversed both attributions.

This is a generation-side error, not a retrieval failure: the right facts were
extracted, but bound to the wrong entity. The source passage uses ambiguous
pronoun references ("it") across sentences, which likely contributed. This is
a known weakness class for small local models like Phi-3-mini, and a good
argument for either a larger generation model or an added verification step
(e.g. asking the model to quote the exact source sentence for each claim) in
a production version of this pipeline.

## Limitations

- CPU-only inference is noticeably slower than a hosted API (several seconds
  per answer).
- The faithfulness metric is a lexical heuristic; it does not verify factual
  correctness or catch attribution errors like the one described above.
- Chunking is fixed-size by character count, not sentence- or
  paragraph-aware, so chunks can cut mid-sentence.
- Small corpus (3 papers, 221 chunks); not tested at larger scale.

## Possible next steps

- Swap the hand-rolled stemmer for `nltk`'s Porter Stemmer.
- Add sentence-aware chunking instead of fixed character windows.
- Add a verification step where the model must cite the exact source sentence
  for each claim, to catch attribution errors like the LIME/SHAP swap.
- Expand the eval set with a few "trick" questions with no answer in the
  corpus, to test whether the system correctly declines rather than
  hallucinating.

## Repo structure

```
GenAI_RAG_Repo/
├── data/                     # source PDFs (not committed if large)
├── minimal_local_rag.py      # full pipeline: chunking, embedding, retrieval, generation, eval
├── README.md
└── requirements.txt
```
