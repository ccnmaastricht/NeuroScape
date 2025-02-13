# NeuroScape: Mapping the Neuroscience Research Landscape

A comprehensive pipeline for analyzing and visualizing neuroscience research through large-scale analysis of scientific abstracts, citation networks, and semantic clustering.

## Overview

NeuroScape processes scientific abstracts from PubMed to:
- Identify research clusters and communities
- Extract key research dimensions and trends
- Analyze citation patterns and network density
- Surface open research questions
- Map the evolution of research topics

## Project Structure



### Workflow

1. scraping
2. merge_and_clean
3. inital embedding (using Voyage AI)
4. prepare training data for discipline classifier
5. train discipline classifier
6. filter data using discipline classifier
7. build adjacency matrix
8. community detection
9. cluster definition
10. cluster distinction
11. density graph
12. dimensions extraction
13. dimension categorization
14. open questions
15. trends extraction

Optional is the possibility to update general embeddings using a different Voyage AI model using the update_embedding script



## Installation

1. Clone the repository
2. Create a `.env` file:

3. Install dependencies:
```bash
pip install -r requirements.txt


# 1. Data ingestion
python scripts/ingestion/scrape_pubmed.py
python scripts/ingestion/merge_and_clean.py

# 2. Generate embeddings and train classifier
python scripts/preprocessing/train_discipline_classifier.py
python scripts/preprocessing/filter_disciplines.py

# 3. Run analysis
python scripts/clustering/community_detection.py
python scripts/semantic_analysis/cluster_definition.py
python scripts/semantic_analysis/extract_open_questions.py