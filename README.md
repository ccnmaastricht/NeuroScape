# NeuroScape: Mapping the Neuroscience Research Landscape

Codebase accompanying the article:

Senden, M. (2025). The Evolving Landscape of Neuroscience [Preprint]. bioRxiv. [https://doi.org/10.1101/2025.02.13.638094](https://doi.org/10.1101/2025.02.13.638094)

This repository contains scripts and notebooks for analyzing and visualizing neuroscience research through large-scale data collection, filtering, clustering, and semantic analysis of scientific articles.

## Abstract
Neuroscience emerged as a distinct academic discipline during the 20th century and has undergone rapid expansion and diversification. A comprehensive analysis of its evolving landscape becomes increasingly important to retain an overview of cross-domain insights and research questions. This study leverages text-embedding and clustering techniques together with large language models to analyze 461,316 articles published between 1999 and 2023 and reveals the field's structural organization and dominant research domains. Inter-cluster citation analysis uncovers a surprisingly integrated picture and key intellectual hubs that shape the broader research landscape. An analysis of how research clusters align with pre-defined dimensions demonstrates a strong experimental focus, widespread reliance on specific mechanistic explanations rather than unifying theoretical frameworks, and a growing emphasis on applied research. Fundamental research is at the risk of decline and cross-scale integration remains limited. This study provides a framework for understanding neuroscience's trajectory and identifies potential avenues for strengthening the field.

## Repository Structure
```
.
├── config/                 # TOML config files (scraping, clustering, analysis, etc.)
├── notebooks/              # Jupyter notebooks for exploration and results
├── scripts/                # Main processing scripts organized by function
│   ├── ingestion/          # Data collection and cleaning
│   ├── preprocessing/      # Filtering and classification
│   ├── domain_embedding/   # Training & applying domain-specific embeddings
│   ├── clustering/         # Building semantic graphs & community detection
│   ├── graph_analysis/     # Citation/network density analysis
│   └── semantic_analysis/  # Dimension analysis, cluster characterization, trends, etc.
├── src/
│   ├── classes/            # Python classes for data structures and model architectures
│   └── utils/              # Utility modules (parsing, data loading, plotting, etc.)
└── README.md
```

## Data

This repository has been used to collect, curate, and analyze the following dataset:

Senden, M. (2025). NeuroScape (1.0.1) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.14865161](https://doi.org/10.5281/zenodo.14865161)

## Workflow Overview

1. **Scrape Data**
   - *scripts/ingestion/scraping.py*  
   - Query PubMed for relevant neuroscience articles.

3. **Merge and Clean**  
   - *scripts/ingestion/merge_and_clean.py*  
   - Consolidate scraped data, remove duplicates, and clean metadata.

4. **Initial Embedding**  
   - *scripts/ingestion/initial_embedding.py*  
   - Generate general-purpose text embeddings (via Voyage AI) for each abstract.

5. **Prepare Classifier Training Data**  
   - *scripts/preprocessing/prepare_classifier_training_data.py*  
   - Create labeled samples for discipline classification (to distinguish neuroscience from other fields).

6. **Train Discipline Classifier**  
   - *scripts/preprocessing/train_discipline_classifier.py*  
   - Train a neural network to identify neuroscience-related articles.

7. **Filter Data**  
   - *scripts/preprocessing/filter_disciplines.py*  
   - Retain only articles classified as neuroscience with high confidence.

8. **Build Adjacency Matrix**  
   - *scripts/ingestion/build_adjacencies.py*  
   - Obtains citing and cited articles for each article in the dataset.

9. **Train Domain Embedding Model**
    - *scripts/domain_embedding/train_embedding_model.py*
    - Trains a domain-specific embedding model on top of the initial embeddings (through contrastive learning).

10. **Domain Embedding**
    - *scripts/domain_embedding/embed_abstracts.py*
    - Re-embeds abstracts in a lower-dimensional, neuroscience-focused space for semantic clustering.

11. **Build the Semantic Graph**
    - *scripts/clustering/graph_construction.py*
    - Uses the domain-specific embeddings to construct a similarity graph (e.g., KNN) needed for community detection.

12. **Community Detection**  
    - *scripts/clustering/community_detection.py*  
    - Perform clustering (e.g., Leiden community detection) on the network.

12. **Cluster Definition**  
    - *scripts/semantic_analysis/cluster_definition.py*  
    - Generate descriptive titles, keywords, and summaries for each cluster.

13. **Cluster Distinction**  
    - *scripts/semantic_analysis/cluster_distinction.py*  
    - Identify key differences between similar clusters.

14. **Dimensions Extraction**  
    - *scripts/semantic_analysis/assess_dimensions.py*  
    - Analyze each cluster across multiple research dimensions (e.g., appliedness, modality).

15. **Dimension Categorization**  
    - *scripts/semantic_analysis/assess_dimension_categories.py*  
    - Categorize clusters along specific sub-dimensions (e.g., spatial vs. temporal scales).

16. **Open Questions**  
    - *scripts/semantic_analysis/extract_open_questions.py*  
    - Identify important open research questions from recent review articles.

17. **Trends Extraction**  
    - *scripts/semantic_analysis/extract_trends.py*  
    - Compare older vs. recent publications to reveal emerging and declining trends.

18. **Density Graph**  
    - *scripts/graph_analysis/cluster_density.py*  
    - Assess citation density and connections between clusters.

**Optional** 
- *scripts/preprocessing/update_embedding.py* allows updating the general embeddings with a newer Voyage AI model if desired.


## **Setup Instructions**

### **1. Create a Conda Environment**
First, create and activate a Conda environment with Python 3.12:
```bash
conda create --name neuroscape_env python==3.12
conda activate neuroscape_env
```

### **2. Install PyTorch**
Install PyTorch **before** installing other dependencies. Follow the official instructions based on your system:
- **Visit**: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)


### **3. Install Other Dependencies**
Once PyTorch is installed, install the remaining dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Disclaimer

This repository provides the scripts and workflow used in *The Evolving Landscape of Neuroscience* study and is intended for research and educational purposes. While I encourage other researchers to use and build upon this work, **I am primarily a researcher, not a full-time software developer**. As such:

- I **welcome issues and pull requests** and will try to address them as time permits.
- However, **active maintenance is not guaranteed**. Users should **not** expect frequent updates or extensive support.
- The code is provided **as is**, without warranties regarding performance, correctness, or long-term compatibility.

If you use this repository for your research, I would appreciate a citation to the accompanying manuscript.
