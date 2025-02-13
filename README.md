# NeuroScape: Mapping the Neuroscience Research Landscape

Codebase accompanying the article:

Senden, M. (n.d.). The Evolving Landscape of Neuroscience. *Manuscript submitted*.

This repository contains scripts and notebooks for analyzing and visualizing neuroscience research through large-scale data collection, filtering, clustering, and semantic analysis of scientific articles.

## Abstract
Neuroscience emerged as a distinct academic discipline during the 20th century and has undergone rapid expansion since then. However, a comprehensive analysis of its evolving landscape remains lacking. The present study provides a large-scale, data-driven, mapping of neuroscience research. Leveraging large language models and clustering techniques to analyze 461,316 articles published between 1999 and 2023, this study reveals the field's structural organization and highlights dominant themes. Citation network analysis uncovers a surprisingly integrated picture and key intellectual hubs that shape the broader research landscape. An analysis of how research clusters align with pre-defined dimensions demonstrates a strong experimental focus, widespread reliance on micro theories, and a growing emphasis on applied and translational research.  At the same time, fundamental research is at the risk of decline while theoretical work and cross-scale integration remain limited. This study provides a framework for understanding neuroscience's trajectory and identifies potential avenues for strengthening the field.

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

## Workflow Overview

1. **Scrape Data**  
   - *scripts/ingestion/scraping.py*  
   - Query PubMed for relevant neuroscience articles.

2. **Merge and Clean**  
   - *scripts/ingestion/merge_and_clean.py*  
   - Consolidate scraped data, remove duplicates, and clean metadata.

3. **Initial Embedding**  
   - *scripts/ingestion/initial_embedding.py*  
   - Generate general-purpose text embeddings (via Voyage AI) for each abstract.

4. **Prepare Classifier Training Data**  
   - *scripts/preprocessing/prepare_classifier_training_data.py*  
   - Create labeled samples for discipline classification (to distinguish neuroscience from other fields).

5. **Train Discipline Classifier**  
   - *scripts/preprocessing/train_discipline_classifier.py*  
   - Train a neural network to identify neuroscience-related articles.

6. **Filter Data**  
   - *scripts/preprocessing/filter_disciplines.py*  
   - Retain only articles classified as neuroscience with high confidence.

7. **Build Adjacency Matrix**  
   - *scripts/ingestion/build_adjacencies.py*  
   - Construct citation or similarity networks among the retained articles.

8. **Train Domain Embedding Model**
    - *scripts/domain_embedding/train_embedding_model.py*
    - Trains a domain-specific embedding model on top of the initial embeddings (contrastive learning)

9. **Domain Embedding**
    - *scripts/domain_embedding/embed_abstracts.py*
    - Re-embeds abstracts in a lower-dimensional, neuroscience-focused space for semantic clustering.

10. **Build the Semantic Graph**
    - *scripts/clustering/graph_construction.py*
    - Uses the domain-specific embeddings to construct a similarity graph (e.g., KNN) needed for community detection.

11. **Community Detection**  
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
