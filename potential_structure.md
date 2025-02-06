.                                  # Root of the repository
├── .env                           # Environment variables (e.g., BASEPATH=/path/to/external/data)
├── .gitignore                     # Excludes large external data directories, caches, etc.
├── LICENSE
├── README.md
├── requirements.txt               # Project dependencies

├── config/                        # All configuration files (TOML)
│   ├── base_directories.toml      # Base directories relative to BASEPATH (e.g., raw, cleaned, embeddings, graphs, models)
│   ├── plotting.toml   # Settings for figures and visualization
│   ├── ingestion/                 # Configurations for data scraping/ingestion
│   │   ├── scraping.toml
│   │   └── cleaning.toml
│   ├── models/                    # Configurations for model training and parameters
│   │   ├── discpline_classification.toml
│   │   ├── domain_embedding.toml
│   ├── clustering/                    # Settings for graph construction & community detection (clustering of abstracts)
│   │   ├── graph_construction.toml
│   │   └── community_detection.toml
│   ├── analysis/                  # Configurations for downstream cluster analysis
│       ├── cluster_definition.toml
│       ├── cluster_density_analysis.toml
│       ├── cluster_dim_categorical.toml
│       └── cluster_selection.toml
├── notebooks/                     # Jupyter notebooks for exploration, experiments, and visualization
│   ├── 00_ingestion.ipynb         # Data scraping and cleaning exploration
│   ├── 01_preprocessing.ipynb     # Data filtering and training data preparation
│   ├── 02_embedding.ipynb         # Embedding extraction and transformation
│   ├── 03_clustering.ipynb        # Graph construction and community detection
│   ├── 04_analysis.ipynb          # Cluster analysis (density, distinctions, etc.)
│   └── 05_visualization.ipynb     # Figures and visual results
├── scripts/                       # Executable scripts for the end-to-end pipeline
│   ├── ingestion/                 # Data acquisition: scraping and merging
│   │   ├── scrape_pubmed.py       # Scraping abstracts from PubMed
│   │   └── merge_and_clean.py     # Merging raw CSVs and cleaning the data
│       └── build_adjacencies.py   # Constructs citation networks from PubMed and CrossRef data (needs to occur after discipline filtering)
│   ├── preprocessing/             # Filtering and preparing datasets
│   │   ├── filter_disciplines.py  # Applies the discipline classifier to remove non-neuroscience articles
│   │   └── prepare_training_data.py  # Generates labeled training sets from cleaned data
│   ├── modeling/                  # Training and evaluation of models
│   │   ├── train_classifier.py    # Trains the discipline classifier
│   │   ├── train_embedding_model.py  # Trains the sparse embedding model using contrastive learning
│   │   └── evaluate_models.py     # Evaluates model performance and outputs reports
│   ├── embedding/                 # Embedding extraction and transformation
│   │   ├── extract_embeddings.py  # Converts abstracts to general-purpose embeddings (via VoyageAI)
│   │   └── apply_embedding_model.py  # Transforms embeddings to neuroscience-specific space
│   ├── clustering/                # Graph-based clustering and community detection
│   │   ├── construct_graph.py     # Builds the semantic similarity (k-NN) graph using FAISS
│   │   └── detect_communities.py  # Runs the Leiden algorithm for community detection
│   ├── analysis/                  # Downstream analysis on clusters
│   │   ├── define_clusters.py     # Uses GPT to generate cluster descriptions/labels
│   │   ├── analyze_density.py     # Analyzes citation networks and computes density metrics
│   │   ├── distinguish_clusters.py  # Highlights key differences between similar clusters
│   │   └── extract_open_questions.py  # Extracts open research questions from review articles


├── src/                           # Shared library code, organized into subpackages
│   ├── __init__.py
│   ├── data/                      # Modules for data handling and I/O
│   │   ├── data_types.py         # Data structures for articles, embeddings, etc.
│   │   └── load_and_save.py      # Functions for HDF5 storage and CSV I/O
│   ├── models/                    # Neural network models and loss functions
│   │   ├── classifier.py         # Discipline classifier model implementation
│   │   ├── sparse_embedding_network.py  # Model for embedding transformation
│   │   ├── info_nce_loss.py      # Implementation of the InfoNCE loss function
│   │   └── npair_loss.py         # Implementation of the N-pair loss
│   ├── embedding/                 # Code for embedding operations
│   │   └── embedder.py           # Interface for applying the trained sparse embedding model
│   └── utils/                     # General utility modules (cleaning, scraping, plotting, etc.)
│       ├── cleaning.py
│       ├── scraping.py
│       ├── general.py
│       ├── grokfast.py
│       ├── plotting.py
│       └── analysis.py

