## SpectRAG (Spectral Graph-Augmented Retrieval for Diverse and Connected Context Selection in RAG)
This repository contains the official implementation and paper resources for Spectral Graph-Augmented Retrieval (SpectRAG), a retrieval framework that improves context selection in retrieval-augmented generation systems.

SpectRAG augments dense vector retrieval with a weighted chunk graph and spectral graph structure. Instead of retrieving only the top-ranked chunks according to semantic similarity, SpectRAG aims to select a context set that is jointly relevant, non-redundant, structurally connected, and sufficient for multi-hop reasoning.

## Method Components
SpectRAG contains the following core modules:
- Weighted chunk graph construction
- Laplacian spectral coordinate computation
- Query-aware graph expansion
- Spectral diffusion of relevance scores
- Cluster-aware context-set selection
- Retrieval-level and generation-level evaluation

## Citation
Citation information will be added after the arXiv preprint is released.

## License
This repository is intended for academic research and reproducibility. The final license will be specified before public release.