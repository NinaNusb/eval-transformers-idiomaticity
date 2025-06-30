# MWE representation in transformers
Exploring how Transformer models encode Multiword Expressions (MWEs), particularly those expressions having both a compositional and an idiomatic meaning (e.g. "silver spoon" can refer to the cutlery or the privilege).

This repo contains the data and code for the paper Nusbaumer, Wisniewski, Crabbé, 2025. Évaluer la capacité des transformeurs à distinguer les significations compositionnelles et idiomatiques d’une même expression [accepted at TALN 2025].

## Organization
In this repo you will find:

- **data/** : the English section of the dataset curated by ```TAYYAR MADABUSHI H., GOW-SMITH E., SCARTON C. & VILLAVICENCIO A. (2021). AStitchInLanguageModels : Dataset and Methods for the Exploration of Idiomaticity in Pre-Trained Language Models. In M.-F. MOENS, X. HUANG, L. SPECIA & S. W.-T. YIH, Éds., Findings of the Association for Computational Linguistics : EMNLP 2021, p. 3464–3477, Punta Cana, Dominican Republic : Association for Computational Linguistics. DOI : 10.18653/v1/2021.findings-emnlp.294.```
- **BERT-base/**: the notebooks for the main experiments as well as other tests
- **Other models/**: the replication of the main experiences for BERT-large, RoBERTa, and MultiBERT
- **utils/**: the functions used for the analysis

## Citation
If you use this code or data, please cite:
```
@inproceedings{nusbaumer2025eval,
  title={Évaluer la capacité des transformeurs à distinguer les significations compositionnelles et idiomatiques d’une même expression},
  author={Nusbaumer, Wisniewski, Crabbé},
  year={2025},
  booktitle={TALN 2025}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
