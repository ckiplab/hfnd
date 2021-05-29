# H-FND: Hierarchical False-Negative Denoising for Distant Supervision Relation Extraction

This repository holds the code for the paper: H-FND: Hierarchical False-Negative Denoising for Distant Supervision Relation Extraction, to appear in the findings of ACL-2021.

## Usage

### Data Preparation

The preprocessing of human-annotated datasets (SemEval, TACRED) and distantly supervised dataset (NYT10) are all handled in `preprocess/preprocessor.py`

Each class handles:
1. The parsing of raw dataset inputs
2. The generation of synthetic noise (for human-annotated datasets.)
3. Converting the dataset to RelationDataset compatable formats.

Modify the paths that points to the raw data files in the code to fit your file locations.

### The file structures

Each file contains the model definition, trainer function and the run function.

- In `base/` resides the base CNN/PCNN models and their training process.

- `denoise/` contains code of H-FND, as well as other denoising baselines:

    - `denoise/rl.py`: H-FND
    - `denoise/cleanlab.py`: Cleanlab
    - `denoise/coteaching.py`: Co-teaching

- `util/` contains the utiliy functions that are shared among models.
    
    - `data.py` contains the definition of class `RelationDatast`.
    - `embedding.py` contains the wrapper functions to retrieve Spacy embeddings.
    - `measure.py` defines the mircoF1 score and the accuracy measurements.
    - `tokenizer.py` handles the tokenization process that was called in the construction of `RelationDatasets`

### To Run the Codes

To run the codes, first prepare the raw dataset. Then, update the paths in the code to match the file locations.

Then change the hyperparameter and path in each `train_{dataset_name}_()` functions, then call the module using `python3 -m {path_to_.py_file}`