
## Content details:

* We recommend you to first test if you can run the analysis with the default settings i.e., using Magpie and DNNs.
  * Test with ```python hpo_dense.py```
* If you are interested in using other featurizations (Roost/ CrabNet), please refer to [these instructions](optionals.md). 
* For Crabnet, you will need to inherit the KingCrab class and remove the dense layers to get an embedding.

Contents of this directory:

|Filename |Description|
|-----|--------|
|hpo_dense_magpie_tpot.py | Trains the model given the data |
|data.py | Handles all data related operations and featurizations |
|models.py| Definitions for our error-correction neural networks |
|inference.py| For performing inference on a fully trained model |
|featurized_experimental_data.pkl| Experimental data obtained as part of this study |
|prior_models | Directory containing all pre-trained models|
|utilities.py| Extra functions useful for model training|
|saved_models | Directory containing checkpoint of our best model and other transfer learning checkpoints|
|crabnet_feature_extraction.py| Function to extract features from CrabNet|
|roost_feature_extraction.py| Function to extract features from Roost |
