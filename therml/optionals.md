In our study, we compared performances of different representations. <br>

- Specifically, we focused on: Magpie, [Roost](https://github.com/CompRhys/roost) and [CrabNet](https://github.com/anthony-wang/CrabNet). 
- We generate learned embeddings for a material composition by extracting features from trained Roost and CrabNet models.
- The models we used for this purpose are provided in this repository within ```saved_models/transfer_learning``` directory.
- You will need to move the checkpoint corresponding to Roost inside the Roost directory as: ```roost/roost/models/oqmd_100_epochs_model/checkpoint-r0.pth.tar```
- Move the checkpoint for CrabNet in the CrabNet directory as: ```CrabNet/models/trained_models/aflow__agl_thermal_conductivity_300K.pth```
- This will enable you to generate features using Roost and CrabNet.
- For generating features with CrabNet, you will need to inherit the class and remove the final layers. Roost provides a direct utility for performing this. 
