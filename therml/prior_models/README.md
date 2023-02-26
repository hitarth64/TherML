This directory contains the six **prior-models** that we have used in our study.
A brief description is provided below for all of them. For more information, refer to the SI.

All these models have been trained using the [MRL dataset](http://www.mrl.ucsb.edu:8080/datamine/thermoelectric.jsp) with 5-fold cross validation to optimize the hyperparameters.
We used the data as prepared by [Dopnet](https://github.com/ngs00/DopNet) referred to as mrl.xlsx on their github repository. 

|File |Description|
|-----|--------|
| magpie_rfr_model | Random forest trained using Magpie representation |
| model_magpie_tpot  | TPOT optimized pipeline trained using Magpie representation |
| model_crab_rfr | Random forest trained using Crabnet transfer learned representation  |
| model_crabnet_tpot  | TPOT optimized pipeline trained using Crabnet transfer learned representation |
| model_roost_rfr | Random forest trained using Roost transfer learned representation |
| model_roost_tpot2 | TPOT optimized pipeline trained using Roost transfer learning representation |

