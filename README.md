# Weakly supervised anomaly detection
 This is the github repo for the project "Weakly supervised anomaly detection" by: Drozdova Anastasia, Jensen Marc & Jørgensen Andreas.

Note that this repo contain some __init__.py files to make some directories into py packages. These files will be ignored since they do not contain any informaiton.

The structure of the code is as follows:

\Data

[download_features](download_features): This dir is used to obtain the MGFN-features from the link: [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE)
- [auto_download.py](download_features%2Fauto_download.py): This file is used to download the features one at a time.

[MGFNmain](MGFNmain): This dir is used to implement the MGFN model.
- [cheat](MGFNmain%2Fcheat): This dir contain files for calculating the test-results of the train-epochs of the model.
  - [cheat_test.py](MGFNmain%2Fcheat%2Fcheat_test.py): Is used to test/validate the MGFN model on the UCF train set.
  - [cheat_test_xd.py](MGFNmain%2Fcheat%2Fcheat_test_xd.py): Is used to test/validate the MGFN model on the XD train set.
- [datasets](MGFNmain%2Fdatasets) Contain the datasets used in the training/testing of the MGFN model.
  - [dataset.py](MGFNmain%2Fdatasets%2Fdataset.py): This is the specific data-set used. It can load both UCF and XD data.
- [models](MGFNmain%2Fmodels): Contain the implementation of the MGFN model.
  - [mgfn.py](MGFNmain%2Fmodels%2Fmgfn.py): The specific file that contain the code for the MGFN model.
- [utils](MGFNmain%2Futils): Contain some utility functions used in the MGFN network. 

- [MGFN_cheat.py](MGFNmain%2FMGFN_cheat.py): Is the training of the MGFN model on the UCF data using the testing on each epoch.
- [MGFN_cheat_xd.py](MGFNmain%2FMGFN_cheat_xd.py): Is the training of the MGFN model on the XD data using the testing on each epoch.

- [params.py](MGFNmain%2Fparams.py): This file is used to set the hyper-parameteres of the model.
- [train.py](MGFNmain%2Ftrain.py): This file contain the loss functions, train and validation functions for the training for the network.
- 