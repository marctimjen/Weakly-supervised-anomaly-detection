# Weakly supervised anomaly detection

> This is the github repo for the project "Weakly supervised anomaly detection" by: Drozdova Anastasia, Jensen Marc & Jørgensen Andreas.

Note that this repo contain some __init__.py files to make some directories into py packages. These files will be ignored since they do not contain any informaiton.

Note that there in some of the scripts will be a doc-string telling more about what the goal of the specific scripts is.


How to get started: 
1. Download or generate video features. In this repo features has been downloaded from the sites:
[xd-violence](https://roc-ng.github.io/XD-Violence/) and [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE).
Note that the xd-violence features comes crop-wise (of five crops). This means that one video exists as five .npy feature-files when downloaded. 
The networks implemented in this repo use a concatenated version of the feature files. To concatenate the files this script has been used: [crop_to_file.py](data%2Fxd_crop_to_file%2Fcrop_to_file.py).

2. The next step is to obtain the lists with paths to the individual files.

3. Generate or download the GT-file (ground truth) for the test set.

4. Train the models.

5. Test the models and upload data to Neptune.

The structure of the code is as follows:

[data](data): This dir is used to preprocess the data in different ways.
- [create_gt](data%2Fcreate_gt): This dir is specifically used to generate the ground truth files (gt) files.
  - [create_gt_xd.py](data%2Fcreate_gt%2Fcreate_gt_xd.py): Create the gt file for the xd-violence dataset.
  - [create_test_gt.py](data%2Fcreate_gt%2Fcreate_test_gt.py): This file creates the gt file for the UCF-crime dataset.
  - [create_UCF_masks.py](data%2Fcreate_gt%2Fcreate_UCF_masks.py): This script is used to generate masks for the different classes for the gt file.
- [create_val_split](data%2Fcreate_val_split): This dir is used to create a validation dataset for the UCF-data.
  - [create_val_split_UCF.py](data%2Fcreate_val_split%2Fcreate_val_split_UCF.py): Creates the val/train split of the training data.
  - [move_files_UCF.py](data%2Fcreate_val_split%2Fmove_files_UCF.py): This .py file is used to move the validation features from the train-dir to a validation-dir.
  - [test_UCF_gt.py](data%2Fcreate_val_split%2Ftest_UCF_gt.py): This script is used to test the gt-file of the UCF-dataset.
- [make_lists](data%2Fmake_lists): This dir is used to make lists that point to the different train/test files.
  - [make_lists_xd.py](data%2Fmake_lists%2Fmake_lists_xd.py): This script produce the RGB lists for the xd-violence dataset.
- [xd_crop_to_file](data%2Fxd_crop_to_file): The features downloaded from the page: [xd-violence](https://roc-ng.github.io/XD-Violence/) came crop-wise and not as a whole feature.
  - [crop_to_file.py](data%2Fxd_crop_to_file%2Fcrop_to_file.py): This file turns the individually crops into a total feature.

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


[RTFMmain](RTFMmain): This dir is used to implement the RTFM model.
- [cheat](RTFMmain%2Fcheat): This dir contain files for calculating the test-results of the train-epochs of the model.
  - [cheat_test_ucf.py](RTFMmain%2Fcheat%2Fcheat_test_ucf.py): This file is used for calculate the test/val loss of the RTFM model on the ucf dataset.
  - [cheat_test_xd.py](RTFMmain%2Fcheat%2Fcheat_test_xd.py): This script calculate the test/val loss for the RTFM model on the XD data.
- [UCF_test](RTFMmain%2FUCF_test): This dir is used to upload different results to the Neptune for the UCF data.

- [xd_test](RTFMmain%2Fxd_test): This dir is used to upload results from the RTFM model to Neptune.
  - [1_RTFM_xd_upload.py](RTFMmain%2Fxd_test%2F1_RTFM_xd_upload.py): This file is used to upload the train and test-loss of the model.
  - [2_RTFM_xd_test.py](RTFMmain%2Fxd_test%2F2_RTFM_xd_test.py): This script is used to upload the different results of the metrics.
  - [3_RTFM_xd_test_plotter.py](RTFMmain%2Fxd_test%2F3_RTFM_xd_test_plotter.py): This file makes plots of the predictions and a AUC + PR curve that is uploaded to Neptune.

- [dataset.py](RTFMmain%2Fdataset.py): This script contain the implementation of the pytorch dataset for both XD and UCF.
- [model.py](RTFMmain%2Fmodel.py): This file implements the RTFM model.
- [Model_params.txt](RTFMmain%2FModel_params.txt): This file contain a summary of the RTFM model - how many parameters it contains.
- [params.py](RTFMmain%2Fparams.py): This script contain the hyperparameter grid search and different validation parameters.
- [RFTM_cheat_ucf.py](RTFMmain%2FRFTM_cheat_ucf.py): Train script for the RTFM model 


- [train.py](RTFMmain%2Ftrain.py): This file contain the train/validation and loss functions for the training in the different scripts.


[process_master](process_master): This dir is used to run different processes - starting training/testing of networks.
- [delete_failed_data.py](process_master%2Fdelete_failed_data.py): This file deletes the testing results on neptune if the data-upload has failed.
- [process.txt](process_master%2Fprocess.txt): Contain some note on some runs that has succeeded.
- [process_0.txt](process_master%2Fprocess_0.txt)/[process_1.txt](process_master%2Fprocess_1.txt)/[process_2.txt](process_master%2Fprocess_2.txt): Is used to define which runs to start on the cluster.
- [process_marc.txt](process_master%2Fprocess_marc.txt): Is used for running code locally.
- [process_master.py](process_master%2Fprocess_master.py): This scripts starts the code in chronological order. The process_master waits for the current process to finish before starting a new process.

[result_uploader](result_uploader): This dir is used to upload results of the different networks to neptune. This only counts for the models used on the UCF-dataset.
- [nept_plotter.py](result_uploader%2Fnept_plotter.py): This script plots a histogram for each class in the UCF-data for the models specified.
- [rando.npy](result_uploader%2Frando.npy): This file contain the results of a random model.
- [random_predictor.py](result_uploader%2Frandom_predictor.py): Is used to make the [rando.npy](result_uploader%2Frando.npy) file.
- [res_no_file_upload_UCF.py](result_uploader%2Fres_no_file_upload_UCF.py): This file is used to upload data from cpkt_models that is not available to us.
- [res_uploader_UCF.py](result_uploader%2Fres_uploader_UCF.py): This file use prediction file from the different networks tested and upload the result data to neptune.

[test](test): This dir contain old files that has been used for testing/production of code.


How to upload data for the MGFN model to neptune?

Firstly we run the [MGFN_UCF_upload.py](MGFNmain%2FUCF_pretrained_test%2FMGFN_UCF_upload.py) file to get a neptuen run + the loss of the model at the current iteration. Remember to set the path to the correct model and set the parameters for this model correctly.
Then we run the [MGFN_UCF_test.py](MGFNmain%2FUCF_pretrained_test%2FMGFN_UCF_test.py) file to get the results of the model. These results are also uploaded to neptune (remember to set the -n for neptune run argument).
Lastly we can upload the plots using the test-file create before and the script: [MGFN_test_plotter.py](MGFNmain%2FUCF_pretrained_test%2FMGFN_test_plotter.py).