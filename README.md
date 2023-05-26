# Weakly supervised anomaly detection

> This is the github repo for the project "Weakly supervised anomaly detection" by: Anastasia Drozdova, Andreas Jørgensen & Marc Jensen.

Note that this repo contain some __init__.py files to make some directories into py packages. These files will be ignored since they do not contain any informaiton.

Note that there in some of the scripts will be a doc-string telling more about what the goal of the specific scripts is.


How to get started:
1. Download or generate video features. In this repo features has been downloaded from the sites:
[xd-violence](https://roc-ng.github.io/XD-Violence/) and [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE).
Note that the xd-violence features comes crop-wise (of five crops). This means that one video exists as five .npy feature-files.
The networks implemented in this repo use a concatenated version of the feature files. To concatenate the XD files this script has been used: [crop_to_file.py](data%2Fxd_crop_to_file%2Fcrop_to_file.py).

2. The next step is to obtain the lists with paths to the individual files. These lists are known as "rgb_lists". For the XD-violence data-set the list has been generated
with this file: [make_lists_xd.py](data%2Fmake_lists%2Fmake_lists_xd.py).

3. (Optinal) To generate a validation data-set with associated rgb_lists (one for the validation set and one for the train set) use the dir: [create_val_split](data%2Fcreate_val_split).

4. Generate or download the gt-file (ground truth) for the test set. In this repo both the UCF and XD gt has been generated. Here the dir is used: [create_gt](data%2Fcreate_gt).
To use the functions in the dir, one needs the annotation files from the official data-provideres (they are available from the links to the data-sets in 1.).

5. Now the data should be setup for training and testing of the models. In the model-dirs different train-files should exist (see the contents-lists below). For instance to train
a MGFN model on the UCF-data (with a validation on the validation set) run the [MGFN_main_ucf.py](MGFNmain%2FMGFN_main_ucf.py). To run a model with validation on the test-set use the
files containing _cheat. For instance one can run: [MGFN_cheat_ucf.py](MGFNmain%2FMGFN_cheat_ucf.py) to train the MGFN model on the UCF-data with validation on the UCF-test set.

6. To test the models one can use the subdirectories: "xd_test" and "ucf_test". These calculate the loss on the train- and test-sets, result metrics and create plots of the predictions (and AUC + PR curves).

The structure of the code is as follows:

>[data](data): This dir is used to preprocess the data in different ways.
>- [create_gt](data%2Fcreate_gt): This dir is specifically used to generate the ground truth files (gt) files.
>  - [create_gt_xd.py](data%2Fcreate_gt%2Fcreate_gt_xd.py): Create the gt file for the xd-violence dataset.
>  - [create_test_gt.py](data%2Fcreate_gt%2Fcreate_test_gt.py): This file creates the gt file for the UCF-crime dataset.
>  - [create_UCF_masks.py](data%2Fcreate_gt%2Fcreate_UCF_masks.py): This script is used to generate masks for the different classes for the gt file.
>- [create_val_split](data%2Fcreate_val_split): This dir is used to create a validation split of the training data.
>  - [create_val_split_UCF.py](data%2Fcreate_val_split%2Fcreate_val_split_UCF.py): Creates the val/train split of the UCF-crime data.
>  - [create_val_split_XD.py](data%2Fcreate_val_split%2Fcreate_val_split_XD.py): Creates the val/train split of the XD-violence training data.
>  - [move_files_UCF.py](data%2Fcreate_val_split%2Fmove_files_UCF.py): This .py file is used to move the validation features from the train-dir to a validation-dir (for UCF-crime).
>  - [test_UCF_gt.py](data%2Fcreate_val_split%2Ftest_UCF_gt.py): This script is used to test the gt file of the UCF-dataset.
>  - [test_xd_gt.py](data%2Fcreate_val_split%2Ftest_xd_gt.py): This script is used to test the gt file of the XD-dataset.
>- [make_lists](data%2Fmake_lists): This dir is used to make lists that point to the different train/test files.
>  - [make_lists_xd.py](data%2Fmake_lists%2Fmake_lists_xd.py): This script produce the RGB lists for the xd-violence dataset.
>- [xd_crop_to_file](data%2Fxd_crop_to_file): The features downloaded from the page: [xd-violence](https://roc-ng.github.io/XD-Violence/) came crop-wise and not as a whole feature.
>  - [crop_to_file.py](data%2Fxd_crop_to_file%2Fcrop_to_file.py): This file turns the individually crops into a total feature.

> [download_features](download_features): This dir is used to obtain the MGFN-features from the link: [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE)
> - [auto_download.py](download_features%2Fauto_download.py): This file is used to download the features one at a time.

> [files](files): This dir contain different files. Such as annotation files and files with paths to train/test features.
> - [UCF_list](files%2FUCF_list): This sub dir is for the UCF-crime dataset.
>   - [Temporal_Anomaly_Annotation_for_Testing_Videos.txt](files%2FUCF_list%2FTemporal_Anomaly_Annotation_for_Testing_Videos.txt): This is the annotation file from the UCF-crime dataset.
>   - [ucf-i3d.list](files%2FUCF_list%2Fucf-i3d.list)/[ucf-i3d-train_cheat.list](files%2FUCF_list%2Fucf-i3d-train_cheat.list): These files contain all the train-files for the UCF-crime data-set.
>   - [ucf-i3d-test.list](files%2FUCF_list%2Fucf-i3d-test.list): This is all the test-files for the UCF-crime data-set.
>   - [ucf-i3d-train.list](files%2FUCF_list%2Fucf-i3d-train.list): This file contain the path to the train part of the train/val split of the UCF-crime data-set.
>   - [ucf-i3d-val.list](files%2FUCF_list%2Fucf-i3d-val.list): This is the validation set from the val-train split of the UCF-crime data.
> - [XD_list](files%2FXD_list): This dir contain the different files for training of the XD-violence data-set.
>   - [rgb.list](files%2FXD_list%2Frgb.list): This file contain all the paths to the XD-violence training features.
>   - [rgb_train.list](files%2FXD_list%2Frgb_train.list): This file contain paths to the train part of the train/val split of the XD-data.
>   - [rgb_val.list](files%2FXD_list%2Frgb_val.list): This file contain paths to the val part of the train/val split of the XD-data.
>   - [rgbtest.list](files%2FXD_list%2Frgbtest.list): This file contain the paths to the test-files of the XD-violence data.
>   - [xd.list](files%2FXD_list%2Fxd.list): This is the file with the gt frames of the test-data of XD-violence.

> [MGFNmain](MGFNmain): This dir is used to implement the MGFN model.
> -[cheat](MGFNmain%2Fcheat): This dir contain files for calculating the test-results of the train-epochs of the model.
>  - [cheat_test.py](MGFNmain%2Fcheat%2Fcheat_test.py): Is used to test/validate the MGFN model on the UCF train set.
>  - [cheat_test_xd.py](MGFNmain%2Fcheat%2Fcheat_test_xd.py): Is used to test/validate the MGFN model on the XD train set.
>- [datasets](MGFNmain%2Fdatasets) Contain the datasets used in the training/testing of the MGFN model.
>  - [dataset.py](MGFNmain%2Fdatasets%2Fdataset.py): This is the specific data-set used. It can load both UCF and XD data.
> - [models](MGFNmain%2Fmodels): Contain the implementation of the MGFN model.
>  - [mgfn.py](MGFNmain%2Fmodels%2Fmgfn.py): The specific file that contain the code for the MGFN model.
> - [test](MGFNmain%2Ftest): This dir is used to calculate some test-metrics from the prediction files.
>   - [test_files_ucf.py](MGFNmain%2Ftest%2Ftest_files_ucf.py): This file does testing of the models prediction (with and without thresholding) on the UCF-crime dataset.
>   - [total_test_ucf.py](MGFNmain%2Ftest%2Ftotal_test_ucf.py): Used to calculate the measures for different models on the UCF-crime.
>   - [total_test_xd.py](MGFNmain%2Ftest%2Ftotal_test_xd.py): Used to calculate the measures for different models on the XD-violence.
> - [UCF_test](MGFNmain%2FUCF_test): This dir is used to upload different results to the Neptune for the UCF data.
>  - [1_MGFN_UCF_upload.py](MGFNmain%2FUCF_test%2F1_MGFN_UCF_upload.py): This file is used to upload the loss of the train and test-set to Neptune.
>  - [2_MGFN_UCF_test.py](MGFNmain%2FUCF_test%2F2_MGFN_UCF_test.py): This script does the supervised test of the RTFM-model and upload the results to Neptune.
>  - [3_MGFN_test_plotter.py](MGFNmain%2FUCF_test%2F3_MGFN_test_plotter.py): This file create prediction plots (and AUC + PR plots) and does upload these to Neptune.
>  - : This file calculate the standard error of the loss and upload this to Neptune.
> - [utils](MGFNmain%2Futils): Contain some utility functions used in the MGFN network.
>  - [utils.py](MGFNmain%2Futils%2Futils.py): This is the script that contain the utility functions.
> - [xd_test](MGFNmain%2Fxd_test): This dir is used to upload results from the RTFM model to Neptune.
>  - [1_MGFN_xd_upload.py](MGFNmain%2Fxd_test%2F1_MGFN_xd_upload.py): This file is used to upload the train and test-loss of the model.
>  - [2_MGFN_xd_test.py](MGFNmain%2Fxd_test%2F2_MGFN_xd_test.py): This script does the supervised test of the RTFM-model and upload the results to Neptune.
>  - [3_MGFN_xd_test_plotter.py](MGFNmain%2Fxd_test%2F3_MGFN_xd_test_plotter.py): This file makes plots of the predictions (and AUC + PR plots) that is uploaded to Neptune.
>  - [4_MGFN_xd_losswidth.py](MGFNmain%2Fxd_test%2F4_MGFN_xd_losswidth.py): This file calculate the standard error of the loss and upload this to Neptune.
> - [config.py](MGFNmain%2Fconfig.py): Contain some configuration functions that is used while training the networks.
> - [MGFN_cheat_ucf.py](MGFNmain%2FMGFN_cheat_ucf.py): Is the training of the MGFN model on the UCF data using the testing on each epoch.
> - [MGFN_cheat_xd.py](MGFNmain%2FMGFN_cheat_xd.py): Is the training of the MGFN model on the XD data using the testing on each epoch.
> - [MGFN_main_ucf.py](MGFNmain%2FMGFN_main_ucf.py): This is the main script for training the MGFN model on UCF with validation on the validation set.
> - [MGFN_main_xd.py](MGFNmain%2FMGFN_main_xd.py): This is the main script for training the MGFN model on XD with validation on the validation set.
> - [params.py](MGFNmain%2Fparams.py): This file is used to set the hyper-parameteres of the model.
> - [test.py](MGFNmain%2Ftest.py): This is a test-script for testing and debugging the model locally (so no upload to Neptune).
> - [train.py](MGFNmain%2Ftrain.py): This file contain the loss functions, train and validation functions for the training for the network.

> [plotter](plotter): This dir creates different types of plots for testing purposes.
> - [ucf_test_plotter.py](plotter%2Fucf_test_plotter.py): This file creates prediction plots for the UCF-data.
> - [xd_test_plotter.py](plotter%2Fxd_test_plotter.py): This file creates prediction plots for the XD-data.

> [process_master](process_master): This dir is used to run different processes - starting training/testing of networks.
>- [delete_failed_data.py](process_master%2Fdelete_failed_data.py): This file deletes the testing results on neptune if the data-upload has failed.
>- [process.txt](process_master%2Fprocess.txt): Contain some note on some runs that has succeeded.
>- [process_0.txt](process_master%2Fprocess_0.txt)/[process_1.txt](process_master%2Fprocess_1.txt)/[process_2.txt](process_master%2Fprocess_2.txt): Is used to define which runs to start on the cluster.
>- [process_marc.txt](process_master%2Fprocess_marc.txt): Is used for running code locally.
>- [process_master.py](process_master%2Fprocess_master.py): This scripts starts the code in chronological order. The process_master waits for the current process to finish before starting a new process.

> [RTFMmain](RTFMmain): This dir is used to implement the RTFM model.
> - [cheat](RTFMmain%2Fcheat): This dir contain files for calculating the test-results (on the test-set) of the train-epochs of the model.
>  - [cheat_test_ucf.py](RTFMmain%2Fcheat%2Fcheat_test_ucf.py): This file is used for calculate the test/val loss of the RTFM model on the ucf dataset.
>  - [cheat_test_xd.py](RTFMmain%2Fcheat%2Fcheat_test_xd.py): This script calculate the test/val loss for the RTFM model on the XD data.
> - [UCF_test](RTFMmain%2FUCF_test): This dir is used to upload different results to the Neptune for the UCF data.
>  - [1_RTFM_UCF_upload.py](RTFMmain%2FUCF_test%2F1_RTFM_UCF_upload.py): This file is used to upload the loss of the train and test-set to Neptune.
>  - [2_RTFM_UCF_test.py](RTFMmain%2FUCF_test%2F2_RTFM_UCF_test.py): This script does the supervised test of the RTFM-model and upload the results to Neptune.
>  - [3_RTFM_test_plotter.py](RTFMmain%2FUCF_test%2F3_RTFM_test_plotter.py): This file create prediction plots (and AUC + PR plots) and does upload these to Neptune.
>  - [4_RTFM_UCF_losswidth.py](RTFMmain%2FUCF_test%2F4_RTFM_UCF_losswidth.py): This file calculate the standard error of the loss and upload this to Neptune.
>- [xd_test](RTFMmain%2Fxd_test): This dir is used to upload results from the RTFM model to Neptune.
>  - [1_RTFM_xd_upload.py](RTFMmain%2Fxd_test%2F1_RTFM_xd_upload.py): This file is used to upload the train and test-loss of the model.
>  - [2_RTFM_xd_test.py](RTFMmain%2Fxd_test%2F2_RTFM_xd_test.py): This script does the supervised test of the RTFM-model and upload the results to Neptune.
>  - [3_RTFM_xd_test_plotter.py](RTFMmain%2Fxd_test%2F3_RTFM_xd_test_plotter.py): This file makes plots of the predictions (and AUC + PR plots) that is uploaded to Neptune.
>  - [4_RTFM_xd_losswidth.py](RTFMmain%2Fxd_test%2F4_RTFM_xd_losswidth.py): This file calculate the standard error of the loss and upload this to Neptune.
> - [dataset.py](RTFMmain%2Fdataset.py): This script contain the implementation of the pytorch dataset for both XD and UCF.
> - [model.py](RTFMmain%2Fmodel.py): This file implements the RTFM model.
> - [Model_params.txt](RTFMmain%2FModel_params.txt): This file contain a summary of the RTFM model - how many parameters it contains.
> - [params.py](RTFMmain%2Fparams.py): This script contain the hyperparameter grid search and different validation parameters.
> - [RTFM_cheat_ucf.py](RTFMmain%2FRTFM_cheat_ucf.py): This is the main script for training the RTFM on UCF with validation on the test-set.
> - [RTFM_cheat_xd.py](RTFMmain%2FRTFM_cheat_xd.py): This is the main script for training the RTFM on XD with validation on the test-set.
> - [RTFM_main_ucf.py](RTFMmain%2FRTFM_main_ucf.py): This is the main script for training the RTFM model on UCF with validation on the validation set.
> - [RTFM_main_xd.py](RTFMmain%2FRTFM_main_xd.py): This is the main script for training the RTFM model on XD with validation on the validation set.
> - [test.py](RTFMmain%2Ftest.py): Is a test-script for testing and debugging the model locally (so no upload to Neptune).
> - [train.py](RTFMmain%2Ftrain.py): This file contain the train/validation and loss functions for the training in the different scripts.
> - [utils.py](RTFMmain%2Futils.py): This script contain some utility functions used for training the network.


> [result_uploader](result_uploader): This dir is used to upload results of the different networks to neptune. This only counts for the models used on the UCF-dataset.
> - [nept_plotter.py](result_uploader%2Fnept_plotter.py): This script plots a histogram for each class in the UCF-data for the models specified.
> - [rando.npy](result_uploader%2Frando.npy): This file contain the results of a random model.
> - [random_predictor.py](result_uploader%2Frandom_predictor.py): Is used to make the [rando.npy](result_uploader%2Frando.npy) file.
> - [res_no_file_upload_UCF.py](result_uploader%2Fres_no_file_upload_UCF.py): This file is used to upload data from cpkt_models that is not available to us.
> - [res_uploader_UCF.py](result_uploader%2Fres_uploader_UCF.py): This file use prediction file from the different networks tested and upload the result data to neptune.


> [S3R](S3R): This dir contains the implementation of the S3R model.
> - [anomaly](S3R%2Fanomaly)
>   - [apis](S3R%2Fanomaly%2Fapis)
>     - [comm.py](S3R%2Fanomaly%2Fapis%2Fcomm.py) This file contains primitives for multi-gpu communication for possibility of distributed training.
>     - [logger.py](S3R%2Fanomaly%2Fapis%2Flogger.py) Set up logger.
>     - [opts.py](S3R%2Fanomaly%2Fapis%2Fopts.py) Defines training parameters.
>     - [utils.py](S3R%2Fanomaly%2Fapis%2Futils.py) Some utility functions.
>   - [datasets](S3R%2Fanomaly%2Fdatasets)
>     - [video_dataset.py](S3R%2Fanomaly%2Fdatasets%2Fvideo_dataset.py) Contain the datasets class used in the training/testing of the S3R model.
>   - [engine](S3R%2Fanomaly%2Fengine)
>     - [inference.py](S3R%2Fanomaly%2Fengine%2Finference.py) This file contain the metric calculation for testing the model.
>     - [trainer.py](S3R%2Fanomaly%2Fengine%2Ftrainer.py) This file contain the loss functions, train and validation functions for the training for the network.
>   - [losses](S3R%2Fanomaly%2Flosses) Loss functions.
>     - [sigmoid_mae_loss.py](S3R%2Fanomaly%2Flosses%2Fsigmoid_mae_loss.py)
>     - [smooth_loss.py](S3R%2Fanomaly%2Flosses%2Fsmooth_loss.py)
>     - [sparsity_loss.py](S3R%2Fanomaly%2Flosses%2Fsparsity_loss.py)
>   - [models](S3R%2Fanomaly%2Fmodels)
>     - [detectors](S3R%2Fanomaly%2Fmodels%2Fdetectors)
>       - [detector.py](S3R%2Fanomaly%2Fmodels%2Fdetectors%2Fdetector.py) The model implementation.
>     - [modules](S3R%2Fanomaly%2Fmodels%2Fmodules)
>       - [memory_module.py](S3R%2Fanomaly%2Fmodels%2Fmodules%2Fmemory_module.py) Implementation of enNormalModule
>       - [residual_attention.py](S3R%2Fanomaly%2Fmodels%2Fmodules%2Fresidual_attention.py) Implementation of deNormalModule and GlobalStatistics
>   - [utilities](S3R%2Fanomaly%2Futilities)
> - [configs](S3R%2Fconfigs) Configs that define used datasets.
> - [data](S3R%2Fdata) This dir contains the datasets: files lists for train/test/val
> - [logs](S3R%2Flogs) Directory for saving training logs
> - [tools](S3R%2Ftools) Directory for main scripts
>   - [_init_paths.py](S3R%2Ftools%2F_init_paths.py) Add to path
>   - [trainval_anomaly_detector.py](S3R%2Ftools%2Ftrainval_anomaly_detector.py) This is the main script which runs training or testing. It also initiates a neptune run to log losses and metrics.
> - [config.py](S3R%2Fconfig.py) Config class
> - [utils.py](S3R%2Futils.py) Some utility functions.

> [test](test): This dir contain old files that has been used for testing/production of code.


How to upload result data for the models to neptune? -> Use either the xd_test og ucf_test for the MGFN or RTFM model.

Firstly we run the [1_RTFM_xd_upload.py](RTFMmain%2Fxd_test%2F1_RTFM_xd_upload.py) file to get a neptune run + the loss of the model at the current iteration. Remember to set the path to the correct model and set the parameters for this model correctly.
Then we run the [2_RTFM_xd_test.py](RTFMmain%2Fxd_test%2F2_RTFM_xd_test.py) file to get the results of the model. These results are also uploaded to neptune (remember to set the -n for neptune run argument).
Lastly we can upload the plots using the test-file create before and the script: [3_RTFM_xd_test_plotter.py](RTFMmain%2Fxd_test%2F3_RTFM_xd_test_plotter.py). 
If multiple loss calculations has been done, then an average can be taken and uploaded to Neptune with: [4_RTFM_xd_losswidth.py](RTFMmain%2Fxd_test%2F4_RTFM_xd_losswidth.py).
