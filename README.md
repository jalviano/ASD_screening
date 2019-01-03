# Detecting Autism Spectrum Disorder with Neural Networks

## Abstract

We investigated the application of several different neural networks to the identification of autism spectrum disorder 
(ASD) based on functional and structural brain imaging patterns. Existing research has focused on the use of brain 
functional magnetic resonance imaging (MRI) data to identify regions in the brain that correlate with ASD. We replicated 
a state of the art study using functional MRIs and also explored the use of structural MRI analysis. Our research used 
the publicly available ABIDE dataset, containing functional and structural MRI data for over 1100 subjects. To analyze 
the fMRI data, we transferred weights learned by a denoising autoencoder to a fully-connected network (FCN) and achieved 
0.64 accuracy in detecting ASD. However, we found that the autoencoder was unnecessary and were 0.71 accurate using only 
a slightly deeper fully-connected network, the highest reported accuracy for the ABIDE dataset. For structural MRI 
analysis, we trained 3D and 2D convolutional neural networks (CNNs) as well as a second FCN but were unable to get any 
signal from the test set for any of our approaches. We conclude that while we were able to improve upon the current 
state of the art for functional MRI analysis, further experimentation is necessary to successfully use structural MRI 
data.

## Experiments

To run all experiments, first download the required datasets. Next train the models and generate figure and table data.

### Download ABIDE I preprocessed datasets

To download data, run:

    cd src/main/utils
    python3 download_abide_preproc.py -d rois_cc200 -p cpac -s filt_global -o ../data
    python3 download_abide_preproc.py -d anat_thickness -p ants -s ants -o ../data
    python3 download_abide_preproc.py -d roi_thickness -p ants -s ants -o ../data


### Train models

To train all models, run:

    cd src/main
    python3 gen_paper_data.py
    
### Generate figures

To generate all figures, run:

    cd src/main
    python3 gen_paper_figs.py
