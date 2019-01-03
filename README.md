# Detecting Autism Spectrum Disorder with Neural Networks

## Download ABIDE I preprocessed datasets

To download data, run:

    cd src/main/utils
    python3 download_abide_preproc.py -d rois_cc200 -p cpac -s filt_global -o ../data
    python3 download_abide_preproc.py -d anat_thickness -p ants -s ants -o ../data
    python3 download_abide_preproc.py -d roi_thickness -p ants -s ants -o ../data


## Train models

To train all models, run:

    cd src/main
    python3 gen_paper_data.py
    
## Generate figures

To generate all figures, run:

    cd src/main
    python3 gen_paper_figs.py
