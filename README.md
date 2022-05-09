# its_research
We are developing possible solutions to solve problems in the context of Intelligent Transportation System.
More precisely, labeling problems and anomaly detection on bus trajectory (GPS). 

# Disclaimer
The preprocessed data from Dublin is available at:
https://drive.google.com/file/d/1tkpxtFulyWQhcaRuCqsDVMBLWW8UN3LL/view?usp=sharing
https://drive.google.com/file/d/1FmjM2Xi-mbwZALOTQcHBEgmg7uc71zwq/view?usp=sharing

All Recife data is private.

The dataset can be used to reproduce the experiments.

## Point Activity Classification
It is a stacked deep-learning model composed of recurrent and attention layers that learn a vector representation for each trajectory point by classifying it into a set of activity points (in route, bus stop, traffic signal, and other stops) based on temporal and spatial features.


## A Spatial-Temporal Outlier Detection
We aim to detect anomalous bus trajectories using supervised learning. Thus, we  propose a multi-class classifier that learns 
the typical behavior of buses but, instead of performing a hard detection decision as literature approaches, ou solution calculates an anomaly score based on the uncertainty of the classifier.

## Applying A Transformer Language Model for Anomaly Detection in Bus Trajectories
* Contents:
   * process_dublin.py
   * pipeline_to_generate_data.py
   * preprocess_recife.py
   * run_loop_transformer.py
   * transformer_model.py
 

## Contents
* Requirements:
* library versions:
    * numpy = 1.18.1
    * h5py = 1.10.4
    * tensorflow = 2.1.0 
    * h3 = 3.4.3
    * gmplot = 1.2.0
    * scikit-learn = 0.22.2
    * scipy = 1.4.1
    * gensim = 3.8.0
    * keras = 2.3.1
    * plotly = 4.5.4
* Models
