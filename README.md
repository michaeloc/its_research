# its_research
we are developing possible solutions to solve problems in the context of Intelligent Transportation System.
More precisely, labeling problems and anomaly detection on bus trajectory (GPS). 

# Disclaimer
The preprocessed data from Dublin is available at:

All Recife data is private.

The dataset can be used to reproduce the experiments.

## Point Activity Classification
It is a stacked deep-learning model composed of recurrent and attention layers that learn a vector representation for each trajectory point by classifying it into a set of activity points (in route, bus stop, traffic signal, and other stops) based on temporal and spatial features.


## A Spatial-Temporal Outlier Detection
We aim to detect anomalous bus trajectories using supervised learning. Thus, we  propose a multi-class classifier that learns 
the typical behavior of buses but, instead of performing a hard detection decision as literature approaches, ou solution calculates an anomaly score based on the uncertainty of the classifier.

## Contents
* Requirements
* Models
