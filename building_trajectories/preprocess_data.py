from abc import ABC, abstractclassmethod

class PreprocessData(ABC):
    def create_sentences(self):
        pass
    
    def bearing(self):
        pass
    
    def acceleration(self):
        pass
    
    def delta_space(self):
        pass

    def delta_time(self):
        pass
    
    def days_of_week(self):
        pass
    
    def hours_of_day(self):
        pass
    
    def get_false_labels(self):
        pass
    
    def complete_trajectory(self):
        pass

    def put_statistic_metrics(self):
        pass
    
    def add_padding(self):
        pass
    
    def get_format_date(self):
        pass
    
    def features_trajectory(self):
        '''Based on Trajectory Clustering via Deep Representation Learning
        https://github.com/yaodi833/trajectory2vec/blob/master/tmb2vec.py
        metedos rolling_window, compute_feas, behavior_ext, generate_behavior_sequences'''
        pass
    
    def rolling_window(self):
        pass
    
    def compute_feas(self):
        pass
    
    def behavior_ext(self):
        pass
    
    def generate_behavior_sequences(self):
        pass
    
    def add_bus_stop_label(self):
        pass
    
    def add_traffic_light_label(self):
        pass
    
    def add_other_stop_label(self):
        pass
    
    def label_encoder(self):
        pass
    

    

    