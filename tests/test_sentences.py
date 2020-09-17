import unittest
from sentences import Sentences

class Tdd_Sentences(unittest.TestCase):

    def setUp(self):
        import pandas as pd
        features = ['timestamp','line_id','direction','journey_id',
            'time_frame','vehicle_journey_id','operator',
            'congestion','lng','lat','delay','block_id',
            'vehicle_id','stop_id', 'stop']
        self.data_merged = pd.read_csv(
            '../data/siri.20130110.csv.gz',
            compression='gzip',
            names=features,
            header=None)
        trajetoria = ['line_id','journey_id','time_frame','vehicle_journey_id','operator','vehicle_id']
        self.sentence = Sentences(self.data_merged.columns, trajetoria)
        self.one_sentence = self._create_one_sentence()

    def test_trajectory_features_is_subset_of_full_features(self):
        self.assertTrue(set(self.sentence.trajectory_features).issubset(set(self.sentence.full_features)))

    def _create_one_sentence(self):
        some_trajectory = [66.0, '066A0001', '2013-01-09', 2390, 'PO', 33376]
        query = "(line_id == {}) & (journey_id == '{}') & (time_frame == '{}')  & (vehicle_journey_id == '{}') & (operator == '{}') & (vehicle_id == {})".format(*some_trajectory)
        df = self.data_merged.query(query)
        return self.sentence.create_sentences(df)

    def test_create_one_sentence_type(self):
        self.assertIsInstance(self.one_sentence, list)

    def test_create_one_sentence_dimensionality(self):
        matrix = self.one_sentence
        dims = []
        while (isinstance(matrix, list) and matrix is not None):
            dims.append(len(matrix))
            matrix = matrix[0]
        self.assertEqual(len(dims), 3)

if __name__ == '__main__':
    unittest.main()
