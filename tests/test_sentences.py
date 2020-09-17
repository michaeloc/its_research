import unittest
from sentences import Sentences
import pandas as pd

class Tdd_Sentences(unittest.TestCase):

    def setUp(self):
        features = ['timestamp','line_id','direction','journey_id',
            'time_frame','vehicle_journey_id','operator',
            'congestion','lng','lat','delay','block_id',
            'vehicle_id','stop_id', 'stop']
        self.data = pd.read_csv(
            '../data/siri.20130110.csv.gz',
            compression='gzip',
            names=features,
            header=None)
        trajetoria = ['line_id','journey_id','time_frame','vehicle_journey_id','operator','vehicle_id']
        self.sentence = Sentences(self.data, trajetoria)
        self.one_sentence = self._create_one_sentence([66.0, '066A0001', '2013-01-09', 2390, 'PO', 33376])

    def test_trajectory_features_is_subset_of_full_features(self):
        self.assertTrue(set(self.sentence.trajectory_features).issubset(set(self.sentence.full_features)))

    def _create_one_sentence(self, trajectory_id):
#        some_trajectory = [66.0, '066A0001', '2013-01-09', 2390, 'PO', 33376]
        query = "(line_id == {}) & (journey_id == '{}') & (time_frame == '{}')  & (vehicle_journey_id == '{}') & (operator == '{}') & (vehicle_id == {})".format(*trajectory_id)
        df = self.data.query(query)
        return self.sentence.create_sentences(df)

    def _list_dimensionality(self, matrix):
        dims = []
        while (isinstance(matrix, list) and matrix is not None):
            dims.append(len(matrix))
            matrix = matrix[0]
        return len(dims)

    def test_create_one_sentence_type(self):
        self.assertIsInstance(self.one_sentence, list)

    def test_create_one_sentence_dimensionality(self):
        dimm = self._list_dimensionality(self.one_sentence)
        self.assertEqual(dimm, 3)

    def test_create_multiple_sentences(self):
        df_trajSizes = self.data.groupby(self.sentence.trajectory_features).size().reset_index()
        trajetories_matrix = []
        for index in df_trajSizes.index[50:60]:
            print(index)
            trajectory_id = df_trajSizes.loc[index][self.sentence.trajectory_features].values
            one_trajectory = self._create_one_sentence(trajectory_id)
            if (one_trajectory != []):
                print('-')
                trajetories_matrix.append(one_trajectory)
        self.assertEqual(self._list_dimensionality(trajetories_matrix),4)
        #self.assertEqual(1,4)

if __name__ == '__main__':
    unittest.main()
