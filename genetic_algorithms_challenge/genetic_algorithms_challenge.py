"""
DOCSTRING
"""
import numpy
import pandas
import sklearn
import tpot

def demo():
    """
    DOCSTRING
    """
    telescope = pandas.read_csv('data/MAGIC Gamma Telescope Data.csv')
    telescope_shuffle = telescope.iloc[numpy.random.permutation(len(telescope))]
    tele = telescope_shuffle.reset_index(drop=True)
    tele['Class'] = tele['Class'].map({'g':0, 'h':1})
    tele_class = tele['Class'].values
    training_indices, testing_indices = sklearn.model_selection.train_test_split(
        tele.index, stratify=tele_class, train_size=0.75, test_size=0.25)
    validation_indices = testing_indices
    tpot = tpot.TPOTClassifier(generations=5, verbosity=2)
    tpot.fit(tele.drop('Class', axis=1).loc[training_indices].values,
             tele.loc[training_indices, 'Class'].values)
    tpot.score(tele.drop('Class', axis=1).loc[validation_indices].values,
               tele.loc[validation_indices, 'Class'].values)
    tpot.export('pipeline.py')

def pipeline():
    """
    DOCSTRING
    """
    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
    features = tpot_data.drop('target', axis=1)
    training_features, testing_features, training_target, testing_target = \
                sklearn.model_selection.train_test_split(
                    features, tpot_data['target'], random_state=None)
    # Average CV score on the training set was: 0.8818791853962873
    exported_pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=False),
        sklearn.feature_selection.SelectPercentile(
            score_func=sklearn.feature_selection.f_classif, percentile=93),
        sklearn.ensemble.GradientBoostingClassifier(learning_rate=0.1,
                                                    max_depth=7,
                                                    max_features=0.45,
                                                    min_samples_leaf=20,
                                                    min_samples_split=16,
                                                    n_estimators=100,
                                                    subsample=0.6000000000000001))
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
