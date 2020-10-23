### CLASSIFICATION ###
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np

class CosineMeanClassifier(BaseEstimator, ClassifierMixin):
    """ 
    https://scikit-learn.org/stable/developers/develop.html
    sci-kit Learn compatible classifier that implements classification based on cosine similarity.

    Classifier is fit to a n x m numpy vector (n = number of observations, m = number of column vectors)

    Parameters
    ----------
    cut_off : float, default=0.5
        A parameter used to determine cut-off threshold for prediction.
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    mean_vector : np.array
        The mean vector of the input data fitted to the classifier.

    Example usage:
    >> from capstone import CosineMeanClassifier     
    >> from sklearn.feature_extraction.text import TfidfVectorizer
    >> from sklearn.decomposition import TruncatedSVD
    >> from sklearn.pipeline import Pipeline

    >> cosine_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=2, tokenizer = nlp, sublinear_tf = True)),  # outputs (n x v) array
        ('svd', TruncatedSVD(n_components=50)),  # outputs n x 50 array
        ('cosine', CosineMeanClassifier(cut_off=0.5))
    ])

    >> cosine_pipeline.fit(event_text)  # event text is a pd.Series or np.array of strings
    Pipeline(steps=[('tfidf',
                 TfidfVectorizer(min_df=2, sublinear_tf=True,
                                 tokenizer=<spacy.lang.en.English object at 0x7f7de6d47f70>)),
                ('svd', TruncatedSVD(n_components=50)),
                ('cosine', CosineMeanClassifier())])

    >> cosine_pipeline.predict(df.event_text.values)
    array([0, 0, 0, ..., 0, 0, 1])
    """

    def __init__(self, cutoff=0.5):
        self.cutoff = cutoff  # threshold cut-off for classification : cosine distance is bounded [0,1]

    def fit(self, X, y=None):  # Does not involve labels - we are just using mean vector like a nearest neighbor
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        self.mean_vector = X.mean(axis=0)
        self.n_features_ = X.shape[1]
        return self # Return the classifier
    
    def transform(self, X, y=None):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing the cosine similarities of the input samples.
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')
        X = check_array(X)  # input validation
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
            
        # compute cosine similarity
        magnitude = lambda x : np.sqrt(np.sum(np.power(x,2)))  # utility function to calculate magnitude
        similarity = lambda a, b : (np.dot(a, b)) / (magnitude(a)*magnitude(b)) 
        
        return np.array([similarity(x, self.mean_vector) for x in X])
    
    def predict_proba(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing the cosine similarities of the input samples.
            in ``X``.
        """
        return self.transform(X)
    
    def predict(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing a predictions based on taking a boolean condition cosine similarities of the input samples
            in ``X``. If the cosine similiarity is greater than self.cutoff, then return 1, else 0.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')
        X = check_array(X)  # input validation
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        # compute cosine similarity
        magnitude = lambda x : np.sqrt(np.sum(np.power(x,2)))  # utility function to calculate magnitude
        similarity = lambda a, b : (np.dot(a, b)) / (magnitude(a)*magnitude(b)) 
        
        # apply mean vertically over col and return a (1,n) vector,  return binary integer array
        return np.array([(similarity(x, self.mean_vector) > self.cutoff) for x in X], dtype=np.int64)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
