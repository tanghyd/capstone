import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator, TransformerMixin


class EventVectoriser(TransformerMixin, BaseEstimator):

    def __init__(self,
                 tag_col='event_id',
                 token_col='tokens',
                 max_epochs=100,
                 min_count=1,
                 vec_size=50,
                 alpha=0.025,
                 min_alpha=0.00025,
                 dm=1,
                 num_workers=4,
                 model_path='models/d2v.model'):

        self.tag_col = tag_col
        self.token_col = token_col
        self.max_epochs = max_epochs
        self.min_count = min_count
        self.vec_size = vec_size
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.dm = dm
        self.num_workers = num_workers
        self.model_path = model_path
        self.model = None
        self.predict_vectors = False

    def get_tagged_docs(self, df):
        return df.apply(lambda row: TaggedDocument(row[self.token_col], [row[self.tag_col]]), axis=1)

    def fit(self, X, y=None):
        """ train model on tagged data """
        if self.pretrained:
            return self
        else:
            docs = self.get_tagged_docs(X)
            self.model = Doc2Vec(docs,
                                 vector_size=self.vec_size,
                                 alpha=self.alpha,
                                 min_alpha=self.min_alpha,
                                 min_count=self.min_count,
                                 dm=self.dm,
                                 epochs=self.max_epochs,
                                 workers=self.num_workers)
            return self

    def transform(self, X):
        if self.predict_vectors:
            return self.predict(X)
        else:
            tags = X[self.tag_col]
            vectors = [self._get_vector_from_tag(tag) for tag in tags]
            return pd.DataFrame(vectors, index=tags)

    def save(self, model_path=None):
        model_path = model_path or self.model_path
        self.model.save(model_path)

    def load(self, model_path=None):
        model_path = model_path or self.model_path
        self.model = Doc2Vec.load(model_path)

    def _get_vector_from_tag(self, tag):
        try:
            return self.model.docvecs[tag]
        except:
            return np.nan
        
    def predict(self, X):
        vectors = np.array([self.model.infer_vector(tokens) for tokens in X])
        return vectors
    
    def predict_proba(self, X):
        return self.predict(X)
    
#     def delete_temporary_training_data(self, keep_doctags_vectors=False, keep_inference=True):
#         self.model.delete_temporary_training_data(keep_doctags_vectors=keep_doctags_vectors, keep_inference=keep_inference)