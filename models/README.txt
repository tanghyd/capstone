Load the LSA model with:

from joblib import dump, load
# lsa_model_path = os.path.join('models','lsa.joblib') # or whereever it is sotred
lsa = load(lsa_model_path)


# if tokens is a LIST of STRINGS - transform vectors with:
vecs = lsa.transform(df.tokens.apply(lambda row : ' '.join(row).strip()))
vecs = pd.DataFrame(vecs, index=df.index)


Load the Doc2Vec model with:

d2v_model_path = os.path.join('models','d2v_v50-c3-e200.model')

# settings to specify (not sure if required, but we'll match the original train data)
vec_size = 50
epochs = 200
min_count = 3

# threads
num_workers = 4

d2v = EventVectoriser(num_workers=num_workers, min_count=min_count, vec_size=vec_size, max_epochs=epochs,
        model_path=d2v_model_path)
d2v.load()

documents = df.token.values # INPUT DOCUMENTS ARE A LIST OF STRINGS

# TRANSFORM/INFER VECTORS FROM MODEL

# https://stackoverflow.com/questions/39580232/doc2vec-how-to-infer-vectors-of-documents-faster
def infer_vector_worker(document):
    vector = d2v.model.infer_vector(document, steps=20, alpha=0.025)
    return vector

with Timer():
    with Pool(processes=num_workers) as pool:
        vecs = np.array(pool.map(infer_vector_worker, documents))

vecs = pd.DataFrame(vecs, index=high_conf.index)
