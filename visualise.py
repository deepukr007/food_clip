import numpy as np
import umap
import umap.plot

embeddings = np.load("image_embeddings.npy")
mapper = umap.UMAP().fit(embeddings)
umap.plot.points(mapper)
