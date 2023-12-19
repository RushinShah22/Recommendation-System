import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow_recommenders as tfrs
from tensorflow.python.framework.tensor import Tensor
import tensorflow as tf
import pprint
import pandas as pd
import numpy as np
from typing import Dict, Text

downloads = pd.read_csv("./data/downloads_db.csv",  index_col=0)
users = pd.read_csv("./data/users_db.csv",  index_col=0)
datasets = pd.read_csv("./data/datasets_db.csv", index_col=0)

downloads = downloads.astype(str)

downloads_ds = tf.data.Dataset.from_tensor_slices(dict(downloads))
users_ds = tf.data.Dataset.from_tensor_slices(dict(users))
datasets_ds = tf.data.Dataset.from_tensor_slices(dict(datasets))


tf.random.set_seed(42)


shuffled = downloads_ds.shuffle(1_00_000, seed = 42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)


dataset_titles = datasets_ds.batch(100).map(lambda x: x["title"])
user_ids = downloads_ds.batch(1_00_000).map(lambda x: tf.as_string(x["user_id"]))

unique_dataset_titles = np.unique(np.concatenate(list(dataset_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))



#Creating Vector Embedding

embedding_dimension = 32
user_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32)
])

dataset_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_dataset_titles, mask_token=None),
    tf.keras.layers.Embedding(len(unique_dataset_titles) + 1, 32)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates= datasets_ds.map(lambda x: x['title']).batch(128).map(dataset_model),
    ks= (100,)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)

# Model Definition
class BhuvanModel(tfrs.Model):
    def __init__(self, user_model, dataset_model):
        super().__init__()
        self.dataset_model: tf.keras.Model = dataset_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training: bool = False) ->tf.Tensor:
        user_embedding = self.user_model(features["user_id"])
        positive_dataset_embeddings = self.dataset_model(features["title"])

        return self.task(user_embedding, positive_dataset_embeddings)
    




# Training Two Tower Model 

model = BhuvanModel(user_model, dataset_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)

model.evaluate(cached_test, return_dict = True)

