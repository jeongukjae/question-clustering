import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
from tqdm import tqdm
from absl import logging, flags, app
from scipy import stats


FLAGS = flags.FLAGS
flags.DEFINE_string("encoder", "./models/question_model/", help='backbone encoder')
flags.DEFINE_string("preprocess", "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/1", help='preprocesing layer')


def main(argv):
    preprocess = hub.KerasLayer(FLAGS.preprocess)
    encoder = hub.KerasLayer(FLAGS.encoder)

    stsb = (
        tfds.load("glue/stsb", split='validation')
        .batch(256)
        .map(lambda x: (preprocess(x['sentence1']), preprocess(x['sentence2']), x['label']), num_parallel_calls=tf.data.AUTOTUNE)
    )

    similarities = []
    labels = []

    for s1, s2, label in tqdm(stsb):
        s1 = tf.nn.l2_normalize(encoder(s1), axis=-1)
        s2 = tf.nn.l2_normalize(encoder(s2), axis=-1)

        similarities.append(tf.reduce_sum(s1 * s2, axis=-1).numpy())
        labels.append(label.numpy())

    similarities = tf.concat(similarities, axis=0)
    labels = tf.concat(labels, axis=0)

    print(stats.spearmanr(labels, similarities))


if __name__ == "__main__":
    app.run(main)
