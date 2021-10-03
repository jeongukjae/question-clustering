import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
from tqdm import tqdm
from absl import logging, flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("encoder", "./models/question_model/", help='backbone encoder')
flags.DEFINE_string("encoder_to_search", "./models/question_model/", help='backbone encoder')
flags.DEFINE_string("preprocess", "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/1", help='preprocesing layer')

def main(argv):
    preprocess = hub.KerasLayer(FLAGS.preprocess)
    encoder = hub.KerasLayer(FLAGS.encoder)
    encoder_to_search = hub.KerasLayer(FLAGS.encoder_to_search)

    qqp = (
        tfds.load("glue/qqp", split='validation')
        .map(lambda x: (x['question1'], x['question2']), num_parallel_calls=tf.data.AUTOTUNE)
    )

    strings = set()
    for q1, q2 in qqp:
        strings.add(q1.numpy().decode('utf8'))
        strings.add(q2.numpy().decode('utf8'))

    questions = (
        tf.data.Dataset.from_tensor_slices(tf.constant(list(strings)))
        .batch(256)
        .map(lambda x: (x, preprocess(x)), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    vectors = []
    sentences = []

    for question, preprocessed_question in tqdm(questions):
        sentences.append(question.numpy())
        vectors.append(tf.nn.l2_normalize(encoder(preprocessed_question), axis=-1).numpy())

    sentences = tf.concat(sentences, axis=0)
    vectors = tf.concat(vectors, axis=0)

    def get_similar_string(target_string, top_k=10):
        target_vector = encoder_to_search(preprocess(tf.constant([target_string])))
        target_vector = tf.nn.l2_normalize(target_vector, axis=-1)
        scores = tf.tensordot(vectors, target_vector, axes=[[1], [1]])[:, 0]
        top_k_scores, indices  = tf.math.top_k(scores, k=top_k)
        gathered_string = tf.gather(sentences, indices, batch_dims=-1)
        return gathered_string, top_k_scores

    while True:
        try:
            sentence = input("input sentence: ")
        except KeyboardInterrupt:
            raise
        except:
            continue

        gathered_string, top_k_scores = get_similar_string(sentence)
        for string, score in zip(gathered_string, top_k_scores):
            tf.print("  Score:", score, ", String:", string)
        print()


if __name__ == "__main__":
    app.run(main)
