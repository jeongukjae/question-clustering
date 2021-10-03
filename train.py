import os
import time

import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from absl import logging, flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer("hidden_size", 768, help="hidden size")
flags.DEFINE_string("encoder", "https://tfhub.dev/jeongukjae/distilbert_en_uncased_L-6_H-768_A-12/1", help='backbone encoder')
flags.DEFINE_string("preprocess", "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/1", help='preprocesing layer')
flags.DEFINE_integer("batch_size", 384, help="")
flags.DEFINE_integer("shuffle_size", 1_000_000, help="")
flags.DEFINE_integer("epochs", 5, help="")
flags.DEFINE_float("label_smoothing", 0.1, help="")
flags.DEFINE_float("learning_rate", 2e-4, help="")
flags.DEFINE_float("warmup_ratio", 0.1, help="")
flags.DEFINE_float('initial_temperature', 0.1, help="")
flags.DEFINE_string("tb_log_dir", "./logs", help='log dir')


def main(argv):
    nq_ds = tfds.load('natural_questions_open')
    nq_ds = {
        key: ds.map(lambda x: {"question": x['question'], "answer": x['answer'][0]})
        for key, ds in nq_ds.items()
    }
    preprocessor = hub.KerasLayer(FLAGS.preprocess)
    train_ds = (
        nq_ds['train']
        .shuffle(FLAGS.shuffle_size, reshuffle_each_iteration=True)
        .batch(FLAGS.batch_size)
    )
    num_train_steps = len([1 for _ in train_ds]) * FLAGS.epochs
    train_ds = train_ds.map(lambda x: (preprocessor(x['question']), preprocessor(x['answer'])), num_parallel_calls=tf.data.AUTOTUNE)
    dev_ds = (
        nq_ds['validation']
        .batch(FLAGS.batch_size)
        .map(lambda x: (preprocessor(x['question']), preprocessor(x['answer'])), num_parallel_calls=tf.data.AUTOTUNE)
    )
    logging.info(f"Element spec: {train_ds.element_spec}")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        question_model = _create_model("question_model")
        answer_model = _create_model("answer_model")

        model = ModelForContrastive(question_model, answer_model, temperature=FLAGS.initial_temperature)
        lr_scheduler = BertScheduler(
            rate=FLAGS.learning_rate,
            warmup_ratio=FLAGS.warmup_ratio,
            total_steps=num_train_steps,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_scheduler),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=FLAGS.label_smoothing),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.TopKCategoricalAccuracy(),
            ],
        )

    log_dir = os.path.join(FLAGS.tb_log_dir, str(int(time.time())))
    model.fit(
        train_ds,
        epochs=FLAGS.epochs,
        validation_data=dev_ds,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir, update_freq='batch')]
    )
    question_model.save("./models/question_model/")
    answer_model.save("./models/answer_model/")


def _create_model(name: str):
    encoder_inputs = {
        "input_word_ids": tf.keras.Input([None], dtype=tf.int32, name="input_word_ids"),
        "input_mask": tf.keras.Input([None], dtype=tf.int32, name="input_mask"),
        # "input_type_ids": tf.keras.Input([None], dtype=tf.int32, name="input_type_ids"),
    }
    encoder = hub.KerasLayer(FLAGS.encoder, trainable=True)
    sentence_embedding = encoder(encoder_inputs)["pooled_output"]
    transformed = tf.keras.layers.Dense(FLAGS.hidden_size, activation='tanh', name='transform')(sentence_embedding)

    model = tf.keras.Model(encoder_inputs, transformed, name=name)
    model.summary()
    return model


class ModelForContrastive(tf.keras.Model):
    def __init__(self, model1, model2, temperature, **kwargs):
        super().__init__(**kwargs)

        self.model1 = model1
        self.model2 = model2
        self.temperature = tf.Variable(temperature, trainable=True)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            model1_output = tf.nn.l2_normalize(self.model1(x, training=True), axis=-1)
            model2_output = tf.nn.l2_normalize(self.model2(y, training=True), axis=-1)

            ctx = tf.distribute.get_replica_context()
            model2_output = ctx.all_gather(model2_output, axis=0)

            similarity = tf.tensordot(model1_output, model2_output, axes=[[1], [1]])
            similarity /= self.temperature
            logit = tf.nn.softmax(similarity, axis=-1)

            logit_shape = tf.shape(logit)
            label = tf.range(logit_shape[0]) + (tf.shape(model1_output)[0] * ctx.replica_id_in_sync_group)
            label = tf.one_hot(label, depth=logit_shape[1])

            loss = self.compiled_loss(label, logit, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(label, logit)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        model1_output = tf.nn.l2_normalize(self.model1(x, training=None), axis=-1)
        model2_output = tf.nn.l2_normalize(self.model2(y, training=None), axis=-1)

        ctx = tf.distribute.get_replica_context()
        model2_output = ctx.all_gather(model2_output, axis=0)

        similarity = tf.tensordot(model1_output, model2_output, axes=[[1], [1]])
        similarity /= self.temperature
        logit = tf.nn.softmax(similarity, axis=-1)

        logit_shape = tf.shape(logit)
        label = tf.range(logit_shape[0]) + (tf.shape(model1_output)[0] * ctx.replica_id_in_sync_group)
        label = tf.one_hot(label, depth=logit_shape[1])

        self.compiled_loss(label, logit, regularization_losses=self.losses)
        self.compiled_metrics.update_state(label, logit)
        return {m.name: m.result() for m in self.metrics}


class BertScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate, warmup_ratio, total_steps, name=None):
        super().__init__()

        self.rate = rate
        self.warmup_ratio = warmup_ratio
        self.total_steps = float(total_steps)
        self.warmup_steps = warmup_ratio * total_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope("BertScheduler"):
            total_steps = tf.convert_to_tensor(self.total_steps, name="total_steps")
            warmup_steps = tf.convert_to_tensor(self.warmup_steps, name="warmup_steps")

            current_step = tf.cast(step + 1, tf.float32)

            return self.rate * tf.cond(
                current_step < warmup_steps,
                lambda: self.warmup(current_step, warmup_steps),
                lambda: self.decay(current_step, total_steps, warmup_steps),
            )

    @tf.function
    def warmup(self, step, warmup_steps):
        return step / tf.math.maximum(tf.constant(1.0), warmup_steps)

    @tf.function
    def decay(self, step, total_steps, warmup_steps):
        return tf.math.maximum(
            tf.constant(0.0), (total_steps - step) / tf.math.maximum(tf.constant(1.0), total_steps - warmup_steps)
        )

    def get_config(self):
        return {
            "warmup_ratio": self.warmup_ratio,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }

if __name__ == "__main__":
    app.run(main)
