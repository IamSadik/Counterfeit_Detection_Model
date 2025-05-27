import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, losses
from tensorflow.keras.applications import EfficientNetB3

# === CONFIG ===
TEST_DATASET_DIR = "datasets/capsule_verification_dataset_triplet/test"
MODEL_PATH = "models/verification_model/model_triplet.h5"
TARGET_SIZE = (2048, 2048)  # Width x Height
EMBEDDING_DIM = 512
MARGIN = 1.0

# === Triplet Loss (for inference only, not used in accuracy)
class TripletLoss(losses.Loss):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor, positive, negative = tf.unstack(tf.reshape(y_pred, (-1, 3, EMBEDDING_DIM)), 3, 1)
        d_ap = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        d_an = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        loss = tf.maximum(d_ap - d_an + self.margin, 0.0)
        return tf.reduce_mean(loss)

# === Embedding model (EfficientNetB3 -> Dense(512))
def create_embedding_model():
    base_model = EfficientNetB3(include_top=False, input_shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3), pooling='avg')
    base_model.trainable = False
    inputs = layers.Input(shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3))
    x = base_model(inputs)
    x = layers.Dense(EMBEDDING_DIM)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return models.Model(inputs, x, name="embedding_model")

# === Full triplet model wrapper
def build_triplet_model():
    embedding_model = create_embedding_model()

    input_anchor = layers.Input(shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3), name="anchor_input")
    input_positive = layers.Input(shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3), name="positive_input")
    input_negative = layers.Input(shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3), name="negative_input")

    emb_anchor = embedding_model(input_anchor)
    emb_positive = embedding_model(input_positive)
    emb_negative = embedding_model(input_negative)

    merged_output = layers.Concatenate(axis=1)([emb_anchor, emb_positive, emb_negative])
    model = models.Model(inputs=[input_anchor, input_positive, input_negative], outputs=merged_output)
    return model

# === Load test triplets
def load_triplet_data(dataset_path):
    anchor_dir = os.path.join(dataset_path, "anchor")
    positive_dir = os.path.join(dataset_path, "positive")
    negative_dir = os.path.join(dataset_path, "negative")

    files = sorted(os.listdir(anchor_dir))
    anchor_paths = [os.path.join(anchor_dir, f) for f in files]
    positive_paths = [os.path.join(positive_dir, f) for f in files]
    negative_paths = [os.path.join(negative_dir, f) for f in files]

    dataset = tf.data.Dataset.from_tensor_slices((anchor_paths, positive_paths, negative_paths))

    def process_triplet(anchor_path, pos_path, neg_path):
        def load_and_preprocess(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize_with_pad(img, TARGET_SIZE[1], TARGET_SIZE[0])
            img = tf.cast(img, tf.float32) / 255.0
            return img

        anchor = load_and_preprocess(anchor_path)
        positive = load_and_preprocess(pos_path)
        negative = load_and_preprocess(neg_path)
        return (anchor, positive, negative)

    return dataset.map(process_triplet, num_parallel_calls=tf.data.AUTOTUNE)

# === Compute accuracy
def compute_accuracy(model, dataset):
    correct = 0
    total = 0

    for anchor, positive, negative in dataset:
        embeddings = model.predict([tf.expand_dims(anchor, 0),
                                    tf.expand_dims(positive, 0),
                                    tf.expand_dims(negative, 0)], verbose=0)

        emb_anchor, emb_positive, emb_negative = np.split(embeddings[0], 3)

        d_ap = np.sum(np.square(emb_anchor - emb_positive))
        d_an = np.sum(np.square(emb_anchor - emb_negative))

        if d_ap < d_an:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# === Run evaluation
print("📦 Rebuilding model and loading weights...")
model = build_triplet_model()
model.load_weights(MODEL_PATH)

print("📂 Loading test dataset...")
test_dataset = load_triplet_data(TEST_DATASET_DIR)

print("✅ Evaluating...")
accuracy = compute_accuracy(model, test_dataset)

print(f"🎯 Verification Accuracy: {accuracy * 100:.2f}%")
