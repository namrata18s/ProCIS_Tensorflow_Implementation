import logging
import pickle
import math
import json
import random
import os
import time

import numpy as np
import faiss
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from typing import List, Union
from transformers import TFAutoModel, AutoTokenizer

random.seed(42)
logging.getLogger().setLevel(logging.INFO)

# TensorFlow implementation of a vanilla dense retriever


class VectorIndex:
    def __init__(self, d):
        self.d = d
        self.vectors = []
        self.index = None

    def add(self, v):
        self.vectors.append(v)

    def build(self, use_gpu=False):
        self.vectors = np.array(self.vectors)

        faiss.normalize_L2(self.vectors)

        logging.info('Indexing {} vectors'.format(self.vectors.shape[0]))

        if self.vectors.shape[0] > 8000000:
            num_centroids = 8 * int(math.sqrt(math.pow(2, int(math.log(self.vectors.shape[0], 2)))))

            logging.info('Using {} centroids'.format(num_centroids))

            self.index = faiss.index_factory(self.d, "IVF{}_HNSW32,Flat".format(num_centroids))

            ngpu = faiss.get_num_gpus()
            if ngpu > 0 and use_gpu:
                logging.info('Using {} GPUs'.format(ngpu))

                index_ivf = faiss.extract_index_ivf(self.index)
                clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(self.d))
                index_ivf.clustering_index = clustering_index

            logging.info('Training index...')

            self.index.train(self.vectors)
        else:
            self.index = faiss.IndexFlatL2(self.d)
            if faiss.get_num_gpus() > 0 and use_gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index)

        logging.info('Adding vectors to index...')

        self.index.add(self.vectors)

    def load(self, path):
        self.index = faiss.read_index(path)

    def save(self, path):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def save_vectors(self, path):
        pickle.dump(self.vectors, open(path, 'wb'), protocol=4)

    def search(self, vectors, k=1, probes=512):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        faiss.normalize_L2(vectors)
        try:
            self.index.nprobe = probes
        except:
            pass
        distances, ids = self.index.search(vectors, k)
        similarities = [(2-d)/2 for d in distances]
        return ids, similarities


class InputExample:
    def __init__(self, texts: List[str], label: Union[int, float] = 0):
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, text: {}".format(str(self.label), self.texts[0])


class TFSentenceTransformer(keras.Model):
    def __init__(self, model_name):
        super(TFSentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = TFAutoModel.from_pretrained(model_name)
        
    def call(self, inputs):
        # Get CLS token embedding (equivalent to pooling_mode_cls_token=True in original code)
        return self.transformer(inputs)[0][:, 0, :]
    
    def encode(self, sentences, batch_size=32, show_progress_bar=False):
        all_embeddings = []
        
        # Handle batching
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")
            
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(sentences))
            batch_sentences = sentences[start_idx:end_idx]
            
            # Tokenize
            inputs = self.tokenizer(batch_sentences, padding=True, truncation=True, 
                                   return_tensors="tf", max_length=512)
            
            # Get embeddings
            with tf.GradientTape(persistent=False) as tape:
                embeddings = self(inputs)
            
            # Convert to numpy and normalize
            embeddings = embeddings.numpy()
            # L2 normalize
            faiss.normalize_L2(embeddings)
            
            all_embeddings.append(embeddings)
            
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings


class MultipleNegativesRankingLoss(keras.losses.Loss):
    def __init__(self, scale=20.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        
    def call(self, y_true, y_pred):
        # Reshape predictions: (batch_size, 2, embedding_dim)
        embeddings = tf.reshape(y_pred, (-1, 2, tf.shape(y_pred)[1] // 2))
        
        # Split embeddings for queries and positives
        query_embeddings = embeddings[:, 0]
        positive_embeddings = embeddings[:, 1]
        
        # Compute similarity scores
        similarity_scores = tf.matmul(query_embeddings, positive_embeddings, transpose_b=True) * self.scale
        
        # Create labels (diagonal is positive pairs)
        batch_size = tf.shape(similarity_scores)[0]
        labels = tf.range(0, batch_size)
        
        # Compute softmax cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=similarity_scores
        )
        
        return tf.reduce_mean(loss)


class DenseRetriever:
    def __init__(self, model_path=None, batch_size=256, use_gpu=True, dim=768):
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.vector_index = VectorIndex(dim)
        
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path=None):
        logging.info("Loading model weights...")
        self.model = TFSentenceTransformer(model_path)
        # If there are saved weights in model_path directory, load them
        checkpoint_path = os.path.join(model_path, 'model.weights.h5')
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
        logging.info("Loaded model weights.")

    def create_index_from_documents(self, documents):
        logging.info('Building index...')
        
        self.vector_index.vectors = self.model.encode(documents, batch_size=self.batch_size, show_progress_bar=True)
        self.vector_index.build(self.use_gpu)
        
        logging.info('Built index')

    def create_index_from_vectors(self, vectors_path):
        logging.info('Building index...')
        logging.info('Loading vectors...')
        self.vector_index.vectors = pickle.load(open(vectors_path, 'rb'))
        logging.info('Vectors loaded')
        self.vector_index.build(use_gpu=False)
        
        logging.info('Built index')

    def search(self, queries, topk=1000, probes=512, min_similarity=0):
        query_vectors = self.model.encode(queries, batch_size=self.batch_size)
        ids, similarities = self.vector_index.search(query_vectors, k=topk, probes=probes)
        results = []
        for j in range(len(ids)):
            results.append([
                (ids[j][i], similarities[j][i]) for i in range(len(ids[j])) if similarities[j][i] > min_similarity
            ])
        return results

    def load_index(self, path):
        self.vector_index.load(path)

    def save_index(self, index_path='', vectors_path=''):
        if vectors_path != '':
            self.vector_index.save_vectors(vectors_path)
        if index_path != '':
            self.vector_index.save(index_path)

    def _create_dataset(self, examples, batch_size=64, shuffle=True):
        queries = []
        positives = []
        
        for example in examples:
            if isinstance(example, InputExample):
                if example.label == 1:
                    queries.append(example.texts[0])
                    positives.append(example.texts[1])
            else:  # Tuple format
                if example[2] == 1:
                    queries.append(example[0])
                    positives.append(example[1])
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((queries, positives))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(queries))
            
        # Batch and prefetch for performance
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset, len(queries)

    def _prepare_data_for_model(self, queries, positives):
        # Tokenize queries
        query_inputs = self.model.tokenizer(
            queries, padding=True, truncation=True, return_tensors="tf", max_length=512
        )
        
        # Tokenize positive examples
        positive_inputs = self.model.tokenizer(
            positives, padding=True, truncation=True, return_tensors="tf", max_length=512
        )
        
        # Create stacked inputs
        stacked_inputs = {
            "input_ids": tf.concat([query_inputs["input_ids"], positive_inputs["input_ids"]], axis=0),
            "attention_mask": tf.concat([query_inputs["attention_mask"], positive_inputs["attention_mask"]], axis=0),
        }
        
        if "token_type_ids" in query_inputs:
            stacked_inputs["token_type_ids"] = tf.concat(
                [query_inputs["token_type_ids"], positive_inputs["token_type_ids"]], axis=0
            )
            
        return stacked_inputs

    def _compute_similarity_scores(self, query_embeddings, doc_embeddings):
        # Compute dot product similarity
        similarity_matrix = tf.matmul(query_embeddings, doc_embeddings, transpose_b=True)
        return similarity_matrix

    def evaluate(self, dev_examples, batch_size=64):
        dev_queries = []
        dev_docs = []
        dev_labels = []
        
        for example in dev_examples:
            if len(example) == 3:  # Tuple format
                dev_queries.append(example[0])
                dev_docs.append(example[1])
                dev_labels.append(float(example[2]))
        
        # Encode all queries and documents
        query_embeddings = self.model.encode(dev_queries, batch_size=batch_size)
        doc_embeddings = self.model.encode(dev_docs, batch_size=batch_size)
        
        # Calculate cosine similarities
        sim_scores = np.zeros(len(dev_queries))
        for i in range(len(dev_queries)):
            # Calculate cosine similarity
            sim = np.dot(query_embeddings[i], doc_embeddings[i]) / (
                np.linalg.norm(query_embeddings[i]) * np.linalg.norm(doc_embeddings[i])
            )
            sim_scores[i] = sim
        
        # Calculate Spearman correlation
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(sim_scores, dev_labels)
        
        return {
            "spearman_correlation": correlation,
            "avg_similarity": np.mean(sim_scores)
        }

    def train(self,
              train_examples,
              dev_examples,
              model_name="distilbert-base-uncased",
              output_path="weights",
              epochs=3,
              evaluation_steps=1000,
              warmup_steps=1000,
              batch_size=64
              ):
        self.load_model(model_name)
        
        logging.info("Loading dataset...")
        
        # Prepare training data
        examples = []
        for ex in train_examples:
            if ex[2] == 1:
                examples.append(InputExample(texts=[ex[0], ex[1]], label=1))
        
        train_dataset, train_size = self._create_dataset(examples, batch_size=batch_size)
        
        # Create optimizer with learning rate schedule
        steps_per_epoch = math.ceil(train_size / batch_size)
        total_steps = steps_per_epoch * epochs
        
        optimizer = keras.optimizers.Adam(
            keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=2e-5,
                decay_steps=total_steps,
                end_learning_rate=0
            )
        )
        
        # Initialize the loss function
        loss_fn = MultipleNegativesRankingLoss()
        
        # Prepare the metrics
        train_loss_metric = keras.metrics.Mean(name="train_loss")
        
        # Create checkpoint manager
        os.makedirs(output_path, exist_ok=True)
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, output_path, max_to_keep=3
        )
        
        # Training loop
        logging.info("Training model...")
        
        best_score = -1
        global_step = 0
        
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            
            # Iterate over batches
            for batch_idx, (queries, positives) in enumerate(tqdm(train_dataset, total=steps_per_epoch)):
                global_step += 1
                
                # Prepare data
                stacked_inputs = self._prepare_data_for_model(queries, positives)
                
                # Train step
                with tf.GradientTape() as tape:
                    # Get embeddings for all inputs
                    all_embeddings = self.model(stacked_inputs)
                    
                    # Split into query and positive embeddings
                    batch_size = tf.shape(queries)[0]
                    query_embeddings = all_embeddings[:batch_size]
                    positive_embeddings = all_embeddings[batch_size:]
                    
                    # Concatenate for loss function
                    embeddings = tf.concat([query_embeddings, positive_embeddings], axis=1)
                    
                    # Add dummy labels (just zeros) as they're not used in our loss function
                    dummy_labels = tf.zeros((batch_size,))
                    
                    # Calculate loss
                    loss = loss_fn(dummy_labels, embeddings)
                
                # Compute gradients and update weights
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                # Update metrics
                train_loss_metric.update_state(loss)
                
                # Evaluate periodically
                if global_step % evaluation_steps == 0:
                    # Compute evaluation metrics
                    eval_results = self.evaluate(dev_examples, batch_size=batch_size)
                    
                    logging.info(
                        f"Step {global_step}: Loss: {train_loss_metric.result():.4f}, "
                        f"Eval Spearman: {eval_results['spearman_correlation']:.4f}"
                    )
                    
                    # Save if it's the best model so far
                    if eval_results['spearman_correlation'] > best_score:
                        best_score = eval_results['spearman_correlation']
                        self.model.save_weights(os.path.join(output_path, "model.weights.h5"))
                        logging.info(f"New best model saved with score: {best_score:.4f}")
                    
                    # Reset metrics
                    train_loss_metric.reset_states()
                    
                    # Save checkpoint
                    checkpoint_manager.save()
        
        # Save final model
        self.model.save_weights(os.path.join(output_path, "final_model.weights.h5"))
        logging.info("Training completed!")


def remove_newlines_tabs(text):
    return text.replace("\n", " ").replace("\t", " ").replace("\r", "")


def limit_str_tokens(text, limit):
    tokens = text.split(" ")
    return " ".join(tokens[:limit])


def limit_turns_tokens(texts, limit):
    added_tokens = 0
    trunc_texts = []
    for text in reversed(texts):
        trunc_text_tokens = []
        tokens = text.split(" ")
        for token in tokens:
            if added_tokens == limit:
                break
            trunc_text_tokens.append(token)
            added_tokens += 1
        trunc_texts.append(" ".join(trunc_text_tokens))
    trunc_texts = [t for t in trunc_texts if t != ""]
    trunc_texts = reversed(trunc_texts)
    return " | ".join(trunc_texts)


def prepare_query(d, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100):
    last_k_turns = 0
    use_title = True
    use_content = True

    turns = [t["text"] for t in d["thread"]]

    if turns_max_tokens > 0:
        query_turns = limit_turns_tokens(turns[-last_k_turns:], turns_max_tokens)
    else:
        query_turns = " ".join(turns[-last_k_turns:])

    query_title = d["post"]["title"] if use_title else ""
    if title_max_tokens > 0:
        query_title = limit_str_tokens(query_title, title_max_tokens)

    query_content = d["post"]["text"] if use_content else ""
    if post_max_tokens > 0:
        query_content = limit_str_tokens(query_content, post_max_tokens)

    query = remove_newlines_tabs(" | ".join([query_title, query_content, query_turns]))

    return query


if __name__ == "__main__":
    print('loading data...')
    wiki_to_doc = {}
    with open("collection.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line.rstrip("\n"))
            wiki_to_doc[d["wiki"]] = d["wiki"].replace('_', ' ') + ': ' + d["contents"]

    train_examples = []
    with open("train.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line.rstrip("\n"))
            for wiki in d["wiki_links"]:
               train_examples.append((prepare_query(d), wiki_to_doc[wiki], 1.0))

    print(train_examples[0])
    print('-------------------')
    
    dev_examples = []
    wikis = list(wiki_to_doc.keys())
    with open("gold.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line.rstrip("\n"))
            for wiki in d["wiki_links"]:
                query = prepare_query(d)
                dev_examples.append((query, wiki_to_doc[wiki], 1.0))
                # sample 100 negative examples for the query so that the wiki doc is not in the wiki_links
                for i in range(100):
                    random_wiki = wikis[np.random.randint(len(wikis))]
                    while random_wiki in d["wiki_links"]:
                        random_wiki = wikis[np.random.randint(len(wikis))]
                    dev_examples.append((query, wiki_to_doc[random_wiki], 0.0))
    
    print(dev_examples[0])

    # training example
    de = DenseRetriever()
    de.train(train_examples=train_examples,
             dev_examples=dev_examples,
             output_path="weights_tf_dr_v_64",
             model_name="distilbert-base-uncased",
             epochs=10,
             evaluation_steps=2000,
             warmup_steps=1000,
             batch_size=64)

    # search example
    # de = DenseRetriever(model_path="weights_tf_dr_v_64")
    # de.create_index_from_documents(["hello", "hi"])
    # results = de.search(["hello there"])
