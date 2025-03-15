import tensorflow as tf
import wandb
import json
import numpy as np

from tqdm import tqdm
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from .dr import prepare_query  # Assuming this import works the same way

class BinaryClassifier:
    def __init__(self):
        self.device = "CPU"
        print("Step 1: Starting proactive classifier loading...")

        print("Step 1.1: Loading model architecture...")
        # Using TensorFlow version of the model
        base_model = TFAutoModel.from_pretrained("microsoft/deberta-v3-base")
        
        # Creating the classification head
        inputs = {
            "input_ids": tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids"),
            "attention_mask": tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask"),
        }
        
        outputs = base_model(inputs)[0]  # Get the last hidden state
        # Use the [CLS] token representation (first token)
        cls_output = outputs[:, 0, :]
        # Add classification layers
        logits = Dense(2)(cls_output)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=logits)
        print("Model architecture loaded.")

        print("Step 1.2: Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        print("Tokenizer loaded.")
        
        # Set max_length
        self.max_length = 512  # Default max_length, can be overridden

    def train(self, train_docs, train_labels, val_docs, val_labels, epochs=3, batch_size=64, lr=2e-5):
        # Tokenize the data
        train_encodings = self.tokenizer(train_docs, truncation=True, padding='max_length', 
                                         max_length=self.max_length, return_tensors='tf')
        val_encodings = self.tokenizer(val_docs, truncation=True, padding='max_length', 
                                       max_length=self.max_length, return_tensors='tf')
        
        # Convert labels to numpy arrays
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask']
            },
            train_labels
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask']
            },
            val_labels
        ))
        
        # Handle class imbalance
        class_counts = np.bincount(train_labels)
        class_weights = {i: len(train_labels) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        
        # Configure the dataset for performance
        train_dataset = train_dataset.shuffle(len(train_docs)).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()]
        )
        
        # Set up WandB callback
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            save_model=False
        )
        
        # Set up model checkpoint callback
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_sparse_categorical_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        # Train the model
        self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=[wandb_callback, checkpoint],
            class_weight=class_weights
        )
        
    def evaluate(self, val_docs, val_labels, batch_size=64):
        # Tokenize the data
        val_encodings = self.tokenizer(val_docs, truncation=True, padding='max_length', 
                                      max_length=self.max_length, return_tensors='tf')
        val_labels = np.array(val_labels)
        
        # Create dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask']
            },
            val_labels
        )).batch(batch_size)
        
        # Evaluate the model
        results = self.model.evaluate(val_dataset)
        return results[1]  # Return accuracy
    
    def predict(self, doc):
        # Tokenize the input
        encoding = self.tokenizer(doc, truncation=True, padding='max_length', 
                                 max_length=self.max_length, return_tensors='tf')
        
        # Get predictions
        logits = self.model.predict({
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        })
        
        # Get the predicted class
        pred_label = tf.argmax(logits, axis=1)[0].numpy()
        return int(pred_label)
    
    def load_model(self, model_path):
        # Load the model
        self.model.load_weights(model_path)
        print("Model weights loaded successfully.")
        
        # Set the model to evaluation mode
        self.model.trainable = False

# TF Dataset loader not needed since we're using tf.data.Dataset directly

if __name__ == '__main__':
    # Initialize wandb
    wandb.init(project="binary-classifier-deberta-tf")

    # Prepare the data
    print("Loading data...")
    docs = []
    labels = []
    label_0_count = 0
    label_1_count = 0
    with open('train.jsonl') as f:
        for line in tqdm(f):
            d = json.loads(line)
            for i in range(len(d['thread'])):
                d_sub = d.copy()
                d_sub['thread'] = d['thread'][:i]
                doc = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
                label = 1 if len(d['thread'][i]['wiki_links']) > 0 else 0
                docs.append(doc)
                labels.append(label)
                if label == 0:
                    label_0_count += 1
                else:
                    label_1_count += 1

    print(f"Label 0 count: {label_0_count}, Label 1 count: {label_1_count}")

    # Split the data into train and validation sets
    train_docs, val_docs, train_labels, val_labels = train_test_split(docs, labels, test_size=1000, stratify=labels)

    # Create an instance of the binary classifier
    classifier = BinaryClassifier()
    
    # Train the classifier
    classifier.train(train_docs, train_labels, val_docs, val_labels)

    # Load the best model
    classifier.load_model('best_model.h5')

    # Load some examples from the train set for inference
    docs = []
    labels = []
    limit = 100
    with open('train.jsonl') as f:
        for line in f:
            d = json.loads(line)
            for i in range(len(d['thread'])):
                d_sub = d.copy()
                d_sub['thread'] = d_sub['thread'][:i]  # Exclude item at index i
                doc = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
                label = 1 if len(d['thread'][i]['wiki_links']) > 0 else 0  # Label based on next item
                docs.append(doc)
                labels.append(label)
            limit -= 1
            if limit == 0:
                break
                
    # Example inference
    correct = 0
    total = 0
    for doc, label in zip(docs, labels):
        pred_label = classifier.predict(doc)
        print(f"Predicted label: {pred_label}, True label: {label}")
        if pred_label == label:
            correct += 1
        total += 1
    print(f"Accuracy: {correct / total}")
