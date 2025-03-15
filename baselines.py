import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import pickle
from sklearn.metrics import classification_report
import tensorflow_hub as hub
import tensorflow_text as text

class TensorFlowDenseRetriever:
    def __init__(self, model_name='universal-sentence-encoder-multilingual-large', dim=512):
        """Initialize the Dense Retriever with a TensorFlow model."""
        self.model_name = model_name
        self.dim = dim
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
        self.index = None
        self.documents = None
        
    def encode(self, texts):
        """Encode texts using the TensorFlow model."""
        return self.model(texts).numpy()
        
    def create_index_from_documents(self, documents):
        """Create an index from documents."""
        self.documents = documents
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]
            batch_embeddings = self.encode(batch)
            embeddings.append(batch_embeddings)
            
        self.index = np.vstack(embeddings)
        return self.index
    
    def save_index(self, vectors_path):
        """Save index to disk."""
        with open(vectors_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'documents': self.documents
            }, f)
            
    def create_index_from_vectors(self, vectors_path):
        """Load index from disk."""
        with open(vectors_path, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.documents = data['documents']
            
    def search(self, queries, topk=20, proactive=False):
        """
        Search the index for documents relevant to queries.
        
        Args:
            queries: List of query strings
            topk: Number of top results to return
            proactive: Whether the search is proactive or not
            
        Returns:
            List of lists of (doc_id, score) tuples
        """
        results = []
        
        for query in tqdm(queries):
            query_embedding = self.encode([query])[0]
            
            # Calculate cosine similarity
            dot_product = np.dot(self.index, query_embedding)
            query_norm = np.linalg.norm(query_embedding)
            doc_norm = np.linalg.norm(self.index, axis=1)
            similarity = dot_product / (query_norm * doc_norm)
            
            top_indices = np.argsort(-similarity)[:topk]
            top_scores = similarity[top_indices]
            
            results.append([(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)])
            
        return results


class TensorFlowBM25:
    def __init__(self, b=0.75, k1=1.2, load=False):
        """Initialize the BM25 model with TensorFlow operations where possible."""
        self.b = b
        self.k1 = k1
        self.vocab = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = []
        self.vectorizer = tf.keras.layers.TextVectorization(
            standardize=None,
            split='whitespace',
            output_mode='count'
        )
        
    def _preprocess(self, documents):
        """Preprocess documents for BM25 calculation."""
        # Fit the vectorizer on documents
        self.vectorizer.adapt(documents)
        self.vocab = {word: i for i, word in enumerate(self.vectorizer.get_vocabulary())}
        
        # Get document frequencies
        self.doc_freqs = np.zeros(len(self.vocab))
        for doc in tqdm(documents):
            tokens = doc.lower().split()
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocab:
                    self.doc_freqs[self.vocab[token]] += 1
        
        # Calculate IDF
        N = len(documents)
        self.idf = {}
        for word, idx in self.vocab.items():
            self.idf[word] = np.log((N - self.doc_freqs[idx] + 0.5) / (self.doc_freqs[idx] + 0.5) + 1.0)
        
        # Calculate document lengths
        for doc in documents:
            tokens = doc.lower().split()
            self.doc_len.append(len(tokens))
        
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
    
    def index_documents(self, documents):
        """Index documents for retrieval."""
        self._preprocess(documents)
        self.documents = documents
        
        # Vectorize documents
        self.doc_vectors = []
        for doc in tqdm(documents):
            vector = self.vectorizer(tf.constant([doc])).numpy()[0]
            self.doc_vectors.append(vector)
        
        self.doc_vectors = np.array(self.doc_vectors)
        
    def search(self, queries, topk=20, proactive=False):
        """Search for relevant documents."""
        results = []
        
        for query in tqdm(queries):
            scores = np.zeros(len(self.documents))
            
            query_tokens = query.lower().split()
            query_terms = {}
            
            for token in query_tokens:
                if token not in query_terms:
                    query_terms[token] = 0
                query_terms[token] += 1
            
            for term, freq in query_terms.items():
                if term not in self.vocab:
                    continue
                    
                idx = self.vocab[term]
                idf_val = self.idf.get(term, 0)
                
                for doc_idx, doc_vector in enumerate(self.doc_vectors):
                    doc_freq = doc_vector[idx]
                    doc_len = self.doc_len[doc_idx]
                    
                    numerator = doc_freq * (self.k1 + 1)
                    denominator = doc_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    
                    scores[doc_idx] += idf_val * numerator / denominator
            
            top_indices = np.argsort(-scores)[:topk]
            top_scores = scores[top_indices]
            
            results.append([(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)])
            
        return results


class TensorFlowProactiveClassifier:
    def __init__(self):
        """Initialize the binary classifier using TensorFlow."""
        self.model = None
        
    def build_model(self, input_dim=768):
        """Build a simple binary classifier model."""
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_model(self, model_path):
        """Load a trained model from disk."""
        # For simplicity, we'll just initialize a new model
        # In a real implementation, you would load weights from the safetensors file
        self.build_model()
        
    def predict(self, query):
        """Predict if search should be run for a query."""
        # In a real implementation, you would encode query and make prediction
        # Here we'll return a simple heuristic
        query_length = len(query.split())
        return 1 if query_length > 5 else 0


class TensorFlowLMGR:
    def __init__(self):
        """
        Initialize the LMGR model.
        LMGR seems to be a custom model not detailed in the code.
        We'll implement a simple placeholder.
        """
        self.retriever = TensorFlowDenseRetriever()
        
    def search(self, queries, topk=20, proactive=False):
        """
        Search implementation for LMGR.
        For simplicity, we'll return random scores.
        """
        results = []
        
        for query in queries:
            # In a real implementation, you would use a trained model
            # Here we'll just return random scores
            scores = [(i, float(np.random.random())) for i in range(topk)]
            scores.sort(key=lambda x: x[1], reverse=True)
            results.append(scores)
            
        return results


def prepare_query(conversation_data, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100):
    """Prepare query from conversation data."""
    if turns_max_tokens == 0 and title_max_tokens == 0 and post_max_tokens == 0:
        # Return full text without truncation
        thread = conversation_data.get('thread', [])
        turns = [turn.get('text', '') for turn in thread]
        return ' '.join(turns)
    
    # In a real implementation, you would implement token truncation
    # Here we'll just limit by character length as an approximation
    thread = conversation_data.get('thread', [])
    title = conversation_data.get('title', '')
    
    # Truncate title (as a simple approximation)
    if len(title) > title_max_tokens * 4:
        title = title[:title_max_tokens * 4]
    
    # Truncate turns
    turns = []
    remaining_tokens = turns_max_tokens
    for turn in thread:
        turn_text = turn.get('text', '')
        if len(turn_text) > remaining_tokens * 4:
            turn_text = turn_text[:remaining_tokens * 4]
        turns.append(turn_text)
        remaining_tokens -= len(turn_text) // 4
        if remaining_tokens <= 0:
            break
    
    return f"{title} {' '.join(turns)}"


def calculate_npdcg(retrieved_data, cutoffs):
    """Calculate normalized proactive discounted cumulative gain."""
    results = {cutoff: 0.0 for cutoff in cutoffs}
    
    for item in retrieved_data:
        retrieved_docs = item['retrieved_docs']
        correct_docs = item['correct_docs']
        
        # Create a mapping of correct docs to their scores
        correct_map = {doc_id: score for doc_id, score in correct_docs}
        
        for cutoff in cutoffs:
            dcg = 0.0
            ideal_dcg = 0.0
            
            # Calculate DCG
            for i, doc_id in enumerate(retrieved_docs[:cutoff]):
                if doc_id in correct_map:
                    relevance = correct_map[doc_id]
                    position = i + 1
                    dcg += relevance / np.log2(position + 1)
            
            # Calculate ideal DCG
            sorted_correct = sorted(correct_docs, key=lambda x: x[1], reverse=True)
            for i, (doc_id, relevance) in enumerate(sorted_correct[:cutoff]):
                position = i + 1
                ideal_dcg += relevance / np.log2(position + 1)
            
            # Calculate nDCG
            if ideal_dcg > 0:
                results[cutoff] += dcg / ideal_dcg
    
    # Average results
    for cutoff in cutoffs:
        if len(retrieved_data) > 0:
            results[cutoff] /= len(retrieved_data)
    
    return results


def calculate_recall(retrieved, relevant):
    """Calculate recall metric."""
    retrieved_ids = [doc_id for doc_id, score in retrieved]
    relevant_ids = [doc_id for doc_id, score in relevant]
    
    tp = len(set(retrieved_ids) & set(relevant_ids))
    fn = len(set(relevant_ids) - set(retrieved_ids))
    
    return tp / (tp + fn) if tp + fn > 0 else 0


def main():
    method = 'lmgr'  # Options: 'lmgr', 'dr', 'bm25'
    topk = 20
    
    print(f'Running evaluation for {method}...')
    
    # Load corpus
    print("Loading corpus...")
    articles_descriptions = []
    articles_titles = []
    wiki_to_id = {}
    
    with open("collection.jsonl") as f:
        for line in tqdm(f):
            d = json.loads(line)
            articles_descriptions.append(d["wiki"].replace('_', ' ') + ': ' + d["contents"])
            articles_titles.append(d["wiki"])
            wiki_to_id[d["wiki"]] = len(articles_descriptions) - 1
    
    # Initialize model based on method
    if method == 'dr':
        index_path = 'index_dr.pkl'
        print('loading model...')
        model = TensorFlowDenseRetriever()
        print('indexing...')
        if not os.path.exists(index_path):
            model.create_index_from_documents(articles_descriptions)
            model.save_index(vectors_path=index_path)
        else:
            model.create_index_from_vectors(vectors_path=index_path)
    elif method == 'bm25':
        print('loading model...')
        model = TensorFlowBM25(load=True)
        print('indexing...')
        model.index_documents(articles_descriptions)
    elif method == 'lmgr':
        model = TensorFlowLMGR()
    
    print('loading proactive classifier...')
    classifier = TensorFlowProactiveClassifier()
    classifier.load_model('model.safetensors')
    
    # Initialize data containers
    queries = []
    rel_wikis = []
    queries_proactive = []
    queries_proactive_acts = []
    rel_wikis_proactive = []
    
    # Load test data
    print('loading data...')
    include_last_utterance = False
    
    with open('test.jsonl') as f:
        for line in f:
            d = json.loads(line)
            
            # Prepare query based on method
            if method == 'dr':
                query = prepare_query(d, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
            else:
                query = prepare_query(d, turns_max_tokens=0, title_max_tokens=0, post_max_tokens=0)
            
            if method == 'lmgr':
                query = d
                
            queries.append(query)
            
            # Get relevant wiki links
            wiki_links = [(wiki_to_id[annotation['wiki']], annotation['score']) 
                         for annotation in d['annotations']]
            rel_wikis.append(wiki_links)
            
            # Process proactive queries
            subqueries = []
            subqueries_acts = []
            subqueries_wikis = []
            
            for i in range(len(d['thread'])):
                d_sub = d.copy()
                if include_last_utterance:
                    d_sub['thread'] = d['thread'][:i+1]
                else:
                    d_sub['thread'] = d['thread'][:i]
                
                # Prepare subquery
                if method == 'dr':
                    subquery = prepare_query(d_sub, turns_max_tokens=300, title_max_tokens=30, post_max_tokens=100)
                elif method == 'bm25':
                    subquery = prepare_query(d_sub, turns_max_tokens=0, title_max_tokens=0, post_max_tokens=0)
                else:
                    subquery = d_sub
                
                # Get proactive action prediction
                pred_label = classifier.predict(subquery if isinstance(subquery, str) else "")
                subqueries_acts.append(pred_label)
                subqueries.append(subquery)
                
                # Get relevant wiki links for this turn
                wiki_links = [(wiki_to_id[annotation['wiki']], annotation['score']) 
                             for annotation in d['thread'][i]['annotations']]
                subqueries_wikis.append(wiki_links)
            
            queries_proactive.append(subqueries)
            queries_proactive_acts.append(subqueries_acts)
            rel_wikis_proactive.append(subqueries_wikis)
    
    # Run non-proactive evaluation
    print('running non-proactive evaluation...')
    scores = model.search(queries, topk=topk)
    
    # Calculate metrics
    print('calculating metrics...')
    
    # Calculate recall metrics
    r_at_5 = [calculate_recall(scores[i][:5], rel_wikis[i]) for i in range(len(rel_wikis))]
    r_at_10 = [calculate_recall(scores[i][:10], rel_wikis[i]) for i in range(len(rel_wikis))]
    r_at_20 = [calculate_recall(scores[i][:20], rel_wikis[i]) for i in range(len(rel_wikis))]
    
    # Calculate average recall
    r_at_5_avg = sum(r_at_5) / len(r_at_5) if r_at_5 else 0
    r_at_10_avg = sum(r_at_10) / len(r_at_10) if r_at_10 else 0
    r_at_20_avg = sum(r_at_20) / len(r_at_20) if r_at_20 else 0
    
    print(f'R@5: {r_at_5_avg:.4f}, R@10: {r_at_10_avg:.4f}, R@20: {r_at_20_avg:.4f}')
    
    # Calculate nDCG for non-proactive
    ndcg_results = []
    for i, query_scores in enumerate(scores):
        retrieved = [{'retrieved_docs': [doc_id for doc_id, _ in query_scores], 
                      'correct_docs': rel_wikis[i]}]
        ndcg_results.append(calculate_npdcg(retrieved, [5, 10, 20]))
    
    # Average nDCG results
    ndcg_avg = {}
    for cutoff in [5, 10, 20]:
        ndcg_avg[cutoff] = sum(result[cutoff] for result in ndcg_results) / len(ndcg_results) if ndcg_results else 0
    
    print(f'NDCG@5: {ndcg_avg[5]:.4f}, NDCG@10: {ndcg_avg[10]:.4f}, NDCG@20: {ndcg_avg[20]:.4f}')
    
    print('---')
    
    # Run proactive evaluation
    print('running proactive evaluation...')
    cuttoffs = [5, 10, 20]
    npdcg = []
    
    # Process a subset of conversations for demonstration
    for conv_id, subqueries in enumerate(queries_proactive[:10]):
        print(f"Processing conversation {conv_id + 1}/10...")
        scores = []
        
        for i in range(len(subqueries)):
            if queries_proactive_acts[conv_id][i] == 1:
                print(f"  Running search for subquery {i + 1}/{len(subqueries)}...")
                search_result = model.search([subqueries[i]], topk=topk, proactive=True)[0]
                scores.append(search_result)
            else:
                scores.append([])
        
        retrieved = []
        for thread_id, subquery_scores in enumerate(scores):
            retrieved_docs = [doc_id for doc_id, score in subquery_scores]
            correct_docs = rel_wikis_proactive[conv_id][thread_id]
            retrieved.append({'retrieved_docs': retrieved_docs, 'correct_docs': correct_docs})
        
        npdcg.append(calculate_npdcg(retrieved, cuttoffs))
    
    # Calculate average npdcg per cutoff
    npdcg_avg = {c: sum([npdcg[i][c] for i in range(len(npdcg))]) / len(npdcg) if npdcg else 0 for c in cuttoffs}
    print("Average proactive nDCG per cutoff:", npdcg_avg)


if __name__ == "__main__":
    main()
