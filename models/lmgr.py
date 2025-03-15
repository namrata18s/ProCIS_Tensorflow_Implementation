import json
import os
import openai
import re
import tensorflow as tf
import time
from tqdm import tqdm

# Import from the reimplemented TensorFlow-based DenseRetriever
from .dr_tensorflow import DenseRetriever

def chatgpt_api(prompt, model_name='gpt-4-turbo-preview'):
    openai.api_key = "YOUR-API-KEY"
    openai.api_base = "https://api.openai.com/v1"
    response = openai.ChatCompletion.create(
        model=model_name,
        max_tokens=3000,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    used_tokens = response["usage"]["total_tokens"]
    return response["choices"][0]["message"]["content"].strip(), used_tokens

def custom_chatgpt_api(prompt):
    openai.api_key = "YOUR-API-KEY"
    openai.api_base = "https://api.deepinfra.com/v1/openai"
    gpt_model_name = 'openchat/openchat_3.5'
    response = openai.ChatCompletion.acreate(
        model=gpt_model_name,
        max_tokens=3000,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    used_tokens = response["usage"]["total_tokens"]
    return response["choices"][0]["message"]["content"].strip(), used_tokens


def remove_enumeration(s):
    return re.sub(r'^\d+\.\s', '', s).strip()


def parse_json(data):
    try:
        return json.loads(data)
    except:
        prompt = f"fix the json below, return only valid json in your response like this {{\"title\":..., \"description\":...}}, nothing else:\n\n{data}\n\n"
        response, used_tokens = chatgpt_api(prompt)
        return json.loads(response)


class LMGR:
    def __init__(self):
        # Set TensorFlow memory growth to avoid allocating all GPU memory
        self._configure_tensorflow()
        self.load_corpus()
        self.load_retriever()

    def _configure_tensorflow(self):
        """Configure TensorFlow to use memory growth and set logging level"""
        # Set TensorFlow logging level
        tf.get_logger().setLevel('ERROR')

        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth needs to be set before GPUs have been initialized
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s), configured with memory growth")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"Error configuring GPU memory growth: {e}")

    def load_corpus(self):
        print("Loading corpus...")
        self.docs = []
        self.wikis = []
        with open("collection.jsonl") as f:
            for line in tqdm(f):
                d = json.loads(line)
                self.docs.append(d["wiki"] + ': ' + d["contents"][:60] + '...')
                self.wikis.append(d["wiki"])

    def load_retriever(self):
        print('Loading model...')
        # Note the dimension is 384 for MiniLM-L6-v2 (smaller than the default 768)
        self.retriever = DenseRetriever('sentence-transformers/all-MiniLM-L6-v2', dim=384)
        
        print('Indexing...')
        if os.path.exists('embeddings.pkl'):
            print('Loading cached embeddings...')
            self.retriever.create_index_from_vectors('embeddings.pkl')
        else:
            print('Creating embeddings...')
            # Use TensorFlow-based encoding
            self.retriever.create_index_from_documents(self.docs)
            self.retriever.save_index(vectors_path='embeddings.pkl')

    def search(self, conversations, topk=20, candidates=3, proactive=False, batch_size=16):
        results = []
        
        # Process conversations in batches to optimize GPU utilization
        for i in tqdm(range(0, len(conversations), batch_size)):
            batch = conversations[i:i+batch_size]
            batch_results = self._process_batch(batch, topk, candidates, proactive)
            results.extend(batch_results)
            
        return results
    
    def _process_batch(self, conversations, topk, candidates, proactive):
        batch_results = []
        
        for conversation in conversations:
            comments = conversation["thread"]
            user_to_id = {}
            uid = 1
            for c in comments:
                if c["author"] not in user_to_id:
                    user_to_id[c["author"]] = uid
                    uid += 1
            
            reg_exp = r'\n+'
            comments_str = "\n".join([f"user {user_to_id[c['author']]}: {re.sub(reg_exp, ' ', c['text'])}" for c in comments])

            if proactive:
                prompt = f"\npost title: {conversation['post']['title']}\n\npost text: {conversation['post']['text']}\n\ncomments:\n{comments_str}\n\n" \
                f"based on the last comment, or the post if no comments shown, give up to {topk} wikipedia articles that provide " \
                f"useful information and answer potential questions, ambiguities or misunderstandings or they provide some relevant context, only show articles that add very useful context, you don't need to generate all {topk}, you might also give 0 results if not necesary, " \
                "ordered by relevance and use jsonl format, each line has a json with title and description, nothing else in your response:\n\n{\"title\":..., \"description\":...}\n{\"title\":..., \"description\":...}...\n\ndescription is the first sentence of the wikipedia article\n\n"
            else:
                prompt = f"\npost title: {conversation['post']['title']}\n\npost text: {conversation['post']['text']}\n\ncomments:\n{comments_str}\n\n" \
                f"based on the content of the post and the comments, give up to {topk} wikipedia articles that provide " \
                f"useful information and answer potential questions, ambiguities or misunderstandings from the conversation or they provide some relevant context, only show articles that add very useful context, you don't need to generate all {topk} if you are not confident, you might also give 0 results, " \
                "ordered by relevance and use jsonl format, each line has a json with title and description, nothing else in your response:\n\n{\"title\":..., \"description\":...}\n{\"title\":..., \"description\":...}...\n\ndescription is the first sentence of the wikipedia article\n\n"

            # Get candidate documents from the LLM
            try:
                response, used_tokens = custom_chatgpt_api(prompt)
            except Exception as e:
                print(f"Error with custom API: {e}, falling back to standard API")
                response, used_tokens = chatgpt_api(prompt, model_name='gpt-3.5-turbo')

            # Clean and parse the response
            response = response.replace('\n\n', '\n')
            generated_candidate_docs = response.split('\n')
            generated_candidate_docs = [parse_json(remove_enumeration(d)) for d in generated_candidate_docs if d]
            generated_candidate_docs = [f"{d['title']}: {d['description']}" for d in generated_candidate_docs]
            
            # Process each candidate with tensor-based similarity search
            final_docs = []
            added_docs = set()
            
            # Batch retrieval for efficiency
            if generated_candidate_docs:
                # Process all candidates at once to reduce API calls
                all_retrieved_candidates = self._batch_retrieve_candidates(generated_candidate_docs, candidates)
                
                # For each candidate, verify and add to final list
                for idx, doc in enumerate(generated_candidate_docs):
                    retrieved_candidates = all_retrieved_candidates[idx]
                    if not retrieved_candidates:
                        continue
                        
                    retrieved_candidates_str = "\n".join([
                        f"{i+1}. {self.docs[cand_doc[0]]}" 
                        for i, cand_doc in enumerate(retrieved_candidates)
                    ])
                    
                    prompt = f"Pick the candidate that describes the same concept with the given document, return only the number "\
                             f"in your response, nothing else, if none of them then return 0:\n\ndocument: {doc}\n\ncandidates: {retrieved_candidates_str}\n\n"
                    
                    try:
                        response, used_tokens = custom_chatgpt_api(prompt)
                    except Exception as e:
                        print(f"Error with custom API: {e}, falling back to standard API")
                        response, used_tokens = chatgpt_api(prompt, model_name='gpt-3.5-turbo')
                    
                    # Extract candidate ID
                    cand_id = 0
                    try:
                        cand_id = int(re.sub(r'\D', '', response).strip())
                    except:
                        pass

                    if cand_id > len(retrieved_candidates) or cand_id < 0:
                        cand_id = 0

                    # Add valid candidate to final documents
                    if cand_id > 0:
                        retrieved_candidate = retrieved_candidates[cand_id-1]
                        if retrieved_candidate[0] not in added_docs:
                            added_docs.add(retrieved_candidate[0])
                            final_docs.append(retrieved_candidate)

            batch_results.append(final_docs)
            
            # Add a small delay to avoid API rate limits
            time.sleep(0.1)
            
        return batch_results
    
    def _batch_retrieve_candidates(self, docs, candidates):
        """Retrieve candidates for multiple documents in one batch for efficiency"""
        # Use the TensorFlow-based retriever to search for candidates
        all_results = self.retriever.search(docs, topk=candidates)
        return all_results


class TensorflowOptimizedLMGR(LMGR):
    """An optimized version of LMGR that uses TensorFlow-specific optimizations"""
    
    def __init__(self, cache_embeddings=True):
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {}
        super().__init__()
    
    def _configure_tensorflow(self):
        """Configure TensorFlow with additional optimizations"""
        super()._configure_tensorflow()
        
        # Enable mixed precision for faster computation on compatible GPUs
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set to: {policy.name}")
        
        # Enable XLA compilation for faster execution
        tf.config.optimizer.set_jit(True)
        print("XLA compilation enabled")
    
    def search(self, conversations, topk=20, candidates=3, proactive=False, batch_size=16):
        """Optimized search implementation using TensorFlow-specific features"""
        # Pre-process all conversations to extract queries
        all_queries = []
        for conversation in conversations:
            comments = conversation["thread"]
            user_to_id = {}
            uid = 1
            for c in comments:
                if c["author"] not in user_to_id:
                    user_to_id[c["author"]] = uid
                    uid += 1
            
            reg_exp = r'\n+'
            comments_str = "\n".join([f"user {user_to_id[c['author']]}: {re.sub(reg_exp, ' ', c['text'])}" for c in comments])
            
            # Create query for LLM processing
            if proactive:
                query = f"post: {conversation['post']['title']} {conversation['post']['text']} comments: {comments_str}"
            else:
                query = f"post: {conversation['post']['title']} {conversation['post']['text']} comments: {comments_str}"
                
            all_queries.append({
                "query": query,
                "conversation": conversation,
                "comments_str": comments_str
            })
        
        # Process in batches
        results = []
        for i in tqdm(range(0, len(all_queries), batch_size)):
            batch = all_queries[i:i+batch_size]
            batch_results = self._process_optimized_batch(batch, topk, candidates, proactive)
            results.extend(batch_results)
            
        return results
    
    def _process_optimized_batch(self, queries, topk, candidates, proactive):
        """Process a batch of queries with optimized TensorFlow operations"""
        batch_results = []
        
        for query_data in queries:
            conversation = query_data["conversation"]
            comments_str = query_data["comments_str"]
            
            if proactive:
                prompt = f"\npost title: {conversation['post']['title']}\n\npost text: {conversation['post']['text']}\n\ncomments:\n{comments_str}\n\n" \
                f"based on the last comment, or the post if no comments shown, give up to {topk} wikipedia articles that provide " \
                f"useful information and answer potential questions, ambiguities or misunderstandings or they provide some relevant context, only show articles that add very useful context, you don't need to generate all {topk}, you might also give 0 results if not necesary, " \
                "ordered by relevance and use jsonl format, each line has a json with title and description, nothing else in your response:\n\n{\"title\":..., \"description\":...}\n{\"title\":..., \"description\":...}...\n\ndescription is the first sentence of the wikipedia article\n\n"
            else:
                prompt = f"\npost title: {conversation['post']['title']}\n\npost text: {conversation['post']['text']}\n\ncomments:\n{comments_str}\n\n" \
                f"based on the content of the post and the comments, give up to {topk} wikipedia articles that provide " \
                f"useful information and answer potential questions, ambiguities or misunderstandings from the conversation or they provide some relevant context, only show articles that add very useful context, you don't need to generate all {topk} if you are not confident, you might also give 0 results, " \
                "ordered by relevance and use jsonl format, each line has a json with title and description, nothing else in your response:\n\n{\"title\":..., \"description\":...}\n{\"title\":..., \"description\":...}...\n\ndescription is the first sentence of the wikipedia article\n\n"
            
            # The rest of the processing is similar to the basic implementation
            try:
                response, used_tokens = custom_chatgpt_api(prompt)
            except:
                response, used_tokens = chatgpt_api(prompt, model_name='gpt-3.5-turbo')
                
            # Process results as in the original method
            response = response.replace('\n\n', '\n')
            generated_candidate_docs = response.split('\n')
            generated_candidate_docs = [parse_json(remove_enumeration(d)) for d in generated_candidate_docs if d]
            generated_candidate_docs = [f"{d['title']}: {d['description']}" for d in generated_candidate_docs]
            
            final_docs = []
            added_docs = set()
            
            # Efficient batch retrieval
            if generated_candidate_docs:
                # Cache embeddings if enabled
                if self.cache_embeddings:
                    cached_docs = []
                    new_docs = []
                    doc_indices = []
                    
                    for i, doc in enumerate(generated_candidate_docs):
                        if doc in self.embedding_cache:
                            cached_docs.append(self.embedding_cache[doc])
                            doc_indices.append(i)
                        else:
                            new_docs.append(doc)
                            
                    # Compute new embeddings
                    if new_docs:
                        new_results = self.retriever.search(new_docs, topk=candidates)
                        
                        # Update cache
                        for i, doc in enumerate(new_docs):
                            self.embedding_cache[doc] = new_results[i]
                            
                    # Combine cached and new results
                    all_retrieved_candidates = [None] * len(generated_candidate_docs)
                    
                    # Add cached results
                    for i, idx in enumerate(doc_indices):
                        all_retrieved_candidates[idx] = cached_docs[i]
                        
                    # Add new results
                    new_idx = 0
                    for i in range(len(generated_candidate_docs)):
                        if all_retrieved_candidates[i] is None:
                            all_retrieved_candidates[i] = new_results[new_idx]
                            new_idx += 1
                else:
                    # Just do the search without caching
                    all_retrieved_candidates = self.retriever.search(generated_candidate_docs, topk=candidates)
                
                # Process candidates
                for idx, doc in enumerate(generated_candidate_docs):
                    retrieved_candidates = all_retrieved_candidates[idx]
                    
                    if not retrieved_candidates:
                        continue
                        
                    retrieved_candidates_str = "\n".join([
                        f"{i+1}. {self.docs[cand_doc[0]]}" 
                        for i, cand_doc in enumerate(retrieved_candidates)
                    ])
                    
                    prompt = f"Pick the candidate that describes the same concept with the given document, return only the number "\
                             f"in your response, nothing else, if none of them then return 0:\n\ndocument: {doc}\n\ncandidates: {retrieved_candidates_str}\n\n"
                    
                    try:
                        response, used_tokens = custom_chatgpt_api(prompt)
                    except:
                        response, used_tokens = chatgpt_api(prompt, model_name='gpt-3.5-turbo')
                    
                    cand_id = 0
                    try:
                        cand_id = int(re.sub(r'\D', '', response).strip())
                    except:
                        pass

                    if cand_id > len(retrieved_candidates) or cand_id < 0:
                        cand_id = 0

                    if cand_id > 0:
                        retrieved_candidate = retrieved_candidates[cand_id-1]
                        if retrieved_candidate[0] not in added_docs:
                            added_docs.add(retrieved_candidate[0])
                            final_docs.append(retrieved_candidate)

            batch_results.append(final_docs)
            
        return batch_results


if __name__ == '__main__':
    # Choose which implementation to use
    use_optimized = True
    
    if use_optimized:
        lmgr = TensorflowOptimizedLMGR(cache_embeddings=True)
    else:
        lmgr = LMGR()
        
    with open('test.jsonl') as f:
        conversations = [json.loads(line) for line in f]
        # Test with just the first conversation
        results = lmgr.search(conversations[:1])
        print(f"Found {len(results[0])} relevant documents for the first conversation")
