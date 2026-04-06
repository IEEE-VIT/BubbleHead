import re
import numpy as np
from typing import List, Dict, Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class HybridChunker:
    """
    Hybrid (Semantic + Recursive) Chunker for RAG Pipeline.
    
    This chunker aims to group sentences together based on their semantic meaning 
    (how similar they are) as long as they fit within a specific token size. 
    It prevents breaking sentences in half and ensures related ideas stay together.
    """
    
    def __init__(
        self, 
        semantic_threshold: float = 0.75, 
        max_tokens: int = 512, 
        overlap_tokens: int = 50,
        embedding_model: str = "mistral:7b"
    ):
        # The threshold for grouping sentences. 
        # A higher number means sentences must be more similar to stay in the same chunk.
        self.semantic_threshold = semantic_threshold
        
        # Max capacity for a chunk
        self.max_tokens = max_tokens
        
        # How much context to carry over from the previous chunk when starting a new one
        self.overlap_tokens = overlap_tokens
        
        # We estimate that 1 token is roughly 4 characters in English.
        # This allows us to track length cleanly without needing external tokenizers (like tiktoken)
        self.max_chars = self.max_tokens * 4
        self.overlap_chars = self.overlap_tokens * 4
        
        # We use Ollama Embeddings to convert sentences into numerical vectors. 
        # This allows us to calculate how "similar" they are mathematically.
        self.embedder = OllamaEmbeddings(model=embedding_model)
        
        # If a single sentence is incredibly long (larger than our maximum budget),
        # we fall back to a basic character text splitter to forcefully cut it.
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chars,
            chunk_overlap=self.overlap_chars
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of document sections from a parser and yields properly sized chunks.
        
        Expected input is a list of dictionaries. Example:
        [
            {
                "text": "Raw text content here...",
                "metadata": {
                    "source": "my_document.pdf",
                    "type": "text", # can also be "code", "table", "slide"
                    "page": 1
                }
            }
        ]
        """
        all_chunks = []
        chunk_index = 0
        
        for doc in documents:
            # We copy the original document's metadata to ensure chunks inherit it
            meta = doc.get("metadata", {})
            doc_type = meta.get("type", "text")
            text = doc.get("text", "")
            
            # --- SPECIAL HANDLING ---
            # If the parser identified this block as code, a table, or a slide, we DO NOT split it. 
            # We want to keep tables and code blocks fully intact for the LLM to understand them properly.
            if doc_type in ["code", "table", "slide"]:
                meta_copy = meta.copy()
                meta_copy["chunk_index"] = chunk_index
                all_chunks.append({
                    "text": text,
                    "metadata": meta_copy
                })
                chunk_index += 1
                continue
                
            # --- TEXT CHUNKING ---
            # 1. Break text into paragraphs first (separated by double newlines)
            paragraphs = re.split(r'\n\s*\n', text.strip())
            sentences = []
            
            for p in paragraphs:
                # 2. Break paragraphs into sentences. 
                # This regular expression splits on punctuation (., !, ?) while keeping the punctuation.
                sents = re.split(r'(?<=[.!?])\s+', p.strip())
                sentences.extend([s.strip() for s in sents if s.strip()])
                
            # If there's no actual text, skip
            if not sentences:
                continue
                
            # 3. Calculate embeddings for all sentences at once.
            # We will use these numerical representations to group related sentences together.
            embeddings = self.embedder.embed_documents(sentences)
            
            current_chunk = []
            current_len = 0
            
            # Iterate through each sentence and decide whether to group it with the current chunk
            for i in range(len(sentences)):
                sent = sentences[i]
                sent_len = len(sent)
                
                # EDGE CASE: What if a single sentence is gigantically long? 
                # We have to recursively split it so we don't crash our LLM's context limits.
                if sent_len > self.max_chars:
                    # First, save whatever we've collected in the current chunk so far
                    if current_chunk:
                        meta_copy = meta.copy()
                        meta_copy["chunk_index"] = chunk_index
                        all_chunks.append({
                            "text": " ".join(current_chunk),
                            "metadata": meta_copy
                        })
                        chunk_index += 1
                        
                        # Reset the chunk tracker
                        current_chunk = []
                        current_len = 0
                        
                    # Use the Langchain fallback splitter to cut the massive sentence down
                    sub_chunks = self.char_splitter.split_text(sent)
                    for j, sc in enumerate(sub_chunks):
                        if j < len(sub_chunks) - 1:
                            meta_copy = meta.copy()
                            meta_copy["chunk_index"] = chunk_index
                            all_chunks.append({
                                "text": sc,
                                "metadata": meta_copy
                            })
                            chunk_index += 1
                        else:
                            # Keep the very last sub-chunk to start the next semantic grouping phase
                            current_chunk = [sc]
                            current_len = len(sc)
                    continue

                # NORMAL GROUPING
                if current_chunk and i > 0:
                    sent_emb = embeddings[i]
                    prev_emb = embeddings[i-1]
                    
                    # Calculate Cosine Similarity: How mathematically similar is this sentence to the previous one?
                    sim = np.dot(sent_emb, prev_emb) / (np.linalg.norm(sent_emb) * np.linalg.norm(prev_emb))
                    
                    # We break the chunk and "finalize" it if:
                    # 1. The topic has shifted drastically (similarity < threshold) OR
                    # 2. Adding this sentence would push us past our character size limit
                    if sim < self.semantic_threshold or (current_len + sent_len + 1 > self.max_chars):
                        # Save the completed chunk
                        meta_copy = meta.copy()
                        meta_copy["chunk_index"] = chunk_index
                        all_chunks.append({
                            "text": " ".join(current_chunk),
                            "metadata": meta_copy
                        })
                        chunk_index += 1
                        
                        # --- OVERLAP LOGIC ---
                        # Instead of starting entirely fresh, pull the last few sentences from 
                        # the finished chunk (up to our overlap limit). This creates a bridge of context.
                        overlap_chunk = []
                        overlap_len = 0
                        for prev_s in reversed(current_chunk):
                            if overlap_len + len(prev_s) <= self.overlap_chars:
                                overlap_chunk.insert(0, prev_s)  # insert at front
                                overlap_len += len(prev_s) + 1   # +1 for space between words
                            else:
                                break
                        
                        current_chunk = overlap_chunk
                        current_len = sum(len(s) + 1 for s in current_chunk)

                # Add the sentence to our running list
                current_chunk.append(sent)
                current_len += sent_len + 1
                
            # FLUSH: Once we run out of sentences, save whatever is leftover in the buffer
            if current_chunk:
                meta_copy = meta.copy()
                meta_copy["chunk_index"] = chunk_index
                all_chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": meta_copy
                })
                chunk_index += 1
                
        return all_chunks
