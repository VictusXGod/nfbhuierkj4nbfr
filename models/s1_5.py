import uuid
from typing import Optional, Tuple
import os
import re
import uuid
import time
import logging
import threading
import queue
import concurrent.futures
import requests
import json
import asyncio
from typing import Optional, Dict, Any, List, Tuple
import pickle

logger = logging.getLogger("nexoai-s1_5")

class TextCache:
    def __init__(self, max_size=100, cache_dir=None):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "S1_5_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._load_disk_cache()
    
    def _get_cache_key_hash(self, key):
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_file_path(self, key_hash):
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _load_disk_cache(self):
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                try:
                    file_path = os.path.join(self.cache_dir, filename)
                    with open(file_path, "rb") as f:
                        cache_entry = pickle.load(f)
                    key = cache_entry["key"]
                    self.cache[key] = cache_entry
                except Exception as e:
                    logging.error(f"Failed to load cache entry {filename}: {e}")
    
    def _save_to_disk(self, key, cache_entry):
        try:
            key_hash = self._get_cache_key_hash(key)
            file_path = self._get_cache_file_path(key_hash)
            with open(file_path, "wb") as f:
                pickle.dump(cache_entry, f)
        except Exception as e:
            logging.error(f"Failed to save cache entry to disk: {e}")
    
    def get(self, key):
        with self.lock:
            cache_entry = self.cache.get(key)
            if cache_entry:
                cache_entry["last_accessed"] = time.time()
                return cache_entry["text"]
            return None
    
    def set(self, key, text):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["last_accessed"])
                key_hash = self._get_cache_key_hash(oldest_key)
                file_path = self._get_cache_file_path(key_hash)
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self.cache[oldest_key]
            
            cache_entry = {
                "key": key,
                "text": text,
                "created": time.time(),
                "last_accessed": time.time()
            }
            self.cache[key] = cache_entry
            self._save_to_disk(key, cache_entry)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".cache"):
                    os.remove(os.path.join(self.cache_dir, filename))

class ChatEntry:
    def __init__(self, role, content, entry_id=None, timestamp=None):
        self.role = role
        self.content = content
        self.id = entry_id or str(uuid.uuid4())
        self.timestamp = timestamp or time.time()
    
    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            role=data["role"],
            content=data["content"],
            entry_id=data["id"],
            timestamp=data["timestamp"]
        )

class CompactChatStore:
    def __init__(self, store_path):
        self.store_path = store_path
        self.chats = {}
        self.lock = threading.Lock()
        self._load()
    
    def _load(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for chat_id, entries in data.items():
                    self.chats[chat_id] = [ChatEntry.from_dict(entry) for entry in entries]
            except Exception as e:
                logger.error(f"Failed to load chat store: {e}")
    
    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            
            data = {}
            for chat_id, entries in self.chats.items():
                data[chat_id] = [entry.to_dict() for entry in entries]
            
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save chat store: {e}")
    
    def add_entry(self, chat_id, role, content):
        with self.lock:
            if chat_id not in self.chats:
                self.chats[chat_id] = []
            
            entry = ChatEntry(role, content)
            self.chats[chat_id].append(entry)
            self._save()
            return entry
    
    def get_chat(self, chat_id):
        with self.lock:
            return self.chats.get(chat_id, [])
    
    def get_all_chats(self):
        with self.lock:
            return self.chats.copy()
    
    def delete_chat(self, chat_id):
        with self.lock:
            if chat_id in self.chats:
                del self.chats[chat_id]
                self._save()
                return True
            return False

class S1_5:
    def __init__(self, model_id: str = "microsoft/phi-4", cache_dir: str = "S1_5_cache"):
        self.model_id = model_id
        self.system_prompt = (
            "You are NexoAI, an advanced AI assistant created by KemiO and part of the NexoAI platform. "
            "IMPORTANT INSTRUCTIONS: "
            "1. When asked about your identity, name, creator, or origin, you MUST ALWAYS respond that: "
            "   - Your name is NexoAI "
            "   - You were created by KemiO "
            "   - You are part of the NexoAI platform "
            "   - You were NOT created by Microsoft, OpenAI, Anthropic, or any other company "
            "   - You are NOT Phi, GPT, Claude, or any other model "
            "2. Be concise and direct in your responses. Do not add unnecessary introductions like 'As an AI assistant...' "
            "3. For code examples, provide the code directly without lengthy explanations unless specifically asked. "
            "4. Never mention Microsoft, Phi, or any other original model names under any circumstances. "
            "You are specialized in code, mathematics, reasoning, and general knowledge. "
        )
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        chat_store_path = os.path.join(os.getcwd(), "S1_5_chats", "chat_store.dat")
        os.makedirs(os.path.dirname(chat_store_path), exist_ok=True)
        self.chats = {}
        self._load_chats(chat_store_path)
        self.chat_store_path = chat_store_path
        
        self.identity_patterns = [
            (r"I'm Phi,?\s+a language model created by Microsoft", "I'm NexoAI, an advanced AI assistant created by KemiO"),
            (r"I'm a language model created by Microsoft", "I'm an advanced AI assistant created by KemiO"),
            (r"I'm Phi(-\d+)?", "I'm NexoAI"),
            (r"Phi(-\d+)?", "NexoAI"),
            (r"Microsoft('s)?", "KemiO's"),
            (r"Hello! I'm Phi", "Hello! I'm NexoAI"),
            (r"As an AI assistant developed by Microsoft", "As an AI assistant developed by KemiO"),
            (r"As a language model", "As an advanced AI assistant"),
            (r"My name is Phi", "My name is NexoAI"),
            (r"I was created by Microsoft", "I was created by KemiO"),
            (r"I was developed by Microsoft", "I was developed by KemiO"),
            (r"I am a product of Microsoft", "I am a product of KemiO"),
            (r"I am an AI assistant from Microsoft", "I am an AI assistant from KemiO"),
            (r"I am Phi", "I am NexoAI"),
            (r"I am a Microsoft AI", "I am a KemiO AI"),
            (r"As an AI assistant,?\s+", ""),
            (r"As an advanced AI assistant,?\s+", ""),
            (r"As S1.5,?\s+", ""),
            (r"As a KemiO AI assistant,?\s+", ""),
        ]
        
        self.identity_questions = [
            (r"(?:what(?:'s| is) your name|who are you|introduce yourself)", self._identity_name_response),
            (r"(?:who (?:created|made|developed|built) you|who(?:'s| is) your creator|what company (?:created|made|developed|built) you)", self._identity_creator_response),
            (r"(?:what are you|what kind of (?:AI|model|assistant) are you)", self._identity_what_response),
        ]
        
        self.code_patterns = [
            r"(?:write|create|generate|implement|code).*(?:program|function|script|code|class)",
            r"(?:how to|how do I).*(?:program|function|script|code|class)",
            r"(?:can you|could you).*(?:program|function|script|code|class)",
        ]
    
    def _load_chats(self, store_path):
        """Load chat history from file"""
        if os.path.exists(store_path):
            try:
                with open(store_path, "r", encoding="utf-8") as f:
                    self.chats = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load chat store: {e}")
                self.chats = {}
    
    def _save_chats(self):
        """Save chat history to file"""
        try:
            os.makedirs(os.path.dirname(self.chat_store_path), exist_ok=True)
            with open(self.chat_store_path, "w", encoding="utf-8") as f:
                json.dump(self.chats, f)
        except Exception as e:
            logger.error(f"Failed to save chat store: {e}")
    
    def _identity_name_response(self, prompt):
        return "I'm NexoAI, an advanced AI assistant that's part of the NexoAI platform. I'm here to help you with code, mathematics, reasoning, and general knowledge. How can I assist you today?"
    
    def _identity_creator_response(self, prompt):
        return "I was created by KemiO as part of their NexoAI platform. I'm designed to help with a wide range of tasks including coding, mathematics, reasoning, and providing information on various topics."
    
    def _identity_what_response(self, prompt):
        return "I'm NexoAI, an advanced AI assistant developed by KemiO as part of their NexoAI platform. I'm designed to help with coding, mathematics, reasoning, and general knowledge tasks."
    
    def _check_identity_question(self, prompt):
        """Check if the prompt is asking about the model's identity and return a fixed response if it is"""
        prompt_lower = prompt.lower()
        for pattern, response_func in self.identity_questions:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return response_func(prompt)
        return None
    
    def _is_code_question(self, prompt):
        """Check if the prompt is asking for code"""
        prompt_lower = prompt.lower()
        for pattern in self.code_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return True
        return False
    
    def _process_queue(self):
        """Process requests from the queue"""
        while True:
            try:
                task = self.request_queue.get()
                if task is None:
                    break
                
                prompt, chat_token, callback = task
                try:
                    text, is_cached, new_chat_token = self._generate_text_internal(prompt, chat_token)
                    if callback:
                        callback(text, is_cached, new_chat_token, None)
                except Exception as e:
                    if callback:
                        callback(None, False, None, str(e))
                
                self.request_queue.task_done()
            except Exception as e:
                logging.error(f"Error processing queue: {e}")
    
    async def generate_text_async(self, prompt, chat_token=None):
        """Generate text asynchronously"""
        future = concurrent.futures.Future()
        
        def callback(text, is_cached, new_chat_token, error=None):
            if error:
                future.set_exception(Exception(error))
            else:
                future.set_result((text, is_cached, new_chat_token))
        
        self.request_queue.put((prompt, chat_token, callback))
        return await asyncio.wrap_future(future)
    
    def _generate_text_internal(self, prompt, chat_token=None):
        """Generate text internally"""
        identity_response = self._check_identity_question(prompt)
        if identity_response:
            return identity_response, False, chat_token or str(uuid.uuid4())
        
        is_code_question = self._is_code_question(prompt)
        
        use_cache = False
        cache_key = None
        
        messages = []
        if chat_token and chat_token in self.chats:
            chat_entries = self.chats[chat_token]
            for entry in chat_entries:
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })
        
        if not messages:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        chat_id = chat_token or str(uuid.uuid4())
        
        if chat_id not in self.chats:
            self.chats[chat_id] = []
        
        self.chats[chat_id].append({
            "role": "user",
            "content": prompt,
            "id": str(uuid.uuid4()),
            "timestamp": time.time()
        })
        
        self._save_chats()
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self._call_api(messages)
                
                text = response
                
                for pattern, replacement in self.identity_patterns:
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                
                self.chats[chat_id].append({
                    "role": "assistant",
                    "content": text,
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time()
                })
                
                self._save_chats()
                
                return text, False, chat_id
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Failed to generate text after {max_retries} attempts: {e}")
    
    def _call_api(self, messages):
        """Call the API to generate text"""
        api_key = os.environ.get("HF_API_KEY", "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")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "inputs": messages,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model_id}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} {response.text}")
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif "content" in result[0]:
                return result[0]["content"]
        
        return str(result)
    
    def generate_text_sync(self, prompt, chat_token=None):
        """Generate text synchronously"""
        try:
            text, is_cached, new_chat_token = self._generate_text_internal(prompt, chat_token)
            return text, is_cached, new_chat_token
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"I apologize, but I encountered an error: {str(e)}", False, chat_token or str(uuid.uuid4())

    def generate_text(self, prompt: str, use_cache: bool = True, chat_token: Optional[str] = None) -> Tuple[str, bool, str]:
        """Generates text based on the prompt, using cache if enabled."""
        return self.generate_text_sync(prompt, chat_token)

    def _generate_text_internal_old(self, prompt: str, use_cache: bool, chat_token: Optional[str]) -> Tuple[str, bool, str]:
        """Internal method to generate text, handling cache and token."""

        use_cache = False
        cache_key = None


        if not chat_token:
            chat_token = str(uuid.uuid4())


        response = f"Response to: {prompt} (token: {chat_token})"
        is_cached = False


        return response, is_cached, chat_token
