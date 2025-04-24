import os
import time
import struct
import zlib
import json
import logging
import threading
import uuid
import base64
import hashlib
from typing import Dict, List, Tuple, Optional, Any

class ChatEntry:
    """Represents a single chat message with metadata"""
    def __init__(self, chat_id: str, message_id: str, role: str, content: str, timestamp: float = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatEntry':
        return cls(
            chat_id=data["chat_id"],
            message_id=data["message_id"],
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"]
        )

class CompactChatStore:
    """Efficient storage for chat history using a compact binary format"""
        
    MAGIC = b'NEXOCHAT'
    VERSION = 1
    HEADER_SIZE = 12 
    
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.lock = threading.RLock()
        self.index = {}  
        self.dirty = False
        self.memory_cache = {} 
        
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        
        if os.path.exists(store_path):
            self._load_store()
        else:
            self._initialize_store()
    
    def _initialize_store(self):
        """Create a new empty store file"""
        with open(self.store_path, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))
            
            index_data = zlib.compress(json.dumps({}).encode('utf-8'))
            index_size = len(index_data)
            f.write(struct.pack('<I', index_size))
            f.write(index_data)
    
    def _load_store(self):
        """Load an existing store file"""
        try:
            with open(self.store_path, 'rb') as f:
                magic = f.read(8)
                if magic != self.MAGIC:
                    raise ValueError(f"Invalid file format: {self.store_path}")
                
                version = struct.unpack('<I', f.read(4))[0]
                if version != self.VERSION:
                    raise ValueError(f"Unsupported version: {version}")
                
                index_size = struct.unpack('<I', f.read(4))[0]
                index_data = f.read(index_size)
                self.index = json.loads(zlib.decompress(index_data).decode('utf-8'))
        except Exception as e:
            logging.error(f"Error loading chat store: {e}")
            self._initialize_store()
    
    def _save_index(self):
        """Save the index to the store file"""
        with self.lock:
            if not self.dirty:
                return
            
            with open(self.store_path, 'r+b') as f:
                f.seek(self.HEADER_SIZE)
                
                index_data = zlib.compress(json.dumps(self.index).encode('utf-8'))
                index_size = len(index_data)
                f.write(struct.pack('<I', index_size))
                f.write(index_data)
            
            self.dirty = False
    
    def add_message(self, entry: ChatEntry) -> bool:
        """Add a new message to the store"""
        with self.lock:
            try:
                entry_data = json.dumps(entry.to_dict()).encode('utf-8')
                compressed_data = zlib.compress(entry_data)
                
                with open(self.store_path, 'r+b') as f:
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()
                    
                    entry_offset = file_size
                    entry_size = len(compressed_data)
                    
                    f.write(struct.pack('<I', entry_size))
                    f.write(compressed_data)
                    
                    if entry.chat_id not in self.index:
                        self.index[entry.chat_id] = []
                    
                    self.index[entry.chat_id].append({
                        "message_id": entry.message_id,
                        "offset": entry_offset,
                        "size": entry_size + 4, 
                        "timestamp": entry.timestamp
                    })
                    
                    self.dirty = True
                
                if entry.chat_id not in self.memory_cache:
                    self.memory_cache[entry.chat_id] = []
                self.memory_cache[entry.chat_id].append(entry)
                
                self._save_index()
                return True
            except Exception as e:
                logging.error(f"Error adding message: {e}")
                return False
    
    def get_chat_history(self, chat_id: str, limit: int = 10) -> List[ChatEntry]:
        """Get the most recent messages for a chat"""
        with self.lock:
            if chat_id in self.memory_cache:
                entries = self.memory_cache[chat_id]
                return entries[-limit:] if len(entries) > limit else entries
            
            if chat_id not in self.index:
                return []
            
            entries = sorted(
                self.index[chat_id], 
                key=lambda x: x["timestamp"]
            )
            
            if limit > 0:
                entries = entries[-limit:]
            
            result = []
            with open(self.store_path, 'rb') as f:
                for entry_info in entries:
                    try:
                        f.seek(entry_info["offset"])
                        entry_size = struct.unpack('<I', f.read(4))[0]
                        compressed_data = f.read(entry_size)
                        
                        entry_data = zlib.decompress(compressed_data)
                        entry_dict = json.loads(entry_data.decode('utf-8'))
                        
                        result.append(ChatEntry.from_dict(entry_dict))
                    except Exception as e:
                        logging.error(f"Error reading message {entry_info['message_id']}: {e}")
            
            self.memory_cache[chat_id] = result
            
            return result
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete all messages for a chat"""
        with self.lock:
            if chat_id not in self.index:
                return False
            
            del self.index[chat_id]
            self.dirty = True
            
            if chat_id in self.memory_cache:
                del self.memory_cache[chat_id]
            
            self._save_index()
            
            
            return True
    
    def compact(self) -> bool:
        """Compact the store by removing orphaned data blocks"""
        with self.lock:
            try:
                temp_path = f"{self.store_path}.tmp"
                
                with open(temp_path, 'wb') as temp_f:
                    temp_f.write(self.MAGIC)
                    temp_f.write(struct.pack('<I', self.VERSION))
                    
                    temp_f.write(struct.pack('<I', 0)) 
                    index_pos = temp_f.tell()
                    
                    new_index = {}
                    
                    with open(self.store_path, 'rb') as src_f:
                        for chat_id, entries in self.index.items():
                            new_index[chat_id] = []
                            
                            for entry_info in entries:
                                src_f.seek(entry_info["offset"])
                                entry_size = struct.unpack('<I', src_f.read(4))[0]
                                compressed_data = src_f.read(entry_size)
                                
                                new_offset = temp_f.tell()
                                temp_f.write(struct.pack('<I', entry_size))
                                temp_f.write(compressed_data)
                                
                                new_index[chat_id].append({
                                    "message_id": entry_info["message_id"],
                                    "offset": new_offset,
                                    "size": entry_size + 4,
                                    "timestamp": entry_info["timestamp"]
                                })
                    
                    index_data = zlib.compress(json.dumps(new_index).encode('utf-8'))
                    index_size = len(index_data)
                    
                    temp_f.seek(self.HEADER_SIZE)
                    temp_f.write(struct.pack('<I', index_size))
                    
                    temp_f.write(index_data)
                
                os.replace(temp_path, self.store_path)
                
                self.index = new_index
                self.dirty = False
                
                return True
            except Exception as e:
                logging.error(f"Error compacting chat store: {e}")
                return False
    
    def generate_chat_token(self) -> str:
        """Generate a unique token for a new chat"""
        random_bytes = os.urandom(16)
        token = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
        return token
    
    def get_chat_token(self, chat_id: str) -> str:
        """Get the token for an existing chat"""
        return chat_id
    
    def resume_chat_from_token(self, token: str) -> str:
        """Resume a chat from a token"""
        return token if token in self.index else None
    
    def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat exists"""
        return chat_id in self.index