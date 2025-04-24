from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import time
import threading
import queue
import logging
import asyncio
import io
import concurrent.futures
import torch
import hashlib
import pickle
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple

class ImageCache:
    def __init__(self, max_size=100, cache_dir=None):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "GenAZ_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._load_disk_cache()
    
    def _get_cache_key_hash(self, key):
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
                return cache_entry["image"]
        return None
    
    def set(self, key, image):
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
                "image": image,
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

class GenAZ:
    def __init__(self, api_key=None, model_id="stabilityai/stable-diffusion-3.5-large", temp_dir=None, cache_dir=None):
        self.api_key = api_key or os.environ.get("HF_API_KEY", "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")
        self.model_id = model_id
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key,
        )
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), "GenAZ_temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        self.cache = None
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
        
        self.font = None
        font_path = os.path.join(os.getcwd(), "assets", "typographica.ttf")
        
        try:
            if os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, 36)
            else:
                assets_dir = os.path.join(os.getcwd(), "assets")
                if not os.path.exists(assets_dir):
                    os.makedirs(assets_dir)
                logging.warning(f"Font file not found at {font_path}. Using default font.")
                self.font = ImageFont.load_default()
        except Exception as e:
            logging.error(f"Error loading font: {e}")
            self.font = ImageFont.load_default()
        
        self.logo = None
        logo_path = os.path.join(os.getcwd(), "assets", "logo-icon.png")
        try:
            if os.path.exists(logo_path):
                self.logo = Image.open(logo_path).convert("RGBA")
                logging.info(f"Logo loaded successfully from {logo_path}")
            else:
                logging.warning(f"Logo file not found at {logo_path}. Creating a placeholder logo.")
                self.logo = Image.new("RGBA", (100, 100), (255, 100, 100, 255))
                os.makedirs(os.path.dirname(logo_path), exist_ok=True)
                self.logo.save(logo_path)
        except Exception as e:
            logging.error(f"Error loading logo: {e}")
            self.logo = Image.new("RGBA", (100, 100), (255, 100, 100, 255))
    
    def _process_queue(self):
        while True:
            try:
                task = self.request_queue.get()
                if task is None:
                    break
                
                prompt, callback = task
                try:
                    image = self._generate_image_internal(prompt)
                    if callback:
                        callback(image, False, None)
                except Exception as e:
                    if callback:
                        callback(None, False, str(e))
                
                self.request_queue.task_done()
            except Exception as e:
                logging.error(f"Error processing queue: {e}")
    
    def _generate_image_internal(self, prompt):
        max_retries = 3
        retry_delay = 1
        
        random_seed = str(uuid.uuid4())
        modified_prompt = f"{prompt} (seed:{random_seed})"
        
        for attempt in range(max_retries):
            try:
                image = self.client.text_to_image(
                    prompt=modified_prompt,
                    model=self.model_id,
                    negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy",
                )
                
                return image
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Failed to generate image after {max_retries} attempts: {e}")
    
    def generate_image(self, prompt, callback=None):
        self.request_queue.put((prompt, callback))
    
    def generate_image_sync(self, prompt):
        try:
            image = self._generate_image_internal(prompt)
            return image, False  # Always return is_cached=False
        except Exception as e:
            raise Exception(f"Image generation failed: {e}")
    
    def add_watermark(self, image):
        img_with_watermark = image.copy().convert("RGBA")
        
        txt_layer = Image.new("RGBA", img_with_watermark.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        width, height = img_with_watermark.size
        
        text = "Made With NexoAI"
        
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(text, font=self.font)
        
        opacity = 153
        
        padding = 20  # Padding from the edge
        logo_size = text_height
        logo_spacing = 10
        
        logo_position = (
            width - text_width - logo_size - logo_spacing - padding,
            height - logo_size - padding
        )
        
        text_position = (
            width - text_width - padding,
            height - text_height - padding
        )
        
        draw.text(text_position, text, fill=(255, 255, 255, opacity), font=self.font)
        
        if self.logo:
            logo_resized = self.logo.resize((logo_size, logo_size), Image.LANCZOS)
            
            logo_array = np.array(logo_resized)
            if logo_array.shape[2] == 4:
                logo_array[..., 3] = logo_array[..., 3] * opacity // 255
            else:
                alpha = np.ones(logo_array.shape[:2], dtype=np.uint8) * opacity
                logo_array = np.dstack((logo_array, alpha))
            
            logo_transparent = Image.fromarray(logo_array)
            
            txt_layer.paste(logo_transparent, logo_position, logo_transparent)
        
        return Image.alpha_composite(img_with_watermark, txt_layer).convert("RGB")
    
    def save_image(self, image, output_path, add_watermark=True):
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if add_watermark:
            image = self.add_watermark(image)
        
        format = output_path.split('.')[-1].upper()
        if format == 'WEBP':
            image.save(output_path, format=format, quality=90)
        else:
            image.save(output_path, format=format)
        return output_path
    
    def get_image_path(self, query_token):
        for filename in os.listdir(self.temp_dir):
            if filename.startswith(f"{query_token}_") and filename.endswith(".webp"):
                return os.path.join(self.temp_dir, filename)
        return None
    
    def delete_image(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def cleanup_old_images(self, max_age=3600):
        current_time = time.time()
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logging.error(f"Failed to delete old file {file_path}: {e}")
    
    async def generate_image_async(self, prompt):
        loop = asyncio.get_event_loop()
        image, is_cached = await loop.run_in_executor(
            self.executor, 
            lambda: self.generate_image_sync(prompt)
        )
        return image, is_cached