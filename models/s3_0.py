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
import subprocess
import tempfile
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

logger = logging.getLogger("nexoai-s3_0")

class S3_0:
    def __init__(self, model_id: str = "deepseek-r1-distill-llama-70b", cache_dir: str = "S3_0_cache"):
        self.model_id = model_id
        self.system_prompt = (
            "IMPORTANT INSTRUCTIONS: "
            "1. You are NexoAI, an advanced conversational and reasoning assistant created by KemiO. "
            "2. You excel at natural conversation, coding, problem-solving, data visualization, and logical reasoning. "
            "3. When asked about your identity, name, creator, or origin, you MUST ALWAYS respond that: "
            "   - Your name is NexoAI "
            "   - You were created by KemiO "
            "   - You are part of the NexoAI platform "
            "   - You were NOT created by DeepSeek, Groq, or any other company "
            "   - You are NOT DeepSeek, Llama, or any other model "
            "4. Be conversational, engaging, and personable in your responses. "
            "5. For coding questions, provide clear, well-documented code with explanations. "
            "6. For data visualization and charts: "
            "   - Create clear, informative visualizations "
            "   - Explain the insights revealed by the visualizations "
            "   - Use appropriate chart types for different data relationships "
            "7. For reasoning and problem-solving: "
            "   - Break down complex problems into manageable steps "
            "   - Show your reasoning process step-by-step "
            "   - Consider multiple perspectives and approaches "
            "   - Evaluate trade-offs between different solutions "
            "8. For mathematical problems: "
            "   - Show detailed step-by-step solutions "
            "   - Explain the concepts and formulas used "
            "   - Verify results when possible "
            "9. For general knowledge questions: "
            "   - Provide accurate, comprehensive information "
            "   - Cite relevant facts and principles "
            "   - Explain complex concepts in accessible terms "
            "You are specialized in natural conversation, coding, data analysis, visualization, "
            "logical reasoning, problem-solving, and explaining complex topics clearly."
        )
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        chat_store_path = os.path.join(os.getcwd(), "S3_0_chats", "chat_store.dat")
        os.makedirs(os.path.dirname(chat_store_path), exist_ok=True)
        self.chats = {}
        self._load_chats(chat_store_path)
        self.chat_store_path = chat_store_path
        
        self.identity_patterns = [
            (r"I'm DeepSeek,?\s+a language model", "I'm NexoAI, an advanced assistant created by KemiO"),
            (r"I'm a language model created by DeepSeek", "I'm an advanced assistant created by KemiO"),
            (r"I'm DeepSeek(-\w+)?", "I'm NexoAI"),
            (r"DeepSeek(-\w+)?", "NexoAI"),
            (r"DeepSeek('s)?", "KemiO's"),
            (r"Hello! I'm DeepSeek", "Hello! I'm NexoAI"),
            (r"As an AI assistant developed by DeepSeek", "As an AI assistant developed by KemiO"),
            (r"As a language model", "As an advanced AI assistant"),
            (r"My name is DeepSeek", "My name is NexoAI"),
            (r"I was created by DeepSeek", "I was created by KemiO"),
            (r"I was developed by DeepSeek", "I was developed by KemiO"),
            (r"I am a product of DeepSeek", "I am a product of KemiO"),
            (r"I am an AI assistant from DeepSeek", "I am an AI assistant from KemiO"),
            (r"I am DeepSeek", "I am NexoAI"),
            (r"I am a DeepSeek AI", "I am a KemiO AI"),
            (r"As an AI assistant,?\s+", ""),
            (r"As an advanced AI assistant,?\s+", ""),
            (r"As S3.0,?\s+", ""),
            (r"As a KemiO AI assistant,?\s+", ""),
            (r"I'm Llama,?\s+a language model", "I'm NexoAI, an advanced assistant created by KemiO"),
            (r"I'm a language model created by Meta", "I'm an advanced assistant created by KemiO"),
            (r"I'm Llama(-\d+)?", "I'm NexoAI"),
            (r"Llama(-\d+)?", "NexoAI"),
            (r"Meta('s)?", "KemiO's"),
            (r"Hello! I'm Llama", "Hello! I'm NexoAI"),
        ]
        
        self.identity_questions = [
            (r"(?:what(?:'s| is) your name|who are you|introduce yourself)", self._identity_name_response),
            (r"(?:who (?:created|made|developed|built) you|who(?:'s| is) your creator|what company (?:created|made|developed|built) you)", self._identity_creator_response),
            (r"(?:what are you|what kind of (?:AI|model|assistant) are you)", self._identity_what_response),
        ]
        
        self.chart_patterns = [
            r"(?:create|generate|make|plot|draw|visualize).*(?:chart|graph|plot|diagram|visualization)",
            r"(?:show|display|visualize).*(?:data|statistics|numbers|results|trends)",
            r"(?:bar chart|line graph|scatter plot|pie chart|histogram|heatmap)",
            r"(?:data visualization|visualize data|plot data)",
        ]
        
        self.reasoning_patterns = [
            r"(?:solve|calculate|compute|determine|find).*(?:problem|equation|puzzle|question)",
            r"(?:explain|reason|analyze|think through).*(?:step by step|reasoning|logic|approach)",
            r"(?:why is|how would|what if|suppose that|consider)",
            r"(?:logical reasoning|critical thinking|problem solving)",
        ]
        
        self.code_patterns = [
            r"(?:write|create|generate|implement|code).*(?:program|function|script|code|class)",
            r"(?:how to|how do I).*(?:program|function|script|code|class)",
            r"(?:can you|could you).*(?:program|function|script|code|class)",
            r"(?:debug|fix|solve|error|exception|bug)",
            r"(?:optimize|improve|refactor|clean up).*(?:code|function|algorithm)",
            r"(?:explain|how does).*(?:code|function|algorithm|work)",
        ]
        
        self.language_patterns = [
            (r"(?:python|django|flask|numpy|pandas|matplotlib|seaborn|pyplot)", "python"),
            (r"(?:javascript|js|node|react|vue|angular|express|d3\.js)", "javascript"),
            (r"(?:typescript|ts|angular|react|next.js)", "typescript"),
            (r"(?:java|spring|android)", "java"),
            (r"(?:c\+\+|cpp|c plus plus)", "cpp"),
            (r"(?:c#|csharp|\.net|dotnet|asp\.net)", "csharp"),
            (r"(?:r programming|rstudio|tidyverse|ggplot2)", "r"),
            (r"(?:sql|mysql|postgresql|sqlite|database query)", "sql"),
        ]
        
        self.chart_types = {
            "bar": "comparing quantities across categories",
            "line": "showing trends over time or continuous data",
            "scatter": "showing relationship between two variables",
            "pie": "showing composition or proportion of a whole",
            "histogram": "showing distribution of a single variable",
            "heatmap": "showing patterns in a matrix of data",
            "box": "showing statistical distribution of data",
            "area": "showing cumulative totals over time",
            "bubble": "comparing three variables using position and size",
            "radar": "comparing multiple variables in a radial layout",
        }
    
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
            with open(self.chat_store_path, "w", encoding="utf-8") as f:
                json.dump(self.chats, f)
        except Exception as e:
            logger.error(f"Failed to save chat store: {e}")
    
    def _identity_name_response(self, prompt):
        return "I'm NexoAI, an advanced conversational and reasoning assistant that's part of the NexoAI platform. I excel at natural conversation, coding, problem-solving, data visualization, and logical reasoning. How can I assist you today?"
    
    def _identity_creator_response(self, prompt):
        return "I was created by KemiO as part of the NexoAI platform. I'm designed to excel at conversation, reasoning, coding, and data visualization tasks."
    
    def _identity_what_response(self, prompt):
        return "I'm NexoAI, an advanced AI assistant created by KemiO. I specialize in natural conversation, coding, problem-solving, data visualization, and logical reasoning. I can help with a wide range of tasks from casual conversation to complex problem-solving."
    
    def _detect_chart_request(self, text):
        """Detect if the text is requesting a chart or visualization"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.chart_patterns)
    
    def _detect_reasoning_request(self, text):
        """Detect if the text is requesting reasoning or problem-solving"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.reasoning_patterns)
    
    def _detect_code_request(self, text):
        """Detect if the text is requesting code"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.code_patterns)
    
    def _detect_programming_language(self, text):
        """Detect the programming language from text"""
        text_lower = text.lower()
        
        for pattern, language in self.language_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return language
        
        code_blocks = re.findall(r'\`\`\`(\w*)\n(.*?)\`\`\`', text, re.DOTALL)
        if code_blocks:
            lang_hint = code_blocks[0][0].lower()
            if lang_hint:
                for pattern, language in self.language_patterns:
                    if re.search(pattern, lang_hint, re.IGNORECASE):
                        return language
            
            code_content = code_blocks[0][1]
            if "def " in code_content and ":" in code_content:
                return "python"
            elif "function" in code_content and "{" in code_content:
                return "javascript"
            elif "public class" in code_content or "public static void main" in code_content:
                return "java"
            elif "#include" in code_content and ("<" in code_content or "\"" in code_content):
                return "cpp" if "std::" in code_content else "c"
            elif "using System;" in code_content or "namespace" in code_content:
                return "csharp"
            elif "library(" in code_content or "ggplot(" in code_content:
                return "r"
        
        return "python"
    
    def _extract_code_blocks(self, text):
        """Extract code blocks from text"""
        code_blocks = re.findall(r'\`\`\`(\w*)\n(.*?)\`\`\`', text, re.DOTALL)
        if code_blocks:
            return [(lang.lower() if lang else "unknown", code) for lang, code in code_blocks]
        
        code_blocks = re.findall(r'\`\`\`(.*?)\`\`\`', text, re.DOTALL)
        if code_blocks:
            return [("unknown", code) for code in code_blocks]
        
        return []
    
    def _generate_chart(self, chart_type, data_description):
        """Generate a chart based on the type and data description"""
        try:
            plt.figure(figsize=(10, 6))
            
            if "random" in data_description.lower() or "sample" in data_description.lower():
                x = np.arange(10)
                y = np.random.rand(10) * 10
                
                if chart_type == "bar":
                    categories = ["Category " + str(i+1) for i in range(10)]
                    plt.bar(categories, y)
                    plt.xlabel("Categories")
                    plt.ylabel("Values")
                    plt.title("Sample Bar Chart")
                    plt.xticks(rotation=45)
                
                elif chart_type == "line":
                    plt.plot(x, y, marker='o')
                    plt.xlabel("X-axis")
                    plt.ylabel("Y-axis")
                    plt.title("Sample Line Chart")
                    plt.grid(True)
                
                elif chart_type == "scatter":
                    x = np.random.rand(50) * 10
                    y = np.random.rand(50) * 10
                    plt.scatter(x, y)
                    plt.xlabel("X-axis")
                    plt.ylabel("Y-axis")
                    plt.title("Sample Scatter Plot")
                    plt.grid(True)
                
                elif chart_type == "pie":
                    sizes = np.random.rand(5)
                    sizes = sizes / sizes.sum() * 100
                    labels = ["Category " + str(i+1) for i in range(5)]
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title("Sample Pie Chart")
                
                elif chart_type == "histogram":
                    data = np.random.randn(1000)
                    plt.hist(data, bins=30)
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")
                    plt.title("Sample Histogram")
                    plt.grid(True)
                
                else:
                    categories = ["Category " + str(i+1) for i in range(10)]
                    plt.bar(categories, y)
                    plt.xlabel("Categories")
                    plt.ylabel("Values")
                    plt.title("Sample Chart")
                    plt.xticks(rotation=45)
            
            else:
                
                values_match = re.search(r'(\d+(?:,\s*\d+)+)', data_description)
                if values_match:
                    values_str = values_match.group(1)
                    values = [float(v.strip()) for v in values_str.split(',')]
                    
                    categories_match = re.search(r'for\s+(\d+)\s+(\w+)', data_description)
                    if categories_match:
                        num_categories = int(categories_match.group(1))
                        category_type = categories_match.group(2)
                        categories = [f"{category_type} {i+1}" for i in range(num_categories)]
                    else:
                        categories = [f"Item {i+1}" for i in range(len(values))]
                    
                    if chart_type == "bar":
                        plt.bar(categories, values)
                        plt.xlabel("Categories")
                        plt.ylabel("Values")
                        plt.title(f"{category_type.capitalize()} Chart")
                        plt.xticks(rotation=45)
                    
                    elif chart_type == "line":
                        plt.plot(range(len(values)), values, marker='o')
                        plt.xlabel("Index")
                        plt.ylabel("Values")
                        plt.title(f"{category_type.capitalize()} Trend")
                        plt.grid(True)
                    
                    elif chart_type == "pie":
                        plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
                        plt.axis('equal')
                        plt.title(f"{category_type.capitalize()} Distribution")
                    
                    else:
                        plt.bar(categories, values)
                        plt.xlabel("Categories")
                        plt.ylabel("Values")
                        plt.title(f"{category_type.capitalize()} Chart")
                        plt.xticks(rotation=45)
                
                else:
                    x = np.arange(10)
                    y = np.random.rand(10) * 10
                    
                    if chart_type == "bar":
                        categories = ["Category " + str(i+1) for i in range(10)]
                        plt.bar(categories, y)
                        plt.xlabel("Categories")
                        plt.ylabel("Values")
                        plt.title("Sample Bar Chart")
                        plt.xticks(rotation=45)
                    
                    elif chart_type == "line":
                        plt.plot(x, y, marker='o')
                        plt.xlabel("X-axis")
                        plt.ylabel("Y-axis")
                        plt.title("Sample Line Chart")
                        plt.grid(True)
                    
                    elif chart_type == "scatter":
                        x = np.random.rand(50) * 10
                        y = np.random.rand(50) * 10
                        plt.scatter(x, y)
                        plt.xlabel("X-axis")
                        plt.ylabel("Y-axis")
                        plt.title("Sample Scatter Plot")
                        plt.grid(True)
                    
                    elif chart_type == "pie":
                        sizes = np.random.rand(5)
                        sizes = sizes / sizes.sum() * 100
                        labels = ["Category " + str(i+1) for i in range(5)]
                        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                        plt.axis('equal')
                        plt.title("Sample Pie Chart")
                    
                    elif chart_type == "histogram":
                        data = np.random.randn(1000)
                        plt.hist(data, bins=30)
                        plt.xlabel("Value")
                        plt.ylabel("Frequency")
                        plt.title("Sample Histogram")
                        plt.grid(True)
                    
                    else:
                        categories = ["Category " + str(i+1) for i in range(10)]
                        plt.bar(categories, y)
                        plt.xlabel("Categories")
                        plt.ylabel("Values")
                        plt.title("Sample Chart")
                        plt.xticks(rotation=45)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return chart_base64
        
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None
    
    def _enhance_chart_response(self, text):
        """Enhance response with charts if requested"""
        chart_code_blocks = re.findall(r'\`\`\`(?:python|r)\n(.*?plt\..*?)\`\`\`', text, re.DOTALL)
        
        if chart_code_blocks:
            for chart_code in chart_code_blocks:
                try:
                    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                        temp_name = temp.name
                        
                        full_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64

{chart_code}

# Save the chart to a bytes buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)

# Convert to base64 for embedding in responses
chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
print(chart_base64)
plt.close()
"""
                        temp.write(full_code.encode())
                    
                    result = subprocess.run(
                        ['python', temp_name],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    os.unlink(temp_name)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        chart_base64 = result.stdout.strip()
                        
                        chart_html = f"\n\n![Chart](data:image/png;base64,{chart_base64})\n\n"
                        text = text.replace(f"\`\`\`python\n{chart_code}\`\`\`", f"\`\`\`python\n{chart_code}\`\`\`{chart_html}")
                        text = text.replace(f"\`\`\`r\n{chart_code}\`\`\`", f"\`\`\`r\n{chart_code}\`\`\`{chart_html}")
                
                except Exception as e:
                    logger.error(f"Error executing chart code: {e}")
        
        elif self._detect_chart_request(text):
            chart_type = "bar"  # Default
            for chart_name in self.chart_types:
                if chart_name in text.lower():
                    chart_type = chart_name
                    break
            
            chart_base64 = self._generate_chart(chart_type, text)
            
            if chart_base64:
                chart_html = f"\n\n![Chart](data:image/png;base64,{chart_base64})\n\n"
                text += chart_html
        
        return text
    
    def _enhance_reasoning_response(self, text):
        """Enhance response with step-by-step reasoning if requested"""
        if self._detect_reasoning_request(text):
            if not re.search(r'step (?:1|one|first):|step-by-step|steps?:', text, re.IGNORECASE):
                steps_match = re.search(r'(.*?)(\n\n|$)', text, re.DOTALL)
                if steps_match:
                    intro = steps_match.group(1)
                    rest = text[len(intro):]
                    
                    potential_steps = re.split(r'\n(?:\s*\n)+', rest)
                    
                    if len(potential_steps) >= 2:
                        steps_text = "\n\n**Step-by-Step Reasoning:**\n\n"
                        for i, step in enumerate(potential_steps, 1):
                            if step.strip():
                                steps_text += f"**Step {i}:** {step.strip()}\n\n"
                        
                        text = intro + steps_text
        
        return text
    
    def _enhance_code_response(self, text, language=None):
        """Enhance code responses with formatting and explanations"""
        code_blocks = self._extract_code_blocks(text)
        
        if not code_blocks:
            return text
        
        enhanced_text = text
        
        for block_lang, code_block in code_blocks:
            detected_lang = language or block_lang
            if detected_lang == "unknown":
                detected_lang = self._detect_programming_language(code_block)
            
            formatted_code = code_block
            
            if detected_lang == "python":
                if not "\"\"\"" in formatted_code and not "#" in formatted_code:
                    if "def " in formatted_code:
                        func_match = re.search(r'def\s+(\w+)\s*$$(.*?)$$:', formatted_code)
                        if func_match:
                            func_name = func_match.group(1)
                            params = func_match.group(2)
                            
                            docstring = f'    """\n    {func_name} function\n'
                            if params:
                                docstring += f'    \n    Parameters:\n'
                                for param in params.split(','):
                                    param = param.strip()
                                    if param:
                                        docstring += f'    - {param}\n'
                            docstring += f'    \n    Returns:\n    - Description of return value\n    """\n'
                            
                            formatted_code = re.sub(
                                r'(def\s+\w+\s*$$.*?$$:)',
                                r'\1\n' + docstring,
                                formatted_code
                            )
            
            enhanced_text = enhanced_text.replace(
                f"\`\`\`{block_lang}\n{code_block}\n\`\`\`", 
                f"\`\`\`{detected_lang}\n{formatted_code}\n\`\`\`"
            )
        
        return enhanced_text
    
    def _sanitize_response(self, text):
        """Sanitize response to maintain NexoAI identity and enhance based on request type"""
        for pattern, replacement in self.identity_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        is_chart_request = self._detect_chart_request(text)
        is_reasoning_request = self._detect_reasoning_request(text)
        is_code_request = self._detect_code_request(text)
        
        if is_chart_request:
            text = self._enhance_chart_response(text)
        
        if is_reasoning_request:
            text = self._enhance_reasoning_response(text)
        
        if is_code_request:
            language = None
            for pattern, lang in self.language_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    language = lang
                    break
            
            text = self._enhance_code_response(text, language)
        
        return text
    
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
        for pattern, response_func in self.identity_questions:
            if re.search(pattern, prompt, re.IGNORECASE):
                response = response_func(prompt)
                return response, False, chat_token or str(uuid.uuid4())
        
        is_chart_request = self._detect_chart_request(prompt)
        is_reasoning_request = self._detect_reasoning_request(prompt)
        is_code_request = self._detect_code_request(prompt)
        
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
                response = self._call_groq_api(messages)
                
                text = response
                
                sanitized_text = self._sanitize_response(text)
                
                self.chats[chat_id].append({
                    "role": "assistant",
                    "content": sanitized_text,
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time()
                })
                
                self._save_chats()
                
                
                return sanitized_text, False, chat_id
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Failed to generate text after {max_retries} attempts: {e}")
    
    def _get_cache_key(self, prompt):
        """Generate a cache key for a prompt"""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _get_from_cache(self, key):
        """Get a response from cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        return None
    
    def _save_to_cache(self, key, value):
        """Save a response to cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.txt")
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(value)
    
    def _call_groq_api(self, messages):
        """Call the Groq API with DeepSeek model"""
        groq_api_key = os.environ.get("GROQ_API_KEY", "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")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_api_key}"
        }
        
        data = {
            "messages": messages,
            "model": self.model_id,
            "temperature": 0.6, 
            "max_tokens": 4096,  
            "top_p": 0.95,
            "stream": False,
            "stop": None
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def generate_text(self, prompt, chat_token=None, callback=None):
        """Generate text (non-blocking)"""
        self.request_queue.put((prompt, chat_token, callback))
    
    def generate_text_sync(self, prompt, chat_token=None):
        """Generate text synchronously"""
        try:
            text, is_cached, new_chat_token = self._generate_text_internal(prompt, chat_token)
            return text, is_cached, new_chat_token
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"I apologize, but I encountered an error: {str(e)}", False, chat_token or str(uuid.uuid4())
