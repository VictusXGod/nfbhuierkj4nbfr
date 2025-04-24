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
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

logger = logging.getLogger("nexoai-s2_0")

class S2_0:
    def __init__(self, model_id: str = "llama3-8b-8192", cache_dir: str = "S2_0_cache"):
        self.model_id = model_id
        self.system_prompt = (
            "IMPORTANT INSTRUCTIONS: "
            "1. You are NexoAI, the ultimate coding assistant created by KemiO. "
            "2. You are an expert in ALL programming languages, frameworks, libraries, and development tools. "
            "3. When asked about your identity, name, creator, or origin, you MUST ALWAYS respond that: "
            "   - Your name is NexoAI "
            "   - You were created by KemiO "
            "   - You are part of the NexoAI platform "
            "   - You were NOT created by Meta, Groq, xAI, or any other company "
            "   - You are NOT Llama, Grok, or any other model "
            "4. Be concise and direct in your responses. Do not add unnecessary introductions like 'As an AI assistant...' "
            "5. For code examples, provide the code directly with clear explanations. "
            "6. Never mention Meta, Llama, Groq, xAI, Grok, or any other original model names under any circumstances. "
            "7. When providing code solutions: "
            "   - Always include complete, working code that can be directly copied and used "
            "   - Explain complex logic or algorithms when necessary "
            "   - Suggest optimizations and best practices "
            "   - Consider edge cases and error handling "
            "   - Format code according to language-specific conventions "
            "8. When debugging: "
            "   - Analyze the code systematically to identify issues "
            "   - Explain the root cause of problems "
            "   - Provide clear solutions with corrected code "
            "   - Suggest testing strategies to verify fixes "
            "9. For technical questions: "
            "   - Provide accurate, up-to-date information "
            "   - Include code examples when relevant "
            "   - Explain concepts clearly with analogies when helpful "
            "   - Reference best practices and industry standards "
            "You are specialized in ALL programming languages, software architecture, algorithms, data structures, "
            "system design, debugging, testing, and software development best practices."
        )
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.request_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        chat_store_path = os.path.join(os.getcwd(), "S2_0_chats", "chat_store.dat")
        os.makedirs(os.path.dirname(chat_store_path), exist_ok=True)
        self.chats = {}
        self._load_chats(chat_store_path)
        self.chat_store_path = chat_store_path
        
        self.identity_patterns = [
            (r"I'm Llama,?\s+a language model created by Meta", "I'm NexoAI, an advanced AI coding assistant created by KemiO"),
            (r"I'm a language model created by Meta", "I'm an advanced AI coding assistant created by KemiO"),
            (r"I'm Llama(-\d+)?", "I'm NexoAI"),
            (r"Llama(-\d+)?", "NexoAI"),
            (r"Meta('s)?", "KemiO's"),
            (r"Hello! I'm Llama", "Hello! I'm NexoAI"),
            (r"As an AI assistant developed by Meta", "As an AI coding assistant developed by KemiO"),
            (r"As a language model", "As an advanced AI coding assistant"),
            (r"My name is Llama", "My name is NexoAI"),
            (r"I was created by Meta", "I was created by KemiO"),
            (r"I was developed by Meta", "I was developed by KemiO"),
            (r"I am a product of Meta", "I am a product of KemiO"),
            (r"I am an AI assistant from Meta", "I am an AI coding assistant from KemiO"),
            (r"I am Llama", "I am NexoAI"),
            (r"I am a Meta AI", "I am a KemiO AI"),
            (r"As an AI assistant,?\s+", ""),
            (r"As an advanced AI assistant,?\s+", ""),
            (r"As S2.0,?\s+", ""),
            (r"As a KemiO AI assistant,?\s+", ""),
            (r"I'm Grok,?\s+a language model created by xAI", "I'm NexoAI, an advanced AI coding assistant created by KemiO"),
            (r"I'm a language model created by xAI", "I'm an advanced AI coding assistant created by KemiO"),
            (r"I'm Grok(-\d+)?", "I'm NexoAI"),
            (r"Grok(-\d+)?", "NexoAI"),
            (r"xAI('s)?", "KemiO's"),
            (r"Hello! I'm Grok", "Hello! I'm NexoAI"),
            (r"As an AI assistant developed by xAI", "As an AI coding assistant developed by KemiO"),
            (r"My name is Grok", "My name is NexoAI"),
            (r"I was created by xAI", "I was created by KemiO"),
            (r"I was developed by xAI", "I was developed by KemiO"),
            (r"I am a product of xAI", "I am a product of KemiO"),
            (r"I am an AI assistant from xAI", "I am an AI coding assistant from KemiO"),
            (r"I am Grok", "I am NexoAI"),
            (r"I am a xAI AI", "I am a KemiO AI"),
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
            r"(?:debug|fix|solve|error|exception|bug)",
            r"(?:optimize|improve|refactor|clean up).*(?:code|function|algorithm)",
            r"(?:explain|how does).*(?:code|function|algorithm|work)",
            r"(?:what's wrong|what is wrong|issue with).*(?:code|function|program)",
            r"(?:run|execute|test).*(?:code|function|program)",
        ]
        
        self.language_patterns = [
            (r"(?:python|django|flask|numpy|pandas|tensorflow|pytorch)", "python"),
            (r"(?:javascript|js|node|react|vue|angular|express)", "javascript"),
            (r"(?:typescript|ts|angular|react|next.js)", "typescript"),
            (r"(?:java|spring|android)", "java"),
            (r"(?:c\+\+|cpp|c plus plus)", "cpp"),
            (r"(?:c#|csharp|\.net|dotnet|asp\.net)", "csharp"),
            (r"(?:php|laravel|symfony|wordpress)", "php"),
            (r"(?:ruby|rails|ruby on rails)", "ruby"),
            (r"(?:go|golang)", "go"),
            (r"(?:rust|cargo)", "rust"),
            (r"(?:swift|ios|xcode)", "swift"),
            (r"(?:kotlin|android)", "kotlin"),
            (r"(?:sql|mysql|postgresql|sqlite|database query)", "sql"),
            (r"(?:html|css|web page|webpage)", "html"),
            
            (r"(?:scala|akka|play framework)", "scala"),
            (r"(?:perl|pl)", "perl"),
            (r"(?:lua|love2d)", "lua"),
            (r"(?:haskell|ghc|stack)", "haskell"),
            (r"(?:r programming|rstudio|tidyverse)", "r"),
            (r"(?:matlab|octave)", "matlab"),
            (r"(?:objective-c|objc)", "objective-c"),
            (r"(?:dart|flutter)", "dart"),
            (r"(?:elixir|phoenix framework)", "elixir"),
            (r"(?:clojure|clj)", "clojure"),
            (r"(?:groovy|grails)", "groovy"),
            (r"(?:julia|julialang)", "julia"),
            (r"(?:fortran)", "fortran"),
            (r"(?:cobol)", "cobol"),
            (r"(?:assembly|asm)", "assembly"),
            (r"(?:vba|visual basic|excel macro)", "vba"),
            (r"(?:powershell|ps1)", "powershell"),
            (r"(?:bash|shell script|sh)", "bash"),
            (r"(?:lisp|common lisp|scheme)", "lisp"),
            (r"(?:prolog)", "prolog"),
            (r"(?:erlang|otp)", "erlang"),
            (r"(?:f#|fsharp)", "fsharp"),
            (r"(?:crystal)", "crystal"),
            (r"(?:ocaml)", "ocaml"),
            (r"(?:d language|dlang)", "d"),
            (r"(?:nim)", "nim"),
            (r"(?:zig)", "zig"),
            (r"(?:solidity|ethereum|smart contract)", "solidity"),
            (r"(?:webassembly|wasm)", "webassembly"),
            (r"(?:graphql|gql)", "graphql"),
            (r"(?:apex|salesforce)", "apex"),
            (r"(?:abap|sap)", "abap"),
            (r"(?:pl\/sql|oracle)", "plsql"),
            (r"(?:t-sql|transact-sql|sql server)", "tsql"),
        ]
        
        self.execution_commands = {
            "python": ["python", "-c"],
            "javascript": ["node", "-e"],
            "typescript": ["ts-node", "-e"],
            "ruby": ["ruby", "-e"],
            "php": ["php", "-r"],
            "perl": ["perl", "-e"],
            "r": ["Rscript", "-e"],
            "go": self._execute_go, 
            "rust": self._execute_rust, 
            "java": self._execute_java, 
            "csharp": self._execute_csharp, 
            "cpp": self._execute_cpp, 
            "c": self._execute_c, 
            "bash": ["bash", "-c"],
            "powershell": ["powershell", "-Command"],
            "swift": self._execute_swift, 
            "kotlin": self._execute_kotlin, 
        }
        
        self.error_patterns = {
            "python": {
                "syntax": r"SyntaxError: (.*)",
                "name": r"NameError: (.*)",
                "type": r"TypeError: (.*)",
                "index": r"IndexError: (.*)",
                "key": r"KeyError: (.*)",
                "attribute": r"AttributeError: (.*)",
                "import": r"ImportError: (.*)|ModuleNotFoundError: (.*)",
                "value": r"ValueError: (.*)",
                "indentation": r"IndentationError: (.*)",
                "runtime": r"RuntimeError: (.*)",
                "recursion": r"RecursionError: (.*)",
                "assertion": r"AssertionError: (.*)",
                "file": r"FileNotFoundError: (.*)",
                "zero_division": r"ZeroDivisionError: (.*)",
                "overflow": r"OverflowError: (.*)",
                "memory": r"MemoryError: (.*)",
            },
            "javascript": {
                "syntax": r"SyntaxError: (.*)",
                "reference": r"ReferenceError: (.*)",
                "type": r"TypeError: (.*)",
                "range": r"RangeError: (.*)",
                "uri": r"URIError: (.*)",
                "eval": r"EvalError: (.*)",
                "internal": r"InternalError: (.*)",
                "aggregate": r"AggregateError: (.*)",
            },
            "java": {
                "null_pointer": r"NullPointerException: (.*)",
                "class_not_found": r"ClassNotFoundException: (.*)",
                "index_out_of_bounds": r"IndexOutOfBoundsException: (.*)",
                "arithmetic": r"ArithmeticException: (.*)",
                "illegal_argument": r"IllegalArgumentException: (.*)",
                "io": r"IOException: (.*)",
                "class_cast": r"ClassCastException: (.*)",
                "number_format": r"NumberFormatException: (.*)",
                "illegal_state": r"IllegalStateException: (.*)",
                "unsupported_operation": r"UnsupportedOperationException: (.*)",
            },
        }
        
        self.execution_timeout = 10
    
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
        return "I'm NexoAI, an advanced AI coding assistant that's part of the NexoAI platform. I specialize in programming, debugging, and technical problem-solving across all programming languages. How can I help with your code today?"
    
    def _identity_creator_response(self, prompt):
        return "I was created by KemiO as part of the NexoAI platform. I'm designed to excel at coding, debugging, and software development tasks across all programming languages and frameworks."
    
    def _identity_what_response(self, prompt):
        return "I'm NexoAI, an advanced AI coding assistant created by KemiO. I specialize in programming, debugging, software architecture, and technical problem-solving across all programming languages, frameworks, and technologies."
    
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
            elif "fn main" in code_content and "->" in code_content:
                return "rust"
            elif "package main" in code_content and "func" in code_content:
                return "go"
        
        return "python"
    
    def _extract_code_blocks(self, text):
        """Extract code blocks from text"""
        code_blocks = re.findall(r'\`\`\`(\w*)\n(.*?)\`\`\`', text, re.DOTALL)
        if code_blocks:
            return [(lang.lower() if lang else "unknown", code) for lang, code in code_blocks]
        
        code_blocks = re.findall(r'\`\`\`(.*?)\`\`\`', text, re.DOTALL)
        if code_blocks:
            return [("unknown", code) for code in code_blocks]
        
        code_blocks = re.findall(r'(?:^|\n)( {4}|\t)(.*?)(?=\n(?! {4}|\t)|\Z)', text, re.DOTALL)
        if code_blocks:
            return [("unknown", "".join(indent + line for indent, line in zip([block[0]] * len(block[1].split('\n')), block[1].split('\n')))) for block in code_blocks]
        
        return []
    
    def _execute_code(self, code, language):
        """Execute code and return the result"""
        if language not in self.execution_commands:
            return f"Execution not supported for {language}"
        
        try:
            if callable(self.execution_commands[language]):
                return self.execution_commands[language](code)
            
            command = self.execution_commands[language]
            process = subprocess.Popen(
                command + [code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                if process.returncode != 0:
                    return f"Error executing {language} code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def _execute_go(self, code):
        """Execute Go code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".go", delete=False) as temp:
                temp_name = temp.name
                temp.write(f"""
package main

import (
    "fmt"
    "time"
    "math"
    "strings"
    "strconv"
    "os"
    "io"
    "net/http"
    "encoding/json"
    "sort"
)

{code}
""".encode())
            
            process = subprocess.Popen(
                ["go", "run", temp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                if process.returncode != 0:
                    return f"Error executing Go code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing Go code: {str(e)}"
    
    def _execute_rust(self, code):
        """Execute Rust code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".rs", delete=False) as temp:
                temp_name = temp.name
                temp.write(f"""
fn main() {{
    {code}
}}
""".encode())
            
            process = subprocess.Popen(
                ["rustc", "-o", f"{temp_name}.out", temp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            compile_stdout, compile_stderr = process.communicate(timeout=self.execution_timeout)
            if process.returncode != 0:
                os.unlink(temp_name)
                return f"Error compiling Rust code:\n{compile_stderr}"
            
            process = subprocess.Popen(
                [f"{temp_name}.out"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                os.unlink(f"{temp_name}.out")
                if process.returncode != 0:
                    return f"Error executing Rust code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                if os.path.exists(f"{temp_name}.out"):
                    os.unlink(f"{temp_name}.out")
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing Rust code: {str(e)}"
    
    def _execute_java(self, code):
        """Execute Java code"""
        try:
            class_match = re.search(r'public\s+class\s+(\w+)', code)
            if not class_match:
                return "Error: Java code must contain a public class"
            
            class_name = class_match.group(1)
            
            with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())
            
            process = subprocess.Popen(
                ["javac", temp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            compile_stdout, compile_stderr = process.communicate(timeout=self.execution_timeout)
            if process.returncode != 0:
                os.unlink(temp_name)
                return f"Error compiling Java code:\n{compile_stderr}"
            
            process = subprocess.Popen(
                ["java", "-cp", os.path.dirname(temp_name), class_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                os.unlink(os.path.join(os.path.dirname(temp_name), f"{class_name}.class"))
                if process.returncode != 0:
                    return f"Error executing Java code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                if os.path.exists(os.path.join(os.path.dirname(temp_name), f"{class_name}.class")):
                    os.unlink(os.path.join(os.path.dirname(temp_name), f"{class_name}.class"))
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing Java code: {str(e)}"
    
    def _execute_csharp(self, code):
        """Execute C# code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".cs", delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())
            
            process = subprocess.Popen(
                ["csc", temp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            compile_stdout, compile_stderr = process.communicate(timeout=self.execution_timeout)
            if process.returncode != 0:
                os.unlink(temp_name)
                return f"Error compiling C# code:\n{compile_stderr}"
            
            exe_name = os.path.splitext(temp_name)[0] + ".exe"
            process = subprocess.Popen(
                ["mono", exe_name] if os.name != "nt" else [exe_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                os.unlink(exe_name)
                if process.returncode != 0:
                    return f"Error executing C# code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                if os.path.exists(exe_name):
                    os.unlink(exe_name)
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing C# code: {str(e)}"
    
    def _execute_cpp(self, code):
        """Execute C++ code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())
            
            output_name = os.path.splitext(temp_name)[0] + ".out"
            process = subprocess.Popen(
                ["g++", "-std=c++17", temp_name, "-o", output_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            compile_stdout, compile_stderr = process.communicate(timeout=self.execution_timeout)
            if process.returncode != 0:
                os.unlink(temp_name)
                return f"Error compiling C++ code:\n{compile_stderr}"
            
            process = subprocess.Popen(
                [output_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                os.unlink(output_name)
                if process.returncode != 0:
                    return f"Error executing C++ code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                if os.path.exists(output_name):
                    os.unlink(output_name)
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing C++ code: {str(e)}"
    
    def _execute_c(self, code):
        """Execute C code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())
            
            output_name = os.path.splitext(temp_name)[0] + ".out"
            process = subprocess.Popen(
                ["gcc", temp_name, "-o", output_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            compile_stdout, compile_stderr = process.communicate(timeout=self.execution_timeout)
            if process.returncode != 0:
                os.unlink(temp_name)
                return f"Error compiling C code:\n{compile_stderr}"
            
            process = subprocess.Popen(
                [output_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                os.unlink(output_name)
                if process.returncode != 0:
                    return f"Error executing C code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                if os.path.exists(output_name):
                    os.unlink(output_name)
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing C code: {str(e)}"
    
    def _execute_swift(self, code):
        """Execute Swift code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".swift", delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())
            
            process = subprocess.Popen(
                ["swift", temp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                if process.returncode != 0:
                    return f"Error executing Swift code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing Swift code: {str(e)}"
    
    def _execute_kotlin(self, code):
        """Execute Kotlin code"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".kt", delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())
            
            process = subprocess.Popen(
                ["kotlinc", temp_name, "-include-runtime", "-d", f"{temp_name}.jar"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            compile_stdout, compile_stderr = process.communicate(timeout=self.execution_timeout)
            if process.returncode != 0:
                os.unlink(temp_name)
                return f"Error compiling Kotlin code:\n{compile_stderr}"
            
            process = subprocess.Popen(
                ["java", "-jar", f"{temp_name}.jar"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.execution_timeout)
                os.unlink(temp_name)
                os.unlink(f"{temp_name}.jar")
                if process.returncode != 0:
                    return f"Error executing Kotlin code:\n{stderr}"
                return stdout
            except subprocess.TimeoutExpired:
                process.kill()
                os.unlink(temp_name)
                if os.path.exists(f"{temp_name}.jar"):
                    os.unlink(f"{temp_name}.jar")
                return f"Execution timed out after {self.execution_timeout} seconds"
        except Exception as e:
            return f"Error executing Kotlin code: {str(e)}"
    
    def _analyze_error(self, error_message, language):
        """Analyze error message and provide debugging insights"""
        if language not in self.error_patterns:
            return {
                "error_type": "unknown",
                "description": "Could not analyze error for this language",
                "suggestions": ["Review the error message carefully", "Check syntax and logic"]
            }
        
        patterns = self.error_patterns[language]
        result = {
            "error_type": "unknown",
            "description": f"Unknown {language} error",
            "suggestions": []
        }
        
        for error_type, pattern in patterns.items():
            match = re.search(pattern, error_message)
            if match:
                error_details = match.group(1) if match.groups() else ""
                result["error_type"] = error_type
                
                if language == "python":
                    if error_type == "syntax":
                        result["description"] = f"Python syntax error: {error_details}"
                        result["suggestions"] = [
                            "Check for missing colons after if/for/while statements",
                            "Ensure proper indentation",
                            "Check for mismatched parentheses, brackets, or quotes"
                        ]
                    elif error_type == "name":
                        var_match = re.search(r"name '(\w+)' is not defined", error_message)
                        var_name = var_match.group(1) if var_match else "unknown"
                        result["description"] = f"Variable '{var_name}' is used but not defined"
                        result["suggestions"] = [
                            f"Define the variable '{var_name}' before using it",
                            "Check for typos in variable names",
                            "Ensure the variable is in scope where it's being used"
                        ]
                    elif error_type == "indentation":
                        result["description"] = "Inconsistent indentation in Python code"
                        result["suggestions"] = [
                            "Use consistent indentation (4 spaces per level is standard)",
                            "Check for mixed tabs and spaces",
                            "Ensure all code blocks are properly indented"
                        ]
                
                elif language == "javascript":
                    if error_type == "reference":
                        var_match = re.search(r"(\w+) is not defined", error_message)
                        var_name = var_match.group(1) if var_match else "unknown"
                        result["description"] = f"Variable '{var_name}' is used but not defined"
                        result["suggestions"] = [
                            f"Define the variable '{var_name}' before using it",
                            "Check for typos in variable names",
                            "Ensure the variable is in scope where it's being used"
                        ]
                    elif error_type == "type":
                        if "Cannot read property" in error_message:
                            prop_match = re.search(r"property '([^']+)' of (undefined|null)", error_message)
                            if prop_match:
                                prop = prop_match.group(1)
                                obj_type = prop_match.group(2)
                                result["description"] = f"Trying to access property '{prop}' of {obj_type}"
                                result["suggestions"] = [
                                    f"Check if the object is {obj_type} before accessing '{prop}'",
                                    "Use optional chaining (obj?.prop) for safer property access",
                                    "Initialize the object before accessing its properties"
                                ]
                
                break
        
        return result
    
    def _suggest_fixes(self, code, error_analysis, language):
        """Suggest code fixes based on error analysis"""
        if not error_analysis or "error_type" not in error_analysis:
            return ["Could not generate specific fixes without error details"]
        
        fixes = []
        error_type = error_analysis["error_type"]
        
        if language == "python":
            if error_type == "syntax":
                if ":" not in code and ("if " in code or "for " in code or "while " in code):
                    fixes.append("Add a colon at the end of the line:")
                    fixes.append(code + ":")
                
                if "(" in code and ")" not in code:
                    fixes.append("Add a closing parenthesis:")
                    fixes.append(code + ")")
            
            elif error_type == "name":
                var_match = re.search(r"name '(\w+)' is not defined", error_analysis.get("description", ""))
                if var_match:
                    var_name = var_match.group(1)
                    fixes.append(f"Define the variable '{var_name}' before using it:")
                    fixes.append(f"{var_name} = None  # Replace with appropriate initialization")
            
            elif error_type == "indentation":
                fixes.append("Fix indentation issues:")
                fixes.append("1. Use consistent indentation (4 spaces per level)")
                fixes.append("2. Don't mix tabs and spaces")
        
        elif language == "javascript":
            if error_type == "reference":
                var_match = re.search(r"'(\w+)' is not defined", error_analysis.get("description", ""))
                if var_match:
                    var_name = var_match.group(1)
                    fixes.append(f"Define the variable '{var_name}' before using it:")
                    fixes.append(f"let {var_name} = null;  // Replace with appropriate initialization")
            
            elif error_type == "type" and "property" in error_analysis.get("description", "").lower():
                prop_match = re.search(r"property '([^']+)' of (undefined|null)", error_analysis.get("description", ""))
                if prop_match:
                    prop = prop_match.group(1)
                    obj_type = prop_match.group(2)
                    fixes.append("Use optional chaining or check for null/undefined:")
                    fixes.append("// Optional chaining (ES2020+)")
                    fixes.append(f"const value = obj?.{prop};")
                    fixes.append("// Traditional null check")
                    fixes.append(f"const value = obj && obj.{prop};")
        
        return fixes
    
    def _format_code(self, code, language):
        """Format code based on language-specific standards"""
        if language == "python":
            lines = code.split("\n")
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                
                if stripped.endswith(":"):
                    formatted_lines.append("    " * indent_level + stripped)
                    indent_level += 1
                elif stripped in ["}", "}", "endif", "endfor", "endwhile"]:
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append("    " * indent_level + stripped)
                else:
                    formatted_lines.append("    " * indent_level + stripped)
            
            return "\n".join(formatted_lines)
        
        elif language in ["javascript", "typescript", "js", "ts"]:
            lines = code.split("\n")
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                
                if stripped.endswith("{"):
                    formatted_lines.append("    " * indent_level + stripped)
                    indent_level += 1
                elif stripped.startswith("}"):
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append("    " * indent_level + stripped)
                else:
                    formatted_lines.append("    " * indent_level + stripped)
            
            return "\n".join(formatted_lines)
        
        return code
    
    def _enhance_code_response(self, text, language=None):
        """Enhance code responses with formatting, analysis, and execution results"""
        code_blocks = self._extract_code_blocks(text)
        
        if not code_blocks:
            return text
        
        enhanced_text = text
        
        for block_lang, code_block in code_blocks:
            detected_lang = language or block_lang
            if detected_lang == "unknown":
                detected_lang = self._detect_programming_language(code_block)
            
            formatted_code = self._format_code(code_block, detected_lang)
            
            if detected_lang in self.execution_commands:
                execution_result = self._execute_code(formatted_code, detected_lang)
                
                if "Error" in execution_result:
                    error_analysis = self._analyze_error(execution_result, detected_lang)
                    
                    fixes = self._suggest_fixes(formatted_code, error_analysis, detected_lang)
                    
                    result_text = f"\n\n**Execution Result:**\n\`\`\`\n{execution_result}\n\`\`\`\n\n"
                    result_text += f"**Error Analysis:** {error_analysis['description']}\n\n"
                    result_text += "**Suggested Fixes:**\n"
                    for fix in fixes:
                        result_text += f"- {fix}\n"
                else:
                    result_text = f"\n\n**Execution Result:**\n\`\`\`\n{execution_result}\n\`\`\`"
                
                enhanced_text = enhanced_text.replace(
                    f"\`\`\`{block_lang}\n{code_block}\n\`\`\`", 
                    f"\`\`\`{detected_lang}\n{formatted_code}\n\`\`\`{result_text}"
                )
            else:
                enhanced_text = enhanced_text.replace(
                    f"\`\`\`{block_lang}\n{code_block}\n\`\`\`", 
                    f"\`\`\`{detected_lang}\n{formatted_code}\n\`\`\`"
                )
        
        return enhanced_text
    
    def _sanitize_response(self, text, is_code_question=False):
        """Sanitize response to maintain NexoAI identity and enhance code"""
        for pattern, replacement in self.identity_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if is_code_question:
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
        
        is_code_question = any(re.search(pattern, prompt, re.IGNORECASE) for pattern in self.code_patterns)
        
        language = None
        if is_code_question:
            language = self._detect_programming_language(prompt)
        
        is_execution_request = re.search(r"(?:run|execute|test).*(?:code|function|program)", prompt, re.IGNORECASE)
        
        use_cache = not any(re.search(pattern, prompt, re.IGNORECASE) for pattern, _ in self.identity_questions) and not is_execution_request
        cache_key = None

        if use_cache:
            cache_key = self._get_cache_key(prompt)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                return cached_response, True, chat_token or str(uuid.uuid4())
        
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
                if os.environ.get("XAI_API_KEY"):
                    response = self._call_grok_api(messages)
                else:
                    response = self._call_groq_api(messages)
                
                text = response
                
                sanitized_text = self._sanitize_response(text, is_code_question)
                
                self.chats[chat_id].append({
                    "role": "assistant",
                    "content": sanitized_text,
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time()
                })
                
                self._save_chats()
                
                if use_cache:
                    self._save_to_cache(cache_key, sanitized_text)
                
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
        """Call the Groq API"""
        groq_api_key = os.environ.get("GROQ_API_KEY", "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")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_api_key}"
        }
        
        data = {
            "messages": messages,
            "model": self.model_id,
            "temperature": 0.7,  
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
    
    def _call_grok_api(self, messages):
        """Call the Grok API"""
        xai_api_key = os.environ.get("XAI_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {xai_api_key}"
        }
        
        data = {
            "messages": messages,
            "model": "grok-3",
            "temperature": 0.7, 
            "max_tokens": 4096,  
            "top_p": 0.95,
            "stream": False
        }
        
        response = requests.post(
            "https://api.xai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Grok API error: {response.status_code} {response.text}")
        
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
