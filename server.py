from fastapi import FastAPI, HTTPException, Response, BackgroundTasks, Query
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
import uuid
import logging
import asyncio
from io import BytesIO
from models.genaz import GenAZ
from models.s1_5 import S1_5
from models.s2_0 import S2_0
from models.s3_0 import S3_0

logger = logging.getLogger("nexoai-server")

app = None
model_instances = {}
config = None

def get_model(model_name):
    if model_name not in model_instances:
        if model_name in config.get("models", {}):
            model_id = config["models"][model_name]
        else:
            model_id = model_name
        
        if model_name == "s1_5":
            model_instances[model_name] = S1_5(
                model_id=model_id,
                cache_dir=config.get("storage", {}).get("cache_dir", "S1_5_cache")
            )
        elif model_name == "s2_0":
            model_instances[model_name] = S2_0(
                model_id=model_id,
                cache_dir=config.get("storage", {}).get("cache_dir", "S2_0_cache")
            )
        elif model_name == "s3_0":
            model_instances[model_name] = S3_0(
                model_id=model_id,
                cache_dir=config.get("storage", {}).get("cache_dir", "S3_0_cache")
            )
        else:
            model_instances[model_name] = GenAZ(
                model_id=model_id,
                temp_dir=config.get("storage", {}).get("temp_dir", "GenAZ_temp"),
                cache_dir=config.get("storage", {}).get("cache_dir", "GenAZ_cache")
            )
    return model_instances[model_name]

def create_app(app_config):
    global app, config
    config = app_config
    
    app = FastAPI(title="NexoAI API", description="API for NexoAI image and text generation")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "name": "NexoAI",
            "version": "1.0.0",
            "models": list(config.get("models", {}).get("available", []))
        }
    
    @app.get("/model/{model_name}/{prompt}")
    async def generate_content(
        model_name: str, 
        prompt: str, 
        chat_token: str = Query(None, description="Chat token for conversation continuity"),
        background_tasks: BackgroundTasks = None
    ):
        if model_name not in config.get("models", {}).get("available", []):
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        try:
            model = get_model(model_name)
            
            prompt = prompt.replace("+", " ")
            
            if model_name in ["s1_5", "s2_0", "s3_0"]:
                logger.info(f"Generating text for prompt: '{prompt}' using model: {model_name}")
                start_time = time.time()
                
                text, is_cached, new_chat_token = await model.generate_text_async(prompt, chat_token)
                
                generation_time = time.time() - start_time
                cache_text = " (from cache)" if is_cached else ""
                logger.info(f"Text generated in {generation_time:.2f} seconds{cache_text}")
                
                return JSONResponse(content={
                    "text": text,
                    "chat_token": new_chat_token
                })
            else:
                logger.info(f"Generating image for prompt: '{prompt}' using model: {model_name}")
                start_time = time.time()
                
                image, is_cached = await model.generate_image_async(prompt)
                
                generation_time = time.time() - start_time
                cache_text = " (from cache)" if is_cached else ""
                logger.info(f"Image generated in {generation_time:.2f} seconds{cache_text}")
                
                image = model.add_watermark(image)
                
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/1.5/{prompt}")
    async def generate_text_s1_alias(
        prompt: str,
        chat_token: str = Query(None, description="Chat token for conversation continuity")
    ):
        return await generate_content("s1_5", prompt, chat_token)
    
    @app.get("/2.0/{prompt}")
    async def generate_text_s2_alias(
        prompt: str,
        chat_token: str = Query(None, description="Chat token for conversation continuity")
    ):
        return await generate_content("s2_0", prompt, chat_token)
    
    @app.get("/3.0/{prompt}")
    async def generate_text_s3_alias(
        prompt: str,
        chat_token: str = Query(None, description="Chat token for conversation continuity")
    ):
        return await generate_content("s3_0", prompt, chat_token)
    
    @app.post("/chat")
    async def chat(
        prompt: str,
        chat_token: str = Query(None, description="Chat token for conversation continuity"),
        model_name: str = "s3_0"  
    ):
        if model_name not in config.get("models", {}).get("available", []):
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        try:
            model = get_model(model_name)
            
            if not hasattr(model, 'generate_text_async'):
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support chat")
            
            logger.info(f"Chat request with prompt: '{prompt}' using model: {model_name}")
            start_time = time.time()
            
            text, is_cached, new_chat_token = await model.generate_text_async(prompt, chat_token)
            
            generation_time = time.time() - start_time
            cache_text = " (from cache)" if is_cached else ""
            logger.info(f"Chat response generated in {generation_time:.2f} seconds{cache_text}")
            
            return JSONResponse(content={
                "text": text,
                "chat_token": new_chat_token
            })
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app

def start_server(app_config):
    app = create_app(app_config)
    host = app_config.get("server", {}).get("host", "0.0.0.0")
    port = app_config.get("server", {}).get("port", 8000)
    
    logger.info(f"Starting NexoAI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
