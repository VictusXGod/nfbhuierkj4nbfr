import time
import sys
import os
import logging
import argparse
import yaml
import asyncio
from pathlib import Path
from models.genaz import GenAZ
from models.s1_5 import S1_5
from models.s2_0 import S2_0
from models.s3_0 import S3_0
from PIL import Image
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NexoAI")

def load_config():
    """Load configuration from config.yaml file"""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def check_system_compatibility(config):
    """Check system compatibility and adjust settings if needed"""
    if config.get("optimization", {}).get("use_gpu", True) and not torch.cuda.is_available():
        logger.warning("GPU acceleration requested but no GPU available. Falling back to CPU.")
        if "optimization" in config:
            config["optimization"]["use_gpu"] = False

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.2f}GB memory")
        
        if gpu_memory < 8 and config.get("optimization", {}).get("batch_size", 1) > 1:
            logger.warning("Limited GPU memory detected. Setting batch size to 1.")
            if "optimization" in config:
                config["optimization"]["batch_size"] = 1
    else:
        logger.info("Running on CPU")

    return config

def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config.get("storage", {}).get("temp_dir", "temp"),
        config.get("storage", {}).get("cache_dir", "cache"),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

async def run_async(config, args, parser):
    """Run the application in async mode"""
    if args.list_models:
        print("Available models:")
        for model in config.get("models", {}).get("available", []):
            print(f"  - {model}")
        return

    if not args.prompt:
        parser.print_help()
        return

    if args.acceleration:
        config["optimization"]["use_gpu"] = args.acceleration == "gpu"

    print(f"Running NexoAI in {args.mode} mode with {'GPU' if config['optimization']['use_gpu'] else 'CPU'}")

    try:
        temp_dir = config.get("storage", {}).get("temp_dir", "temp")
        cache_dir = config.get("storage", {}).get("cache_dir", "cache")
        
        model_id = args.model
        if model_id in config.get("models", {}):
            model_id = config["models"][model_id]
        
        if args.model == "s1_5":
            model = S1_5(
                model_id=model_id,
                cache_dir=cache_dir
            )
            
            total_start_time = time.time()
            successful_generations = 0
            
            for i in range(args.batch):
                batch_text = f" (batch {i+1}/{args.batch})" if args.batch > 1 else ""
                print(f"Generating text for prompt: '{args.prompt}'{batch_text}")
                
                start_time = time.time()
                
                try:
                    text, is_cached, chat_token = model.generate_text_sync(args.prompt)
                    
                    generation_time = time.time() - start_time
                    cache_text = " (from cache)" if is_cached else ""
                    print(f"Text generated in {generation_time:.2f} seconds{cache_text}")
                    print(f"\nResponse: {text}\n")
                    successful_generations += 1
                    
                except Exception as e:
                    print(f"Error generating text: {e}")
        
        elif args.model == "s2_0":
            model = S2_0(
                model_id=model_id,
                cache_dir=cache_dir
            )
            
            total_start_time = time.time()
            successful_generations = 0
            
            for i in range(args.batch):
                batch_text = f" (batch {i+1}/{args.batch})" if args.batch > 1 else ""
                print(f"Generating text for prompt: '{args.prompt}'{batch_text}")
                
                start_time = time.time()
                
                try:
                    text, is_cached, chat_token = model.generate_text_sync(args.prompt)
                    
                    generation_time = time.time() - start_time
                    cache_text = " (from cache)" if is_cached else ""
                    print(f"Text generated in {generation_time:.2f} seconds{cache_text}")
                    print(f"\nResponse: {text}\n")
                    successful_generations += 1
                    
                except Exception as e:
                    print(f"Error generating text: {e}")
        
        elif args.model == "s3_0":
            model = S3_0(
                model_id=model_id,
                cache_dir=cache_dir
            )
            
            total_start_time = time.time()
            successful_generations = 0
            
            for i in range(args.batch):
                batch_text = f" (batch {i+1}/{args.batch})" if args.batch > 1 else ""
                
                use_reasoning = not args.no_reasoning
                prompt = args.prompt
                
                if prompt.lower().startswith("[reasoning:"):
                    if prompt.lower().startswith("[reasoning:on]"):
                        use_reasoning = True
                        prompt = prompt[len("[reasoning:on]"):].strip()
                    elif prompt.lower().startswith("[reasoning:off]"):
                        use_reasoning = False
                        prompt = prompt[len("[reasoning:off]"):].strip()
                
                print(f"Generating text for prompt: '{prompt}'{batch_text} (reasoning: {'ON' if use_reasoning else 'OFF'})")
                
                start_time = time.time()
                
                try:
                    text, is_cached, chat_token = model.generate_text_sync(
                        prompt, 
                        use_reasoning=use_reasoning
                    )
                    
                    generation_time = time.time() - start_time
                    cache_text = " (from cache)" if is_cached else ""
                    print(f"Text generated in {generation_time:.2f} seconds{cache_text}")
                    print(f"\nResponse: {text}\n")
                    successful_generations += 1
                    
                except Exception as e:
                    print(f"Error generating text: {e}")
        
            generation_params = {
                "width": args.width or config.get("generation", {}).get("default_width", 1024),
                "height": args.height or config.get("generation", {}).get("default_height", 1024),
                "num_inference_steps": args.steps or config.get("generation", {}).get("default_steps", 30),
                "guidance_scale": args.guidance or config.get("generation", {}).get("default_guidance_scale", 7.5),
                "negative_prompt": args.negative_prompt or config.get("generation", {}).get("default_negative_prompt", ""),
                "seed": args.seed,
                "use_half_precision": config.get("optimization", {}).get("half_precision", True),
                "vae_slicing": config.get("optimization", {}).get("vae_slicing", True),
            }
            
            model = GenAZ(
                model_id=model_id, 
                temp_dir=temp_dir,
                cache_dir=cache_dir,
                use_gpu=config.get("optimization", {}).get("use_gpu", True),
                compile_model=config.get("optimization", {}).get("compile_model", False)
            )
            
            total_start_time = time.time()
            successful_generations = 0
            
            for i in range(args.batch):
                batch_text = f" (batch {i+1}/{args.batch})" if args.batch > 1 else ""
                print(f"Generating image for prompt: '{args.prompt}'{batch_text}")
                
                start_time = time.time()
                
                try:
                    image, is_cached = model.generate_image_sync(
                        args.prompt, 
                        **generation_params
                    )
                    
                    generation_time = time.time() - start_time
                    cache_text = " (from cache)" if is_cached else ""
                    print(f"Image generated in {generation_time:.2f} seconds{cache_text}")
                    successful_generations += 1
                    
                    if args.output:
                        if args.batch > 1:
                            base, ext = os.path.splitext(args.output)
                            output_path = f"{base}_{i+1}{ext}"
                        else:
                            output_path = args.output
                        
                        model.save_image(image, output_path, add_watermark=not args.no_watermark)
                        print(f"Image saved to {output_path}")
                    else:
                        if not args.no_watermark:
                            image = model.add_watermark(image)
                    
                    try:
                        image.show()
                    except Exception as e:
                        logger.debug(f"Could not display image: {e}")
                except Exception as e:
                    print(f"Error generating image: {e}")
        
        total_time = time.time() - total_start_time
        if args.batch > 1:
            print(f"Generated {successful_generations}/{args.batch} outputs in {total_time:.2f} seconds")
            if successful_generations > 0:
                print(f"Average time per generation: {total_time/successful_generations:.2f} seconds")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def main():
    config = load_config()
    config = check_system_compatibility(config)
    create_directories(config)

    if not config.get("test_mode", True):
        from server import start_server
        start_server(config)
        return

    parser = argparse.ArgumentParser(description="NexoAI Image and Text Generation")
    parser.add_argument("prompt", type=str, nargs="?", help="Text prompt for generation")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    parser.add_argument("--model", "-m", type=str, default="genaz", 
                        help="Model ID to use (genaz, s1_5, s2_0, s3_0)")
    parser.add_argument("--mode", type=str, choices=["server", "serverless"], default="server",
                        help="Execution mode")
    parser.add_argument("--acceleration", type=str, choices=["cpu", "gpu"], default=None,
                        help="Hardware acceleration (overrides config)")
    parser.add_argument("--batch", "-b", type=int, default=1, help="Number of outputs to generate")
    parser.add_argument("--no-watermark", action="store_true", help="Disable watermark")
    parser.add_argument("--list-models", "-l", action="store_true", help="List available models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--width", type=int, default=None, help="Image width")
    parser.add_argument("--height", type=int, default=None, help="Image height")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning enhancement for S3.0 model")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.list_models:
        print("Available models:")
        for model in config.get("models", {}).get("available", []):
            print(f"  - {model}")
        return

    if not args.prompt:
        parser.print_help()
        return

    if args.acceleration:
        config["optimization"]["use_gpu"] = args.acceleration == "gpu"

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    return asyncio.run(run_async(config, args, parser))

if __name__ == "__main__":
    sys.exit(main() or 0)