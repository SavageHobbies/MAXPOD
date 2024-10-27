"""MaxPOD FastAPI server for handling pattern generation and mockups."""
import os
import random
import json
import uuid
import socket
import traceback
import logging
from datetime import datetime
from urllib.parse import quote
import subprocess
import threading
import webbrowser
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from util.printify.printify_util import PrintifyUtil
from util.ai_util import AiUtil, AIProvider
from util.image_util import create_text_image
from util.github_util import GithubUploader
from util.llm_config import ALL_CONFIGS
from res.models.tshirt import TshirtFromAiList, TshirtWithIds
from res.models.requests import PatternRequest, MockupRequest
from res.models.responses import PatternResponse, HealthcheckResponse, MockupResponse

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('maxpod.log', mode='w'),  # 'w' mode to start fresh each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add file handler for errors specifically
error_handler = logging.FileHandler('maxpod.error.log', mode='w')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(error_handler)

# Load environment variables from .env file
load_dotenv('.env')

# Random Seeding
random.seed(int(datetime.now().timestamp()))

# Initialize FastAPI
app = FastAPI(title="MaxPOD API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def validate_git_config():
    """Validate GitHub configuration."""
    required_vars = ['GH_UPLOAD_REPO', 'GH_PAT', 'GH_CONTENT_PREFIX']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        error_msg = f"Missing required GitHub environment variables: {', '.join(missing)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def test_ollama_connection(model: str):
    """Test connection to Ollama server and model availability."""
    import requests
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code != 200:
            raise ConnectionError(f"Ollama server returned status code {response.status_code}")
        
        models = response.json().get('models', [])
        if not any(m.get('name') == model for m in models):
            raise ValueError(f"Model {model} not available in Ollama. Available models: {[m.get('name') for m in models]}")
            
    except requests.ConnectionError:
        raise ConnectionError("Failed to connect to Ollama server. Is it running?")


def process_patterns_and_idea(number_of_patterns, idea, llm_config=None):
    """Process patterns and ideas using the specified LLM configuration."""
    try:
        logger.info(f"Starting pattern generation for idea: {idea}, count: {number_of_patterns}, llm: {llm_config}")
        
        # Validate GitHub configuration first
        validate_git_config()
        
        text_colors = [
            {"hex": "000000", "shade": "dark"},
            {"hex": "FFFFFF", "shade": "light"}
        ]

        # Initialize AI
        if llm_config and llm_config in ALL_CONFIGS:
            config = ALL_CONFIGS[llm_config]
            logger.debug(f"Using config: {config}")
            if config.provider == "ollama":
                test_ollama_connection(config.model)
            ai = AiUtil(
                provider=AIProvider(config.provider),
                model=config.model,
                api_key=config.api_key
            )
        else:
            logger.debug("Using default Ollama configuration")
            test_ollama_connection("nemotron-mini:4b-instruct-q4_K_M")
            ai = AiUtil()

        # Initialize and set up Printify
        logger.debug("Initializing Printify")
        printify = PrintifyUtil()
        blueprint = 6  # Unisex Gildan T-Shirt
        printer = 99  # Printify Choice Provider
        variants, light_ids, dark_ids = printify.get_all_variants(
            blueprint, printer)
        
        for color in text_colors:
            if color.get("shade") == "light":
                color["variant_ids"] = dark_ids
            else:
                color["variant_ids"] = light_ids

        # Get patterns from AI
        logger.info("Requesting patterns from AI")
        try:
            response = ai.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot"},
                    {"role": "user", "content": f"Generate {number_of_patterns} t-shirt designs based on this idea: {idea}"}
                ],
                output_model=TshirtFromAiList,
            )
            logger.debug(f"AI Response: {response}")
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"AI generation failed: {str(e)}")

        # Parse the response
        try:
            logger.debug("Parsing AI response")
            parsed_response = TshirtFromAiList.model_validate_json(response)
            patterns = parsed_response.patterns
            logger.info(f"Generated {len(patterns)} patterns")
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            logger.error(f"Raw response: {response}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to parse AI response: {str(e)}")

        # Get the current date and time
        current_time = datetime.now()

        # Create images and push to GitHub
        logger.info("Generating images")
        folder_name = f"./img/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        for pattern in patterns:
            try:
                pattern.uuid = str(uuid.uuid4())
                for color in text_colors:
                    hex_value = color.get("hex")
                    file_name = f"{folder_name}/{pattern.uuid}{hex_value}.png"
                    logger.debug(f"Creating image: {file_name}")
                    create_text_image(
                        text=pattern.tshirt_text,
                        height=2000,
                        width=2000,
                        file_name=file_name,
                        color="#" + hex_value
                    )
            except Exception as e:
                logger.error(f"Error generating image for pattern {pattern.product_name}: {str(e)}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Image generation failed: {str(e)}")

        # Upload to GitHub
        try:
            logger.info("Uploading images to GitHub")
            directory_with_images = f"{folder_name}/"
            github_repository_url = os.getenv("GH_UPLOAD_REPO")
            personal_access_token = os.getenv("GH_PAT")
            uploader = GithubUploader(
                directory_with_images,
                github_repository_url,
                personal_access_token
            )
            uploader.upload()
        except Exception as e:
            logger.error(f"GitHub upload failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"GitHub upload failed: {str(e)}")

        # Process Printify uploads
        logger.info("Processing Printify uploads")
        url_prefix = os.getenv("GH_CONTENT_PREFIX")
        for pattern in patterns:
            try:
                logger.debug(f"Processing pattern: {pattern.product_name}")
                for color in text_colors:
                    hex_value = color.get("hex")
                    image_url = f"{url_prefix}/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}/{quote(pattern.uuid)}{hex_value}.png"
                    logger.debug(f"Uploading image to Printify: {image_url}")
                    image_id = printify.upload_image(image_url)
                    color["image_id"] = image_id

                logger.debug(f"Creating Printify product: {pattern.product_name}")
                product = printify.create_product(
                    blueprint_id=blueprint,
                    print_provider_id=printer,
                    variants=variants,
                    title=pattern.product_name,
                    description=pattern.description,
                    marketing_tags=pattern.marketing_tags,
                    text_colors=text_colors
                )

                pattern.product_id = product
                pattern.image_ids = [color.get("image_id") for color in text_colors]

                logger.debug(f"Publishing product: {product}")
                printify.publish_product(product)
            except Exception as e:
                logger.error(f"Printify processing failed for pattern {pattern.product_name}: {str(e)}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Printify processing failed: {str(e)}")

        logger.info("Pattern generation completed successfully")
        return [TshirtWithIds.model_validate(p.model_dump()) for p in patterns]
        
    except Exception as e:
        logger.error("Error in process_patterns_and_idea:")
        logger.error(traceback.format_exc())
        raise


# FastAPI Endpoints

@app.post("/process_patterns", response_model=PatternResponse)
async def process_patterns(request: PatternRequest):
    """Generate t-shirt patterns based on the provided idea."""
    try:
        logger.info(f"Received pattern request: {request}")
        patterns = process_patterns_and_idea(
            request.patterns,
            request.idea,
            request.llm_config
        )
        return PatternResponse(
            message="Generated Patterns Successfully",
            patterns=patterns
        )
    except Exception as e:
        error_msg = f"Failed to generate patterns: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )


@app.post("/generate_mockups", response_model=MockupResponse)
async def generate_mockups(request: MockupRequest):
    """Generate product mockups using the provided templates and designs."""
    try:
        logger.info(f"Received mockup request: {request}")
        # Implement mockup generation logic here
        # This is a placeholder that returns a sample response
        mockups = [
            {
                "id": str(uuid.uuid4()),
                "mockupKey": f"mockup_{i}.png",
                "templateName": f"Template {i}",
                "designName": f"Design {i}"
            }
            for i in range(len(request.templates))
        ]
        return MockupResponse(
            message="Generated Mockups Successfully",
            mockups=mockups
        )
    except Exception as e:
        error_msg = f"Failed to generate mockups: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )


@app.get("/", response_model=HealthcheckResponse)
@app.get("/healthcheck", response_model=HealthcheckResponse)
async def healthcheck():
    """Check if the API is running."""
    try:
        # Test Ollama connection
        test_ollama_connection("nemotron-mini:4b-instruct-q4_K_M")
        
        # Test GitHub config
        validate_git_config()
        
        return HealthcheckResponse(status="OK")
    except Exception as e:
        logger.error(f"Healthcheck failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "message": str(e)}
        )


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True


def get_next_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Get the next available port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"No available ports found after {max_attempts} attempts")


def run_fastapi(port: int):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


def run_react():
    """Start the React development server."""
    os.chdir("Front-MAXPOD")
    subprocess.run("npm start", shell=True, check=True)


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="MaxPOD - Product Pipeline")
    parser.add_argument('-p', '--patterns', type=int, default=3,
                      help='Number of patterns, default is 3')
    parser.add_argument('idea', type=str, nargs='?',
                      help='The idea to generate patterns for')
    parser.add_argument('--no-ui', action='store_true',
                      help='Run without the React UI')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port for the FastAPI server')
    
    args = parser.parse_args()

    try:
        # Test Ollama connection at startup
        test_ollama_connection("nemotron-mini:4b-instruct-q4_K_M")
        
        # Test GitHub config
        validate_git_config()
        
        # Find an available port
        port = get_next_available_port(args.port)
        logger.info(f"Starting FastAPI server on port {port}")
        
        # Start FastAPI in a thread
        api_thread = threading.Thread(target=run_fastapi, args=(port,))
        api_thread.daemon = True
        api_thread.start()

        # Wait for the API to start
        time.sleep(2)

        if not args.no_ui:
            # Start React in the main thread
            try:
                run_react()
            except KeyboardInterrupt:
                logger.info("\nShutting down gracefully...")
            except Exception as e:
                logger.error(f"Error running React: {e}")
                logger.error(traceback.format_exc())

        if args.idea:
            # If an idea was provided, process it
            try:
                patterns = process_patterns_and_idea(args.patterns, args.idea)
                logger.info(f"Generated {len(patterns)} patterns for idea: {args.idea}")
            except Exception as e:
                logger.error(f"Error processing patterns: {e}")
                logger.error(traceback.format_exc())

        # Keep the main thread running if only running the API
        if args.no_ui:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nShutting down gracefully...")
                
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.error(traceback.format_exc())
        exit(1)