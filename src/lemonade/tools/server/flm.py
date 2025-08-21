import os
import logging
import subprocess
import requests
import json
import uuid
import time
from typing import Dict, Any

from lemonade_server.pydantic_models import (
    PullConfig,
    ChatCompletionRequest,
)

from lemonade.tools.server.wrapped_server import WrappedServerTelemetry, WrappedServer
from lemonade.tools.flm.utils import install_flm
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse


class FlmTelemetry(WrappedServerTelemetry):
    """
    Manages telemetry data collection and display for FLM server.
    """

    def parse_telemetry_line(self, line: str):
        """
        Parse telemetry data from FLM server output lines.
        """

        # TODO: parse perf data

        return


class FlmServer(WrappedServer):
    """
    Routes OpenAI API requests to an FLM server instance and returns the result
    back to Lemonade Server.
    """

    def __init__(self):
        super().__init__(server_name="flm-server", telemetry=FlmTelemetry())

    def _choose_port(self):
        """
        `flm serve` doesn't support port selection as of v0.9.4
        """
        self.port = 11434

    def address(self):
        """
        `flm serve` doesn't support host name selection as of v0.9.4
        """

        return f"http://localhost:{self.port}/v1"

    def install_server(self):
        """
        Check if FLM is installed and at minimum version.
        If not, download and run the GUI installer, then wait for completion.
        """
        install_flm()

    def download_model(
        self, config_checkpoint, config_mmproj=None, do_not_upgrade=False
    ) -> dict:
        command = ["flm", "pull", f"{config_checkpoint}"]

        subprocess.run(command, check=True)

    def _launch_server_subprocess(
        self,
        model_config: PullConfig,
        snapshot_files: dict,
        ctx_size: int,
        supports_embeddings: bool = False,
        supports_reranking: bool = False,
    ):

        # This call is a placeholder for now; eventually we'll pass the
        # port into the command below when its supported
        self._choose_port()

        command = ["flm", "serve", f"{model_config.checkpoint}"]

        # Set up environment with library path for Linux
        env = os.environ.copy()

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

    def _wait_for_load(self):
        """
        FLM doesn't seem to have a health API, so we'll use the "list local models"
        API to check if the server is up.
        """
        status_code = None
        while not self.process.poll() and status_code != 200:
            health_url = f"http://localhost:{self.port}/api/tags"
            try:
                health_response = requests.get(health_url)
            except requests.exceptions.ConnectionError:
                logging.debug(
                    "Not able to connect to %s yet, will retry", self.server_name
                )
            else:
                status_code = health_response.status_code
                logging.debug(
                    "Testing %s readiness (will retry until ready), result: %s",
                    self.server_name,
                    health_response.json(),
                )
            time.sleep(1)

    def _convert_openai_to_ollama_message(
        self, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert OpenAI format message to Ollama format.

        OpenAI format for images:
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }

        Ollama format for images:
        {
            "role": "user",
            "content": "What's in this image?",
            "images": ["base64_image_data"]
        }
        """
        ollama_message = {"role": message["role"]}

        # Handle content
        if isinstance(message.get("content"), str):
            # Simple text message
            ollama_message["content"] = message["content"]
        elif isinstance(message.get("content"), list):
            # Multi-part content (text + images)
            text_parts = []
            images = []

            for part in message["content"]:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    # Extract base64 data from data URL
                    image_url = part["image_url"]["url"]
                    if image_url.startswith("data:"):
                        # Format: data:image/jpeg;base64,<base64_data>
                        base64_data = image_url.split(",", 1)[1]
                        images.append(base64_data)
                    else:
                        # Handle remote URL by downloading and converting to base64
                        logging.warning(
                            "Remote image URLs not yet supported: %s", image_url
                        )

            ollama_message["content"] = " ".join(text_parts) if text_parts else ""
            if images:
                ollama_message["images"] = images
        else:
            ollama_message["content"] = message.get("content", "")
        return ollama_message

    def _convert_openai_to_ollama_request(
        self, chat_completion_request: ChatCompletionRequest
    ) -> Dict[str, Any]:
        """
        Convert OpenAI ChatCompletionRequest to Ollama chat API format.
        """
        # Convert messages
        ollama_messages = []
        for message in chat_completion_request.messages:
            ollama_message = self._convert_openai_to_ollama_message(message)
            ollama_messages.append(ollama_message)

        # Get the actual checkpoint name for FLM
        model_checkpoint = self._get_model_checkpoint(chat_completion_request.model)

        # Build Ollama request
        ollama_request = {
            "model": model_checkpoint,
            "messages": ollama_messages,
            "stream": chat_completion_request.stream or False,
        }

        # Map OpenAI parameters to Ollama options
        options = {}
        if chat_completion_request.temperature is not None:
            options["temperature"] = chat_completion_request.temperature
        if chat_completion_request.top_p is not None:
            options["top_p"] = chat_completion_request.top_p
        if chat_completion_request.max_tokens is not None:
            options["num_predict"] = chat_completion_request.max_tokens
        elif chat_completion_request.max_completion_tokens is not None:
            options["num_predict"] = chat_completion_request.max_completion_tokens
        if chat_completion_request.stop is not None:
            if isinstance(chat_completion_request.stop, str):
                options["stop"] = [chat_completion_request.stop]
            else:
                options["stop"] = chat_completion_request.stop
        if chat_completion_request.repeat_penalty is not None:
            options["repeat_penalty"] = chat_completion_request.repeat_penalty

        if options:
            ollama_request["options"] = options

        return ollama_request

    def chat_completion(self, chat_completion_request: ChatCompletionRequest):
        """
        Override the parent method to convert OpenAI format to Ollama format
        and handle image inputs properly.
        """
        # Convert OpenAI request to Ollama format
        ollama_request = self._convert_openai_to_ollama_request(chat_completion_request)

        # Debug: Log the actual request being sent to FLM
        logging.debug("Sending to FLM: %s", json.dumps(ollama_request, indent=2))

        # Make request to Ollama API endpoint
        ollama_url = f"http://localhost:{self.port}/api/chat"

        try:
            if chat_completion_request.stream:
                # Streaming response
                def event_stream():
                    try:
                        # Retry logic for 503 Service Unavailable errors
                        max_retries = 3
                        retry_delay = 2  # seconds

                        for attempt in range(max_retries):
                            try:
                                response = requests.post(
                                    ollama_url,
                                    json=ollama_request,
                                    stream=True,
                                    timeout=None,
                                )
                                response.raise_for_status()
                                break  # Success, exit retry loop
                            except requests.exceptions.HTTPError as e:
                                if (
                                    response.status_code == 503
                                    and attempt < max_retries - 1
                                ):
                                    logging.debug(
                                        "FLM server returned 503, retrying in %d seconds... (attempt %d/%d)",
                                        retry_delay,
                                        attempt + 1,
                                        max_retries,
                                    )
                                    time.sleep(retry_delay)
                                    continue
                                else:
                                    # Re-raise if it's not a 503 or we've exhausted retries
                                    raise

                        for line in response.iter_lines():
                            if line:
                                try:
                                    # Parse Ollama response
                                    ollama_chunk = json.loads(line.decode("utf-8"))

                                    # Convert to OpenAI format
                                    openai_chunk = self._convert_ollama_to_openai_chunk(
                                        ollama_chunk, chat_completion_request.model
                                    )

                                    yield f"data: {json.dumps(openai_chunk)}\n\n"

                                    if ollama_chunk.get("done", False):
                                        break

                                except json.JSONDecodeError:
                                    continue

                        yield "data: [DONE]\n\n"

                        # Show telemetry after completion
                        self.telemetry.show_telemetry()

                    except Exception as e:
                        logging.error(
                            "Error during streaming chat completion: %s", str(e)
                        )
                        yield f'data: {{"error": "{str(e)}"}}\n\n'

                return StreamingResponse(
                    event_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                # Non-streaming response with retry logic for 503 errors
                max_retries = 3
                retry_delay = 2  # seconds

                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            ollama_url, json=ollama_request, timeout=None
                        )
                        response.raise_for_status()
                        break  # Success, exit retry loop
                    except requests.exceptions.HTTPError as e:
                        if response.status_code == 503 and attempt < max_retries - 1:
                            logging.debug(
                                "FLM server returned 503, retrying in %d seconds... (attempt %d/%d)",
                                retry_delay,
                                attempt + 1,
                                max_retries,
                            )
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Re-raise if it's not a 503 or we've exhausted retries
                            raise

                ollama_response = response.json()
                openai_response = self._convert_ollama_to_openai_response(
                    ollama_response, chat_completion_request.model
                )

                # Show telemetry after completion
                self.telemetry.show_telemetry()

                return openai_response

        except requests.exceptions.RequestException as e:
            logging.error("Error communicating with FLM server: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"FLM server error: {str(e)}",
            ) from e
        except Exception as e:
            logging.error("Unexpected error during chat completion: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat completion error: {str(e)}",
            ) from e

    def _convert_ollama_to_openai_chunk(
        self, ollama_chunk: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """
        Convert Ollama streaming response chunk to OpenAI format.
        """

        openai_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }

        # Handle content
        if "message" in ollama_chunk and "content" in ollama_chunk["message"]:
            openai_chunk["choices"][0]["delta"]["content"] = ollama_chunk["message"][
                "content"
            ]

        # Handle completion
        if ollama_chunk.get("done", False):
            openai_chunk["choices"][0]["finish_reason"] = "stop"
            openai_chunk["choices"][0]["delta"] = {}

        return openai_chunk

    def _convert_ollama_to_openai_response(
        self, ollama_response: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """
        Convert Ollama response to OpenAI format.
        """

        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ollama_response.get("message", {}).get(
                            "content", ""
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": (
                    ollama_response.get("prompt_eval_count", 0)
                    + ollama_response.get("eval_count", 0)
                ),
            },
        }

        return openai_response

    def _get_model_checkpoint(self, model_name: str) -> str:
        """
        Get the actual checkpoint name for FLM from the server models configuration.
        """
        from lemonade_server.model_manager import ModelManager

        try:
            supported_models = ModelManager().supported_models
            model_info = supported_models.get(model_name, {})
            checkpoint = model_info.get("checkpoint", model_name)
            return checkpoint
        except Exception:
            # If we can't determine, use the original model name
            return model_name
