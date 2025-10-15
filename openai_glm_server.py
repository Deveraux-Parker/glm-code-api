#!/usr/bin/env python3
"""
OpenAI-Compatible FastAPI Server for GLM API
===========================================

This server provides an OpenAI API-compatible interface for the GLM (General Language Model) API,
adding support for GLM-specific features like reasoning_content and tool_stream.

Key Features:
------------
1. **OpenAI API Compatibility**: Implements the /v1/chat/completions endpoint with OpenAI's spec
2. **Streaming Support**: Full SSE (Server-Sent Events) streaming with delta-based updates
3. **Reasoning Mode**: Exposes GLM's reasoning capabilities via reasoning_effort parameter
4. **Tool Calling**: Complete support for function/tool calling with streaming tool updates
5. **CORS Enabled**: Allows cross-origin requests for web-based clients

Architecture:
------------
- Uses FastAPI for the HTTP server framework
- Employs curl subprocess for streaming GLM API requests (preserves SSE streaming)
- Translates between OpenAI and GLM formats in both directions
- Logs detailed debug information for monitoring and troubleshooting

API Endpoints:
-------------
- GET  /              - Server info and available endpoints
- GET  /health        - Health check endpoint
- POST /v1/chat/completions - Main chat completions endpoint (OpenAI-compatible)

Environment:
-----------
- Requires apikey.json with GLM API key: {"apiKey": "your-api-key-here"}
- Runs on http://0.0.0.0:8000 by default
- Supports uvicorn for production deployment

Usage Example:
-------------
```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "glm-4.6",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": True,
    "reasoning_effort": "high"  # GLM-specific parameter
})

for line in response.iter_lines():
    if line:
        print(line.decode())
```

Author: Deveraux Parker
Version: 1.0.0
"""

import json
import subprocess
import asyncio
import time
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoTokenizer

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GLM-Server")


# ==================== Pydantic Models ====================
# These models define the request/response schemas following OpenAI's API specification
# with extensions for GLM-specific features.

class Message(BaseModel):
    """
    Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender ("user", "assistant", "system", or "tool")
        content: The message content as a string
    """
    role: str
    content: str


class FunctionDefinition(BaseModel):
    """
    Defines a function/tool that the model can call.

    Attributes:
        name: The unique name of the function
        description: Human-readable description of what the function does
        parameters: JSON Schema defining the function's parameters

    Example:
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """
    Wrapper for a function/tool definition.

    Attributes:
        type: Always "function" (for future extensibility)
        function: The function definition
    """
    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completions endpoint.
    Follows OpenAI's API specification with GLM extensions.

    Attributes:
        model: Model identifier (e.g., "glm-4.6")
        messages: List of conversation messages
        temperature: Sampling temperature (0.0-1.0). Higher = more random
        max_tokens: Maximum tokens to generate
        stream: If True, stream response via Server-Sent Events
        reasoning_effort: GLM-specific. Controls reasoning depth: "low", "medium", "high"
        tools: List of available functions the model can call
        tool_choice: How to select tools: "auto", "none", or specific tool name
        tool_stream: GLM-specific. If True, stream tool calls as they're generated
        stop: List of stop sequences. Generation stops when encountered

    Example:
        {
            "model": "glm-4.6",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "stream": true,
            "reasoning_effort": "high"
        }
    """
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=8000, alias="max_completion_tokens")
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = None  # "high", "medium", "low" for reasoning mode
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"  # "auto", "none", or specific tool name
    tool_stream: Optional[bool] = False  # Enable tool streaming
    stop: Optional[List[str]] = None  # Stop tokens


class ChatCompletionChoice(BaseModel):
    """
    A single choice in a non-streaming chat completion response.

    Attributes:
        index: Choice index (usually 0)
        message: The complete generated message
        finish_reason: Why generation stopped ("stop", "length", "tool_calls", etc.)
    """
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """
    Complete (non-streaming) chat completion response.

    Attributes:
        id: Unique completion identifier
        object: Always "chat.completion"
        created: Unix timestamp of completion creation
        model: Model used for generation
        choices: List of generated choices (usually just one)
    """
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class ToolCallFunction(BaseModel):
    """
    Function call details within a tool call.

    Attributes:
        name: Function name (may be partial during streaming)
        arguments: JSON string of function arguments (may be partial during streaming)
    """
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    """
    Represents a tool/function call made by the model.
    During streaming, multiple chunks build up the complete tool call.

    Attributes:
        index: Index of this tool call (0, 1, 2... for multiple calls)
        id: Unique identifier for this tool call
        type: Always "function"
        function: The function call details

    Streaming Behavior:
        Tool calls are streamed incrementally:
        1. First chunk: index, id, type, function.name
        2. Subsequent chunks: index, function.arguments (partial JSON)
        3. Final chunk: Complete arguments string
    """
    index: int
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[ToolCallFunction] = None


class ChatCompletionChunkDelta(BaseModel):
    """
    Delta update in a streaming response chunk.
    Contains the incremental changes since the last chunk.

    Attributes:
        role: Message role (only in first chunk)
        content: Incremental content string
        reasoning_content: GLM-specific. Incremental reasoning/thinking content
        tool_calls: Incremental tool call updates

    Streaming Flow:
        1. First chunk: role="assistant"
        2. Reasoning chunks: reasoning_content with thinking process (if enabled)
        3. Content chunks: content with actual response text
        4. Tool call chunks: tool_calls with function calls (if any)
        5. Final chunk: finish_reason="stop" or "tool_calls"
    """
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None  # GLM uses reasoning_content
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    """
    A single choice in a streaming chunk.

    Attributes:
        index: Choice index (usually 0)
        delta: The incremental update
        finish_reason: Only present in final chunk. Indicates why generation stopped
    """
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """
    A single chunk in a streaming chat completion response.
    Follows SSE (Server-Sent Events) format: "data: {json}\n\n"

    Attributes:
        id: Unique completion identifier (same across all chunks)
        object: Always "chat.completion.chunk"
        created: Unix timestamp (same across all chunks)
        model: Model identifier
        choices: List of choice deltas (usually just one)
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ==================== API Key Management ====================

def load_api_key() -> Optional[str]:
    """
    Load GLM API key from apikey.json file.

    The file should contain JSON in the format:
    {
        "apiKey": "your-api-key-here"
    }

    Returns:
        The API key string, or None if file not found or invalid

    Raises:
        None - errors are caught and logged
    """
    try:
        with open('apikey.json', 'r') as f:
            data = json.load(f)
            return data.get('apiKey')
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None


# ==================== GLM Client ====================

class GLMClient:
    """
    Client for making HTTP requests to the GLM API.

    This class handles both streaming and non-streaming requests to the GLM API,
    using curl subprocess for reliable SSE streaming. It translates between the
    OpenAI-compatible format and GLM's native format.

    Why curl subprocess?
    --------------------
    We use curl as a subprocess instead of Python HTTP libraries because:
    1. Preserves SSE (Server-Sent Events) streaming without buffering
    2. Reliable line-by-line processing of streaming data
    3. Built-in support for HTTP/2 and connection management
    4. Avoids Python async/await complexity with streaming responses

    Attributes:
        api_key: GLM API authentication key
        api_url: GLM API endpoint URL
    """

    def __init__(self, api_key: str):
        """
        Initialize GLM client with API key.

        Args:
            api_key: GLM API authentication key
        """
        self.api_key = api_key
        self.api_url = "https://api.z.ai/api/coding/paas/v4/chat/completions"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "glm-4.6",
        temperature: float = 0.7,
        max_tokens: int = 8000,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        tool_stream: Optional[bool] = False,
        stop: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None] | Dict[str, Any]:
        """
        Make a chat completion request to the GLM API.

        This method constructs a request payload for GLM and routes it to either
        the streaming or non-streaming handler based on the `stream` parameter.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: GLM model identifier (default: "glm-4.6")
            temperature: Sampling temperature 0.0-1.0. Higher values = more random
            max_tokens: Maximum number of tokens to generate
            stream: If True, return async generator yielding SSE-formatted chunks
            reasoning_effort: Enable reasoning mode. Options: "low", "medium", "high"
                            When enabled, model generates reasoning_content before response
            tools: List of tool/function definitions the model can use
            tool_choice: How model selects tools: "auto" (decides), "none" (disabled),
                        or specific tool name (forces that tool)
            tool_stream: If True with stream=True, tool calls are streamed incrementally
            stop: List of strings that stop generation when encountered

        Returns:
            If stream=False: Dict with complete response
            If stream=True: AsyncGenerator yielding SSE lines ("data: {json}\n\n")

        Example (Streaming with Reasoning):
            ```python
            async for chunk in client.chat_completion(
                messages=[{"role": "user", "content": "Explain quantum computing"}],
                stream=True,
                reasoning_effort="high"
            ):
                print(chunk)
            ```

        Note on Reasoning:
            When reasoning_effort is set, the response will include both:
            1. reasoning_content: The model's internal thinking process
            2. content: The actual response to the user
            The reasoning is generated first, then the content.
        """
        # Build GLM request payload matching GLM API spec
        request = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": stream,
            "temperature": temperature
        }

        # Add GLM-specific reasoning_effort parameter if specified
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort

        # Add tool/function calling configuration if tools provided
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        # Enable incremental tool call streaming (requires stream=True)
        if tool_stream and stream:
            request["tool_stream"] = True

        # Add stop sequences for early termination
        if stop:
            request["stop"] = stop

        # Request usage statistics in final streaming chunk
        if stream:
            request["stream_options"] = {
                "include_usage": True
            }

        # Route to appropriate handler
        if stream:
            return self._stream_completion(request)
        else:
            return await self._non_stream_completion(request)

    async def _non_stream_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make non-streaming completion request"""

        curl_command = [
            "curl", "-X", "POST",
            self.api_url,
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {self.api_key}",
            "-d", json.dumps(request)
        ]

        # Run curl command
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"GLM API error: {stderr.decode()}")

        try:
            response = json.loads(stdout.decode())
            return response
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON response: {e}")

    async def _stream_completion(self, request: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Make streaming completion request"""

        logger.info(f"Starting GLM streaming request with reasoning_effort: {request.get('reasoning_effort', 'None')}")
        logger.debug(f"Full request: {json.dumps(request, indent=2)}")

        curl_command = [
            "curl", "-X", "POST",
            self.api_url,
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {self.api_key}",
            "-H", "Accept: text/event-stream",
            "-d", json.dumps(request),
            "--no-buffer"
        ]

        # Start curl process
        process = await asyncio.create_subprocess_exec(
            *curl_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Stream output line by line
        chunk_count = 0
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line = line.decode().strip()

            # Log raw line
            if line:
                logger.debug(f"Raw GLM line: {line}")

            # Parse SSE format: "data: {...}"
            if line.startswith('data: '):
                data_part = line[6:]
                if data_part == '[DONE]':
                    logger.info(f"Stream ended after {chunk_count} chunks")
                    yield 'data: [DONE]\n\n'
                    break

                try:
                    # Parse and re-yield the chunk
                    chunk_data = json.loads(data_part)
                    chunk_count += 1

                    # Log chunk details
                    if 'choices' in chunk_data and chunk_data['choices']:
                        delta = chunk_data['choices'][0].get('delta', {})
                        if delta.get('reasoning_content'):
                            logger.info(f"Chunk {chunk_count}: REASONING_CONTENT - {len(delta['reasoning_content'])} chars")
                        if delta.get('content'):
                            logger.info(f"Chunk {chunk_count}: CONTENT - {len(delta['content'])} chars")
                        if delta.get('role'):
                            logger.info(f"Chunk {chunk_count}: ROLE - {delta['role']}")
                        if delta.get('tool_calls'):
                            logger.info(f"Chunk {chunk_count}: TOOL_CALLS - {json.dumps(delta['tool_calls'])}")

                        # Log the full delta for debugging
                        logger.debug(f"Chunk {chunk_count} delta: {json.dumps(delta, indent=2)}")

                    yield f"data: {json.dumps(chunk_data)}\n\n"
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse chunk: {e} - Data: {data_part}")
                    continue

        await process.wait()


# ==================== Token Counting ====================

def count_tokens(text: str) -> int:
    """
    Count tokens in a text string using the GLM tokenizer.

    Args:
        text: Input text to tokenize

    Returns:
        Number of tokens, or 0 if tokenizer not available
    """
    if not glm_tokenizer:
        return 0

    try:
        tokens = glm_tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        return 0


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Count tokens in a list of messages.
    Includes overhead for message formatting.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        Total token count including formatting overhead
    """
    if not glm_tokenizer:
        return 0

    total_tokens = 0
    for msg in messages:
        # Count role
        if msg.get('role'):
            total_tokens += count_tokens(msg['role'])

        # Count content
        if msg.get('content'):
            total_tokens += count_tokens(msg['content'])

        # Add formatting overhead (role markers, etc.)
        total_tokens += 4  # Approximate overhead per message

    return total_tokens


# ==================== FastAPI App ====================

app = FastAPI(
    title="OpenAI-Compatible GLM Server",
    description="OpenAI API spec compatible server using GLM backend with reasoning support",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Global GLM client and tokenizer
glm_client: Optional[GLMClient] = None
glm_tokenizer: Optional[AutoTokenizer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize GLM client and tokenizer on startup"""
    global glm_client, glm_tokenizer

    api_key = load_api_key()
    if not api_key:
        print("WARNING: No API key found in apikey.json")
        print("Server will start but requests will fail")
    else:
        glm_client = GLMClient(api_key)
        print(f"✅ GLM client initialized")

    # Initialize tokenizer
    try:
        print("🔤 Loading GLM tokenizer...")
        glm_tokenizer = AutoTokenizer.from_pretrained(
            'THUDM/glm-4-9b',
            trust_remote_code=True
        )
        print(f"✅ GLM tokenizer initialized (vocab size: {glm_tokenizer.vocab_size})")
    except Exception as e:
        print(f"⚠️  Failed to load tokenizer: {e}")
        print("   Token counting will be disabled")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OpenAI-Compatible GLM Server",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "glm_client": "initialized" if glm_client else "not initialized"
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint

    Supports:
    - Streaming and non-streaming responses
    - Temperature control (0.0-1.0)
    - Reasoning mode via reasoning_effort parameter
    """

    if not glm_client:
        raise HTTPException(status_code=500, detail="GLM client not initialized. Check API key.")

    # Convert messages to dict format
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Convert tools to dict format if provided
    tools = None
    if request.tools:
        tools = [{"type": tool.type, "function": tool.function.model_dump()} for tool in request.tools]

    # Handle streaming response
    if request.stream:
        async def generate_stream():
            """Generate OpenAI-compatible streaming response"""

            request_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())

            # Count prompt tokens
            prompt_tokens = count_messages_tokens(messages)

            # Initialize completion token counter
            completion_tokens = 0
            accumulated_completion_text = ""

            # Prepare messages - append /nothink to last user message if reasoning disabled
            modified_messages = messages.copy()
            if not request.reasoning_effort:
                # Find the last user message and append /nothink
                for i in range(len(modified_messages) - 1, -1, -1):
                    if modified_messages[i]["role"] == "user":
                        modified_messages[i] = modified_messages[i].copy()
                        modified_messages[i]["content"] = modified_messages[i]["content"] + " /nothink"
                        break

            # Build GLM request
            glm_request = {
                "messages": modified_messages,
                "model": request.model,
                "max_completion_tokens": request.max_tokens,
                "stream": True,
                "temperature": request.temperature
            }

            if request.reasoning_effort:
                glm_request["reasoning_effort"] = request.reasoning_effort

            if tools:
                glm_request["tools"] = tools
                glm_request["tool_choice"] = request.tool_choice

            if request.tool_stream:
                glm_request["tool_stream"] = True

            # Handle stop tokens - pass through as-is
            if request.stop:
                glm_request["stop"] = request.stop

            glm_request["stream_options"] = {
                "include_usage": True
            }

            # Automatically handle </think> detection when reasoning is disabled
            # This provides additional safety in case model outputs thinking despite /nothink
            accumulated_content = ""
            should_stop_on_think = request.reasoning_effort is None

            # Send the actual GLM request as the first chunk for transparency
            # This allows clients to see exactly what was sent to GLM (including /nothink)
            actual_request_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta()
                    )
                ]
            )
            # Add custom field with actual GLM request
            actual_request_data = actual_request_chunk.model_dump()
            actual_request_data['actual_glm_request'] = glm_request
            yield f"data: {json.dumps(actual_request_data)}\n\n"

            async for chunk_line in glm_client._stream_completion(glm_request):
                if chunk_line.strip() == 'data: [DONE]':
                    # Count completion tokens
                    if accumulated_completion_text:
                        completion_tokens = count_tokens(accumulated_completion_text)

                    # Send usage chunk before [DONE]
                    if glm_tokenizer:  # Only send usage if tokenizer is available
                        usage_chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens
                            }
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"

                    yield 'data: [DONE]\n\n'
                    break

                # Parse GLM chunk
                if chunk_line.startswith('data: '):
                    try:
                        glm_chunk = json.loads(chunk_line[6:])

                        # Extract delta from GLM response
                        if 'choices' in glm_chunk:
                            glm_delta = glm_chunk['choices'][0].get('delta', {})

                            # When reasoning is disabled, GLM may put response in reasoning_content
                            # Move it to content field for proper display
                            if should_stop_on_think and 'reasoning_content' in glm_delta:
                                glm_delta = glm_delta.copy()
                                reasoning_text = glm_delta.get('reasoning_content', '')
                                # Move reasoning_content to content (strip leading/trailing whitespace)
                                if 'content' in glm_delta:
                                    glm_delta['content'] = glm_delta['content'] + reasoning_text.strip()
                                else:
                                    glm_delta['content'] = reasoning_text.strip()
                                # Remove reasoning_content so client doesn't see it
                                del glm_delta['reasoning_content']

                            content = glm_delta.get('content', '')

                            # Check for </think> in content when reasoning is disabled
                            if should_stop_on_think and content:
                                accumulated_content += content

                                # Check if </think> appears in accumulated content
                                if '</think>' in accumulated_content:
                                    # Find where </think> starts
                                    think_index = accumulated_content.find('</think>')

                                    # If </think> is in current chunk, truncate it
                                    if '</think>' in content:
                                        content_before_think = content.split('</think>')[0]
                                        glm_delta['content'] = content_before_think

                                        # Build final chunk with truncated content if any
                                        if content_before_think:
                                            openai_chunk = ChatCompletionChunk(
                                                id=request_id,
                                                created=created,
                                                model=request.model,
                                                choices=[
                                                    ChatCompletionChunkChoice(
                                                        index=0,
                                                        delta=ChatCompletionChunkDelta(
                                                            role=glm_delta.get('role'),
                                                            content=content_before_think,
                                                            reasoning_content=glm_delta.get('reasoning_content'),
                                                            tool_calls=glm_delta.get('tool_calls')
                                                        ),
                                                        finish_reason="stop"
                                                    )
                                                ]
                                            )
                                            yield f"data: {openai_chunk.model_dump_json()}\n\n"

                                        # Send final chunk with finish_reason
                                        final_chunk = ChatCompletionChunk(
                                            id=request_id,
                                            created=created,
                                            model=request.model,
                                            choices=[
                                                ChatCompletionChunkChoice(
                                                    index=0,
                                                    delta=ChatCompletionChunkDelta(),
                                                    finish_reason="stop"
                                                )
                                            ]
                                        )
                                        yield f"data: {final_chunk.model_dump_json()}\n\n"

                                        # Count completion tokens before </think>
                                        if accumulated_completion_text:
                                            completion_tokens = count_tokens(accumulated_completion_text.split('</think>')[0])

                                        # Send usage chunk
                                        if glm_tokenizer:
                                            usage_chunk = {
                                                "id": request_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": request.model,
                                                "choices": [],
                                                "usage": {
                                                    "prompt_tokens": prompt_tokens,
                                                    "completion_tokens": completion_tokens,
                                                    "total_tokens": prompt_tokens + completion_tokens
                                                }
                                            }
                                            yield f"data: {json.dumps(usage_chunk)}\n\n"

                                        yield 'data: [DONE]\n\n'
                                        break

                            # Build OpenAI-compatible chunk
                            # Only include delta fields that are actually present and non-empty
                            delta_kwargs = {}
                            if 'role' in glm_delta and glm_delta['role']:
                                delta_kwargs['role'] = glm_delta['role']
                            if 'content' in glm_delta and glm_delta['content']:
                                delta_kwargs['content'] = glm_delta['content']
                                # Accumulate content for token counting
                                accumulated_completion_text += glm_delta['content']
                            if 'reasoning_content' in glm_delta and glm_delta['reasoning_content']:
                                delta_kwargs['reasoning_content'] = glm_delta['reasoning_content']
                            if 'tool_calls' in glm_delta and glm_delta['tool_calls']:
                                delta_kwargs['tool_calls'] = glm_delta['tool_calls']

                            openai_chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(**delta_kwargs),
                                        finish_reason=glm_chunk['choices'][0].get('finish_reason')
                                    )
                                ]
                            )

                            yield f"data: {openai_chunk.model_dump_json(exclude_none=True)}\n\n"

                    except json.JSONDecodeError:
                        continue

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

    # Handle non-streaming response
    else:
        # Build GLM request
        glm_request = {
            "messages": messages,
            "model": request.model,
            "max_completion_tokens": request.max_tokens,
            "stream": False,
            "temperature": request.temperature
        }

        if request.reasoning_effort:
            glm_request["reasoning_effort"] = request.reasoning_effort

        glm_response = await glm_client._non_stream_completion(glm_request)

        # Convert GLM response to OpenAI format
        if 'choices' not in glm_response:
            raise HTTPException(status_code=500, detail="Invalid GLM response")

        glm_choice = glm_response['choices'][0]
        content = glm_choice.get('message', {}).get('content', '')

        # Build OpenAI-compatible response
        openai_response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason=glm_choice.get('finish_reason', 'stop')
                )
            ]
        )

        return openai_response


# ==================== Main ====================

if __name__ == "__main__":
    print("🚀 Starting OpenAI-Compatible GLM Server")
    print("=" * 60)
    print("📡 Endpoints:")
    print("   - Chat Completions: POST /v1/chat/completions")
    print("   - Health Check: GET /health")
    print("")
    print("✨ Features:")
    print("   - ✅ Streaming and non-streaming responses")
    print("   - ✅ Temperature control (0.0-1.0)")
    print("   - ✅ Reasoning mode support")
    print("   - ✅ OpenAI API spec compatible")
    print("")
    print("🔧 Starting server on http://0.0.0.0:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
