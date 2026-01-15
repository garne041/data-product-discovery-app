import json
import logging
from typing import Iterator, Dict, Any, Optional
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

def _get_endpoint_task_type(endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    w = WorkspaceClient()
    ep = w.serving_endpoints.get(endpoint_name)
    return ep.task

def is_endpoint_supported(endpoint_name: str) -> bool:
    """Check if the endpoint has a supported task type - now supports all conversational endpoints."""
    try:
        task_type = _get_endpoint_task_type(endpoint_name)
        # Support all agent and chat task types, including custom RAG agents
        supported_prefixes = ["agent/", "llm/v1/chat", "llm/v1/completions"]
        is_supported = any(task_type.startswith(prefix) for prefix in supported_prefixes)
        logger.info(f"Endpoint {endpoint_name} has task type: {task_type}, supported: {is_supported}")
        return is_supported
    except Exception as e:
        logger.warning(f"Could not determine task type for endpoint {endpoint_name}: {e}")
        # If we can't determine task type, assume it's supported and let the query fail with a better error
        return True

def _convert_messages_to_input(messages: list[dict[str, str]]) -> list:
    """
    Convert messages format to input format for custom agent endpoints.
    Returns the full messages array as expected by ResponsesAgentRequest schema.
    
    The endpoint expects: {'input': [{'role': 'user', 'content': '...'}, ...]}
    Not: {'input': ['string']}
    
    This preserves conversation history for multi-turn interactions.
    """
    # Return the full messages array - the endpoint expects Message objects, not strings
    if messages:
        return messages
    
    # Fallback for empty messages
    return [{"role": "user", "content": ""}]

def query_endpoint_stream(endpoint_name: str, messages: list[dict[str, str]], max_tokens: int = 1000) -> Iterator[Dict[str, Any]]:
    """
    Query an agent endpoint with streaming support.
    Yields parsed chunks from the streaming response.
    """
    client = get_deploy_client('databricks')
    
    # Prepare input in custom agent format for streaming
    inputs = {
        'input': _convert_messages_to_input(messages),
        'max_output_tokens': max_tokens,
        'stream': True
    }
    
    try:
        response = client.predict(
            endpoint=endpoint_name,
            inputs=inputs
        )
        
        # Handle streaming response
        for chunk in response:
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            
            # Parse SSE format (data: {...})
            if isinstance(chunk, str):
                for line in chunk.strip().split('\n'):
                    if line.startswith('data: '):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        if json_str.strip() == '[DONE]':
                            continue
                        try:
                            data = json.loads(json_str)
                            yield data
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON: {json_str}")
                            continue
            elif isinstance(chunk, dict):
                yield chunk
                
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        raise

def query_endpoint(endpoint_name: str, messages: list[dict[str, str]], max_tokens: int = 1000) -> dict:
    """
    Query an agent endpoint without streaming (fallback method).
    Returns the complete response.
    """
    client = get_deploy_client('databricks')
    
    # Try custom agent format first (for custom RAG agents)
    try:
        inputs = {
            'input': _convert_messages_to_input(messages),
            'max_output_tokens': max_tokens,
            'stream': False
        }
        
        logger.info(f"Trying custom agent format with inputs: {inputs}")
        res = client.predict(
            endpoint=endpoint_name,
            inputs=inputs
        )
        
        # Handle custom agent response format
        if isinstance(res, dict):
            # Custom agent may return direct content or structured format
            if "output" in res:
                content = res["output"]
                return {"role": "assistant", "content": str(content), "raw_response": res}
            elif "content" in res:
                return {"role": "assistant", "content": str(res["content"]), "raw_response": res}
            elif "text" in res:
                return {"role": "assistant", "content": str(res["text"]), "raw_response": res}
            elif "result" in res:
                return {"role": "assistant", "content": str(res["result"]), "raw_response": res}
            else:
                # Return the whole response as string
                logger.info(f"Custom agent response format: {res}")
                return {"role": "assistant", "content": str(res), "raw_response": res}
        else:
            return {"role": "assistant", "content": str(res), "raw_response": res}
            
    except Exception as custom_error:
        logger.warning(f"Custom agent format failed: {custom_error}, trying standard format")
        
        # Fallback to standard OpenAI format
        try:
            res = client.predict(
                endpoint=endpoint_name,
                inputs={'messages': messages, "max_tokens": max_tokens},
            )
            
            # Handle different response formats
            if "messages" in res:
                # Agent framework response format
                return res["messages"][-1] if res["messages"] else {"role": "assistant", "content": "No response"}
            elif "choices" in res:
                # OpenAI-compatible response format
                choice_message = res["choices"][0]["message"]
                choice_content = choice_message.get("content")
                
                if isinstance(choice_content, list):
                    # Structured content
                    combined_content = "".join([part.get("text", "") for part in choice_content if part.get("type") == "text"])
                    return {
                        "role": choice_message.get("role", "assistant"),
                        "content": combined_content
                    }
                elif isinstance(choice_content, str):
                    return choice_message
                else:
                    return {"role": "assistant", "content": str(choice_content)}
            else:
                # Fallback: try to extract any text content
                logger.warning(f"Unexpected response format: {res}")
                return {"role": "assistant", "content": str(res)}
                
        except Exception as e:
            logger.error(f"Both formats failed. Custom error: {custom_error}, Standard error: {e}")
            # Raise the custom error instead since that's likely the right format
            raise Exception(f"Failed to query endpoint {endpoint_name} with custom format: {str(custom_error)}")

def extract_content_from_stream(chunk: Dict[str, Any]) -> Optional[str]:
    """
    Extract text content from a streaming chunk.
    Handles various response formats from Databricks agent endpoints.
    """
    try:
        # Format 1: Direct delta content (OpenAI-style streaming)
        if "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                return delta["content"]
        
        # Format 2: Agent framework streaming with content
        if "content" in chunk:
            content = chunk["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return "".join([part.get("text", "") for part in content if part.get("type") == "text"])
        
        # Format 3: Direct text field
        if "text" in chunk:
            return chunk["text"]
        
        # Format 4: Custom agent with output field
        if "output" in chunk:
            output = chunk["output"]
            if isinstance(output, str):
                return output
            elif isinstance(output, dict):
                # Try to extract text from nested structure
                if "text" in output:
                    return output["text"]
                elif "content" in output:
                    return output["content"]
        
        # Format 5: Delta with output
        if "delta" in chunk:
            delta = chunk["delta"]
            if isinstance(delta, str):
                return delta
            elif isinstance(delta, dict) and "content" in delta:
                return delta["content"]
            
    except Exception as e:
        logger.warning(f"Could not extract content from chunk: {e}")
    
    return None

def parse_rag_response(response: dict) -> Optional[dict]:
    """
    Parse the RAG endpoint response to extract the structured JSON.
    
    The RAG response can have various formats:
    - Direct dict with 'output' key containing list of messages
    - List of messages directly
    - Wrapped in 'raw_response' key
    
    The final message typically contains the structured JSON in the 'text' field
    of the 'content' array.
    
    Args:
        response: The raw response from the RAG endpoint
        
    Returns:
        Parsed JSON data as a dictionary, or None if parsing fails
    """
    try:
        # First, check if response has 'raw_response' key (from our query_endpoint wrapper)
        if isinstance(response, dict) and "raw_response" in response:
            response = response["raw_response"]
        
        # If response is a string, try to parse it as JSON first
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Response is a string but not valid JSON")
                return None
        
        # Case 1: Response is already the structured data we need
        if isinstance(response, dict):
            # Check if it has the expected structure
            if 'query_understanding' in response and 'results' in response:
                logger.info("Response is already in the expected format")
                return response
            
            # Check if it has an 'output' key with a list (agent/v1/responses format)
            if 'output' in response and isinstance(response['output'], list):
                logger.info("Found 'output' key with list, processing as agent response")
                response = response['output']
                # Continue to list processing below
            else:
                # Check if it's wrapped in other keys
                for key in ['content', 'text', 'result']:
                    if key in response:
                        content = response[key]
                        if isinstance(content, str):
                            try:
                                parsed = json.loads(content)
                                if 'query_understanding' in parsed and 'results' in parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(content, dict) and 'query_understanding' in content:
                            return content
                        elif isinstance(content, list):
                            # If content is a list, process it below
                            response = content
                            break
        
        # Case 2: Response is a list (from RAG agent with multiple messages)
        if isinstance(response, list):
            # Find the last assistant message with content
            for item in reversed(response):
                if isinstance(item, dict):
                    # Check for message type
                    if item.get('type') == 'message' and item.get('role') == 'assistant':
                        content = item.get('content', [])
                        
                        # Content can be a list of content blocks
                        if isinstance(content, list):
                            for content_block in content:
                                if isinstance(content_block, dict):
                                    # Look for text in output_text or text fields
                                    text = content_block.get('text') or content_block.get('output_text')
                                    if text and len(text) > 50:  # Skip short messages
                                        # Try to parse as JSON
                                        try:
                                            text=text.replace('```json', '').replace('```', '')  ########### birbal added
                                            parsed = json.loads(text)
                                            if isinstance(parsed, dict) and 'query_understanding' in parsed:
                                                logger.info("Successfully parsed structured JSON from RAG response")
                                                return parsed
                                        except json.JSONDecodeError:
                                            # If not JSON, check if it contains JSON-like structure
                                            # Look for JSON object in the text
                                            import re
                                            # More sophisticated regex to match nested JSON
                                            json_match = re.search(r'\{[\s\S]*?"query_understanding"[\s\S]*?"results"[\s\S]*?\}(?:\s*\})?$', text, re.DOTALL)
                                            if json_match:
                                                try:
                                                    parsed = json.loads(json_match.group())
                                                    logger.info("Successfully extracted JSON from text using regex")
                                                    return parsed
                                                except json.JSONDecodeError:
                                                    pass
                        
                        # Content might be a string
                        elif isinstance(content, str):
                            try:
                                parsed = json.loads(content)
                                if isinstance(parsed, dict) and 'query_understanding' in parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                pass
        
        logger.warning(f"Could not parse RAG response. Response type: {type(response)}")
        if isinstance(response, (dict, list)):
            logger.info(f"Response structure: {json.dumps(response, indent=2)[:500]}...")
        
        return None
        
    except Exception as e:
        logger.error(f"Error parsing RAG response: {e}")
        return None

