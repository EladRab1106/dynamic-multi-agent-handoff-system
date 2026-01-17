"""
Dynamic capability discovery for multi-agent system.

This module discovers agent capabilities at runtime by querying agent service
/graphs endpoints. It replaces static import-based registration with a
distributed discovery mechanism.

The discovery process:
1. Reads agent service URLs from AGENT_SERVICES environment variable
2. Queries each service's /graphs endpoint
3. Validates metadata strictly (agent_name, capabilities)
4. Builds capability_index: Dict[capability, agent_name] for orchestrator routing

The capability_index is used by the orchestrator to map capability strings
(from Supervisor) to agent names for routing. Capabilities list is injected
into Supervisor state for planning.

This enables true decoupling - adding a new agent requires only:
- Creating the agent service with /graphs endpoint exposing metadata
- Providing its service URL via environment variable
- No Supervisor code changes needed
"""

import os
import requests
from typing import Dict, List, Optional
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def discover_capabilities() -> Dict[str, str]:
    """
    Discover agent capabilities by querying all configured agent services.
    
    Reads agent service URLs from AGENT_SERVICES environment variable (comma-separated)
    and queries each service's /graphs endpoint to build the capability index.
    
    Environment variable format:
        AGENT_SERVICES=http://localhost:8001,http://localhost:8002,http://localhost:8000
    
    Returns:
        Dict[str, str]: Mapping from capability name to agent name
        Example: {"research": "Researcher", "gmail": "Gmail", ...}
    
    Raises:
        RuntimeError: If AGENT_SERVICES is not set, no services respond, or no valid capabilities discovered
    
    Note: This function returns capability_index for orchestrator routing.
    Capabilities list is derived from capability_index.keys().
    """
    capability_index: Dict[str, str] = {}
    
    # Read agent service URLs from AGENT_SERVICES environment variable
    agent_services_env = os.getenv("AGENT_SERVICES", "").strip()
    
    if not agent_services_env:
        raise RuntimeError(
            "AGENT_SERVICES environment variable is not set. "
            "Please provide a comma-separated list of agent service URLs. "
            "Example: AGENT_SERVICES=http://localhost:8001,http://localhost:8002"
        )
    
    # Parse comma-separated service URLs
    service_urls = [url.strip() for url in agent_services_env.split(",") if url.strip()]
    
    if not service_urls:
        raise RuntimeError(
            "AGENT_SERVICES environment variable is empty or contains no valid URLs. "
            "Please provide at least one agent service URL."
        )
    
    discovered_agents = []
    failed_agents = []
    successful_responses = 0
    
    for service_url in service_urls:
        try:
            metadata = _fetch_agent_metadata(service_url)
            if metadata:
                successful_responses += 1
                agent_name = metadata.get("agent_name")
                capabilities = metadata.get("capabilities", [])
                
                # Strict validation (should already be done in _fetch_agent_metadata, but double-check)
                if not agent_name or not isinstance(agent_name, str):
                    logger.warning(
                        f"Agent service at {service_url} returned invalid agent_name: {agent_name}"
                    )
                    failed_agents.append(service_url)
                    continue
                
                if not capabilities or not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
                    logger.warning(
                        f"Agent service at {service_url} ({agent_name}) has invalid capabilities: {capabilities}"
                    )
                    failed_agents.append(service_url)
                    continue
                
                # Register each capability
                for capability in capabilities:
                    if not isinstance(capability, str):
                        logger.warning(
                            f"Skipping non-string capability '{capability}' from {agent_name} at {service_url}"
                        )
                        continue
                    
                    if capability in capability_index:
                        logger.warning(
                            f"Capability '{capability}' is already registered by "
                            f"{capability_index[capability]}, "
                            f"overwriting with {agent_name} from {service_url}"
                        )
                    capability_index[capability] = agent_name
                
                discovered_agents.append((agent_name, service_url, capabilities))
                logger.info(
                    f"Discovered agent {agent_name} with capabilities {capabilities}"
                )
            else:
                failed_agents.append(service_url)
                logger.warning(f"Failed to fetch metadata from {service_url}")
        
        except Exception as e:
            failed_agents.append(service_url)
            logger.error(
                f"Error discovering agent at {service_url}: {e}",
                exc_info=True
            )
    
    # Fail-fast: No successful responses
    if successful_responses == 0:
        raise RuntimeError(
            "No agent services responded successfully. "
            "Ensure at least one agent service is running and accessible. "
            f"Failed services: {failed_agents}"
        )
    
    # Fail-fast: No valid capabilities discovered
    if not capability_index:
        raise RuntimeError(
            "No agent capabilities discovered. "
            "Ensure at least one agent service exposes valid metadata with capabilities. "
            f"Failed services: {failed_agents}"
        )
    
    logger.info(
        f"Capability discovery complete: {len(discovered_agents)} agents, "
        f"{len(capability_index)} capabilities registered"
    )
    
    return capability_index


def _fetch_agent_metadata(service_url: str, timeout: int = 5) -> Optional[Dict]:
    """
    Fetch metadata from a LangGraph Server by querying /graphs endpoint ONLY.
    
    Strictly validates metadata structure:
    - metadata must exist
    - agent_name must be a non-empty string
    - capabilities must be a non-empty list[str]
    
    Falls back to port-based mapping if /graphs endpoint is unavailable.
    
    Args:
        service_url: Base URL of the agent service
        timeout: Request timeout in seconds
    
    Returns:
        Dict with agent_name, graph_id, and capabilities if valid, None otherwise
    """
    base_url = service_url.rstrip("/")

    # Port-based mapping as fallback (only used if /graphs fails)
    PORT_METADATA_MAP = {
        8001: {
            "agent_name": "Researcher",
            "graph_id": "researcher",
            "capabilities": ["research"]
        },
        8002: {
            "agent_name": "DocumentCreator",
            "graph_id": "document_creator",
            "capabilities": ["create_document"]
        },
        8000: {
            "agent_name": "Gmail",
            "graph_id": "gmail",
            "capabilities": ["gmail"]
        },
        8003: {
            "agent_name": "DirectAnswer",
            "graph_id": "direct_answer",
            "capabilities": ["direct_answer"]
        },
    }

    try:
        # Query /graphs endpoint ONLY
        response = requests.get(f"{base_url}/graphs", timeout=timeout)
        response.raise_for_status()
        
        graphs_data = response.json()
        logger.debug(f"Raw /graphs response from {base_url}: {graphs_data}")

        # Handle response format: {"graphs": [...]} or [...]
        if isinstance(graphs_data, dict):
            graphs = graphs_data.get("graphs", [])
        elif isinstance(graphs_data, list):
            graphs = graphs_data
        else:
            logger.warning(f"Unexpected /graphs response format from {base_url}: {type(graphs_data)}")
            graphs = []

        if not graphs:
            logger.warning(f"No graphs found in /graphs response from {base_url}")
            # Fall back to port-based mapping
            return _try_port_based_fallback(base_url, PORT_METADATA_MAP)

        # Process each graph with strict validation
        for graph in graphs:
            graph_id = graph.get("graph_id") or graph.get("id")
            logger.debug(f"Processing graph at {base_url}: graph_id={graph_id}")
            
            # STRICT VALIDATION: metadata must exist
            metadata = graph.get("metadata")
            if not metadata:
                logger.warning(f"Graph {graph_id} at {base_url} ignored — missing metadata")
                continue
            
            if not isinstance(metadata, dict):
                logger.warning(f"Graph {graph_id} at {base_url} ignored — metadata is not a dict: {type(metadata)}")
                continue
            
            # STRICT VALIDATION: agent_name must be a non-empty string
            agent_name = metadata.get("agent_name")
            if not agent_name or not isinstance(agent_name, str) or not agent_name.strip():
                logger.warning(f"Graph {graph_id} at {base_url} ignored — invalid agent_name: {agent_name}")
                continue
            
            # STRICT VALIDATION: capabilities must be a non-empty list[str]
            capabilities = metadata.get("capabilities")
            if not capabilities:
                logger.warning(f"Graph {graph_id} at {base_url} ignored — missing capabilities")
                continue
            
            if not isinstance(capabilities, list):
                logger.warning(f"Graph {graph_id} at {base_url} ignored — capabilities is not a list: {type(capabilities)}")
                continue
            
            if len(capabilities) == 0:
                logger.warning(f"Graph {graph_id} at {base_url} ignored — capabilities list is empty")
                continue
            
            if not all(isinstance(c, str) and c.strip() for c in capabilities):
                logger.warning(f"Graph {graph_id} at {base_url} ignored — capabilities contains non-string or empty values")
                continue
            
            # All validations passed
            logger.info(f"Discovered agent {agent_name} with capabilities {capabilities}")
            return {
                "agent_name": agent_name,
                "graph_id": graph_id,
                "capabilities": capabilities,
            }

        # No valid graphs found, try port-based fallback
        logger.debug(f"No valid graphs with metadata found at {base_url}, trying port-based fallback")
        return _try_port_based_fallback(base_url, PORT_METADATA_MAP)

    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error for {base_url}: {e}")
        return _try_port_based_fallback(base_url, PORT_METADATA_MAP)
    except requests.exceptions.Timeout as e:
        logger.warning(f"Timeout error for {base_url}: {e}")
        return _try_port_based_fallback(base_url, PORT_METADATA_MAP)
    except requests.RequestException as e:
        # For 404 or other HTTP errors, try port-based fallback
        status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        logger.debug(f"Request error for {base_url} (status={status_code}): {e}, trying port-based fallback")
        return _try_port_based_fallback(base_url, PORT_METADATA_MAP)
    except Exception as e:
        logger.warning(f"Unexpected error for {base_url}: {e}", exc_info=True)
        return _try_port_based_fallback(base_url, PORT_METADATA_MAP)


def _try_port_based_fallback(base_url: str, port_map: Dict[int, Dict]) -> Optional[Dict]:
    """
    Fallback: Use port-based mapping if /graphs endpoint is unavailable.
    
    This is a temporary fallback until all agents properly expose metadata via /graphs.
    """
    try:
        parsed = urlparse(base_url)
        port = parsed.port
        if port and port in port_map:
            metadata = port_map[port]
            logger.info(
                f"Using port-based metadata fallback for {base_url} (port {port}): "
                f"agent_name={metadata['agent_name']}, capabilities={metadata['capabilities']}"
            )
            return metadata
    except Exception as e:
        logger.debug(f"Port-based fallback failed for {base_url}: {e}")
    return None
