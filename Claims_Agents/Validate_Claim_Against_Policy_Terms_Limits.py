"""
Agent 2: Policy Validation Agent
Validates claims against policy terms, coverage, and exclusions using MCP Server
Uses LangChain and LangGraph for multi-step policy validation workflow
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import aiohttp

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import ASSETS, CLAIM_CONFIG, LANGCHAIN_CONFIG, MCP_CONFIG
from uipath_utils import get_policy_context, sdk_manager
from tracing_utils import traced, async_traced, NodeTracer
from agent_1_cpt_icd_validation import Claim, ValidationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class PolicyValidationResult(BaseModel):
    """Result of policy validation"""
    claim_id: str
    policy_active: bool
    policy_number: str = ""
    coverage_effective: bool = False
    excluded_codes_found: bool = False
    excluded_codes: List[str] = Field(default_factory=list)
    policy_limits_met: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    validation_timestamp: str = ""
    recommendations: str = ""


class PolicyInfo(BaseModel):
    """Policy information retrieved from context grounding"""
    policy_id: str
    policy_holder_id: str
    coverage_type: str
    effective_date: str
    termination_date: Optional[str] = None
    exclusions: List[str] = Field(default_factory=list)
    limitations: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# LangGraph State
# ============================================================================

class PolicyValidationState(BaseModel):
    """State for the policy validation workflow"""
    claim: Claim
    validation_result: Optional[PolicyValidationResult] = None
    policy_info: Optional[PolicyInfo] = None
    mcp_responses: Dict[str, Any] = Field(default_factory=dict)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    current_step: str = "initialized"


# ============================================================================
# MCP Server Integration
# ============================================================================

@async_traced()
async def query_mcp_server(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = MCP_CONFIG.TIMEOUT
) -> Dict[str, Any]:
    """
    Query MCP server for policy validation
    
    Args:
        endpoint: API endpoint (e.g., "/api/v1/policy/validate")
        payload: Request payload
        timeout: Request timeout in seconds
        
    Returns:
        Response from MCP server
    """
    url = f"{MCP_CONFIG.BASE_URL}{endpoint}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"MCP server returned status 200 for {endpoint}")
                    return {"success": True, "data": data}
                else:
                    error_text = await response.text()
                    logger.error(f"MCP server error {response.status}: {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}", "details": error_text}
    
    except asyncio.TimeoutError:
        logger.error(f"MCP server timeout for {endpoint}")
        return {"success": False, "error": "Request timeout"}
    except Exception as e:
        logger.error(f"Error querying MCP server: {str(e)}")
        return {"success": False, "error": str(e)}


# ============================================================================
# Node Functions with Tracing
# ============================================================================

@async_traced()
async def retrieve_policy_node(state: PolicyValidationState) -> PolicyValidationState:
    """
    Node: Retrieve policy information from context grounding
    """
    node_tracer = NodeTracer("retrieve_policy")
    state.current_step = "retrieve_policy"
    
    try:
        logger.info(f"Retrieving policy for claim {state.claim.claim_id}")
        node_tracer.log_data("claim_id", state.claim.claim_id)
        node_tracer.log_data("provider_npi", state.claim.provider_npi)
        
        # Search for policy information
        query = f"Policy information for provider NPI: {state.claim.provider_npi}, patient: {state.claim.patient_id}"
        context_result = await get_policy_context(query)
        
        if context_result.success and context_result.results:
            state.context_data["policy_context"] = context_result.results
            node_tracer.log_data("policy_matches", context_result.total_matches)
            logger.info(f"Found {context_result.total_matches} policy matches")
        else:
            state.errors.append(f"Policy retrieval failed: {context_result.error_message}")
            logger.warning(f"No policy context found for provider {state.claim.provider_npi}")
        
        logger.info(f"Policy retrieval completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error retrieving policy: {str(e)}")
        state.errors.append(f"Policy retrieval error: {str(e)}")
    
    return state


@async_traced()
async def validate_policy_active_node(state: PolicyValidationState) -> PolicyValidationState:
    """
    Node: Validate policy is active using MCP server
    """
    node_tracer = NodeTracer("validate_policy_active")
    state.current_step = "validate_policy_active"
    
    try:
        logger.info(f"Validating policy active status for claim {state.claim.claim_id}")
        
        # Prepare MCP server request
        payload = {
            "provider_npi": state.claim.provider_npi,
            "patient_id": state.claim.patient_id,
            "claim_date": datetime.now().isoformat()
        }
        
        # Query MCP server
        mcp_response = await query_mcp_server(
            endpoint=MCP_CONFIG.POLICY_ENDPOINT,
            payload=payload
        )
        
        state.mcp_responses["policy_validation"] = mcp_response
        node_tracer.log_data("mcp_response", mcp_response)
        
        if mcp_response.get("success"):
            policy_data = mcp_response.get("data", {})
            
            # Extract policy info
            if "policy_active" in policy_data:
                state.validation_result = PolicyValidationResult(
                    claim_id=state.claim.claim_id,
                    policy_active=policy_data.get("policy_active", False),
                    policy_number=policy_data.get("policy_number", ""),
                    coverage_effective=policy_data.get("coverage_effective", False)
                )
                node_tracer.log_data("policy_active", state.validation_result.policy_active)
                logger.info(f"Policy active: {state.validation_result.policy_active}")
        else:
            state.errors.append(f"Policy validation MCP failed: {mcp_response.get('error')}")
        
        logger.info(f"Policy active validation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error validating policy active: {str(e)}")
        state.errors.append(f"Policy active validation error: {str(e)}")
    
    return state


@async_traced()
async def check_coverage_node(state: PolicyValidationState) -> PolicyValidationState:
    """
    Node: Check coverage for claim codes using MCP server
    """
    node_tracer = NodeTracer("check_coverage")
    state.current_step = "check_coverage"
    
    try:
        logger.info(f"Checking coverage for claim {state.claim.claim_id}")
        
        # Collect CPT codes
        cpt_codes = [item.cpt_code for item in state.claim.line_items]
        
        # Prepare MCP request
        payload = {
            "provider_npi": state.claim.provider_npi,
            "cpt_codes": cpt_codes,
            "policy_number": state.validation_result.policy_number if state.validation_result else ""
        }
        
        # Query MCP server
        mcp_response = await query_mcp_server(
            endpoint=MCP_CONFIG.COVERAGE_ENDPOINT,
            payload=payload
        )
        
        state.mcp_responses["coverage_check"] = mcp_response
        node_tracer.log_data("coverage_response", mcp_response)
        
        if mcp_response.get("success"):
            coverage_data = mcp_response.get("data", {})
            
            if state.validation_result:
                state.validation_result.coverage_effective = coverage_data.get("all_covered", False)
                node_tracer.log_data("coverage_effective", state.validation_result.coverage_effective)
                logger.info(f"Coverage effective: {state.validation_result.coverage_effective}")
        else:
            state.errors.append(f"Coverage check failed: {mcp_response.get('error')}")
        
        logger.info(f"Coverage check completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error checking coverage: {str(e)}")
        state.errors.append(f"Coverage check error: {str(e)}")
    
    return state


@async_traced()
async def check_exclusions_node(state: PolicyValidationState) -> PolicyValidationState:
    """
    Node: Check for excluded codes using MCP server
    """
    node_tracer = NodeTracer("check_exclusions")
    state.current_step = "check_exclusions"
    
    try:
        logger.info(f"Checking exclusions for claim {state.claim.claim_id}")
        
        # Collect all codes
        cpt_codes = [item.cpt_code for item in state.claim.line_items]
        all_icd_codes = []
        for item in state.claim.line_items:
            all_icd_codes.extend(item.icd_codes)
        
        # Prepare MCP request
        payload = {
            "provider_npi": state.claim.provider_npi,
            "cpt_codes": cpt_codes,
            "icd_codes": all_icd_codes,
            "policy_number": state.validation_result.policy_number if state.validation_result else ""
        }
        
        # Query MCP server
        mcp_response = await query_mcp_server(
            endpoint=MCP_CONFIG.EXCLUSION_ENDPOINT,
            payload=payload
        )
        
        state.mcp_responses["exclusions_check"] = mcp_response
        node_tracer.log_data("exclusions_response", mcp_response)
        
        if mcp_response.get("success"):
            exclusion_data = mcp_response.get("data", {})
            
            if state.validation_result:
                excluded_codes = exclusion_data.get("excluded_codes", [])
                state.validation_result.excluded_codes_found = len(excluded_codes) > 0
                state.validation_result.excluded_codes = excluded_codes
                
                node_tracer.log_data("excluded_codes_count", len(excluded_codes))
                logger.info(f"Found {len(excluded_codes)} excluded codes")
        else:
            state.errors.append(f"Exclusion check failed: {mcp_response.get('error')}")
        
        logger.info(f"Exclusion check completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error checking exclusions: {str(e)}")
        state.errors.append(f"Exclusion check error: {str(e)}")
    
    return state


@async_traced()
async def validate_policy_limits_node(state: PolicyValidationState) -> PolicyValidationState:
    """
    Node: Validate claim meets policy limits using LLM analysis
    """
    node_tracer = NodeTracer("validate_policy_limits")
    state.current_step = "validate_policy_limits"
    
    try:
        logger.info(f"Validating policy limits for claim {state.claim.claim_id}")
        
        # Initialize LLM
        llm = ChatAnthropic(
            model=LANGCHAIN_CONFIG.MODEL_NAME,
            temperature=LANGCHAIN_CONFIG.TEMPERATURE,
            max_tokens=LANGCHAIN_CONFIG.MAX_TOKENS
        )
        
        # Build validation prompt
        system_prompt = """You are a healthcare policy analyst. Analyze the claim against policy limits.
        
Policy limits to check:
1. Total claim amount within maximum
2. Line item amounts within per-service limits
3. Frequency limits (e.g., max services per year)
4. Age/condition specific limits
5. Provider network status

Respond with JSON containing:
- limits_met: boolean indicating if all limits are satisfied
- violations: list of any limit violations with details
- recommendations: array of recommendations"""
        
        # Format claim and MCP data
        claim_text = f"""
Claim Total Amount: ${state.claim.total_amount}
Claim ID: {state.claim.claim_id}
Provider NPI: {state.claim.provider_npi}

Line Items:
"""
        for item in state.claim.line_items:
            claim_text += f"\nLine {item.line_number}: CPT={item.cpt_code}, Amount=${item.amount}, Units={item.units}"
        
        # Add MCP response data
        if "coverage_check" in state.mcp_responses:
            claim_text += f"\n\nMCP Coverage Data: {json.dumps(state.mcp_responses['coverage_check'], default=str)}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=claim_text)
        ]
        
        response = await llm.ainvoke(messages)
        
        try:
            limit_data = json.loads(response.content)
            node_tracer.log_data("limit_validation", limit_data)
            
            if state.validation_result:
                state.validation_result.policy_limits_met = limit_data.get("limits_met", True)
                
            logger.info(f"Policy limits met: {state.validation_result.policy_limits_met if state.validation_result else 'Unknown'}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON for policy limits")
        
        logger.info(f"Policy limits validation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error validating policy limits: {str(e)}")
        state.errors.append(f"Policy limits validation error: {str(e)}")
    
    return state


@async_traced()
async def generate_policy_result_node(state: PolicyValidationState) -> PolicyValidationState:
    """
    Node: Generate final policy validation result
    """
    node_tracer = NodeTracer("generate_policy_result")
    state.current_step = "generate_policy_result"
    
    try:
        logger.info(f"Generating final policy validation result for claim {state.claim.claim_id}")
        
        if not state.validation_result:
            state.validation_result = PolicyValidationResult(
                claim_id=state.claim.claim_id,
                policy_active=False
            )
        
        # Add validation errors and recommendations
        state.validation_result.validation_errors = state.errors
        
        recommendations = []
        if not state.validation_result.policy_active:
            recommendations.append("Policy is not active. Contact policy holder.")
        if state.validation_result.excluded_codes_found:
            recommendations.append(f"Excluded codes found: {', '.join(state.validation_result.excluded_codes)}. Remove or appeal.")
        if not state.validation_result.coverage_effective:
            recommendations.append("Some codes not covered. Review coverage details.")
        if not state.validation_result.policy_limits_met:
            recommendations.append("Claim exceeds policy limits. Request pre-authorization or adjust claim.")
        
        state.validation_result.recommendations = " ".join(recommendations)
        state.validation_result.validation_timestamp = datetime.now().isoformat()
        
        node_tracer.log_data("policy_active", state.validation_result.policy_active)
        node_tracer.log_data("excluded_codes_found", state.validation_result.excluded_codes_found)
        node_tracer.log_data("coverage_effective", state.validation_result.coverage_effective)
        node_tracer.log_data("policy_limits_met", state.validation_result.policy_limits_met)
        
        logger.info(f"Policy result generation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error generating policy result: {str(e)}")
        state.errors.append(f"Result generation error: {str(e)}")
    
    return state


# ============================================================================
# Graph Construction
# ============================================================================

def build_policy_validation_graph() -> StateGraph:
    """Build the policy validation workflow graph"""
    
    workflow = StateGraph(PolicyValidationState)
    
    # Add nodes
    workflow.add_node("retrieve_policy", retrieve_policy_node)
    workflow.add_node("validate_active", validate_policy_active_node)
    workflow.add_node("check_coverage", check_coverage_node)
    workflow.add_node("check_exclusions", check_exclusions_node)
    workflow.add_node("validate_limits", validate_policy_limits_node)
    workflow.add_node("generate_result", generate_policy_result_node)
    
    # Add edges
    workflow.set_entry_point("retrieve_policy")
    workflow.add_edge("retrieve_policy", "validate_active")
    workflow.add_edge("validate_active", "check_coverage")
    workflow.add_edge("check_coverage", "check_exclusions")
    workflow.add_edge("check_exclusions", "validate_limits")
    workflow.add_edge("validate_limits", "generate_result")
    workflow.add_edge("generate_result", END)
    
    return workflow


# ============================================================================
# Main Agent Interface
# ============================================================================

class PolicyValidationAgent:
    """Agent for claim policy validation"""
    
    def __init__(self):
        self.graph = build_policy_validation_graph().compile()
    
    @traced()
    async def validate_claim_policy(self, claim: Claim) -> PolicyValidationResult:
        """
        Validate claim against policy terms and coverage
        
        Args:
            claim: Claim object to validate
            
        Returns:
            PolicyValidationResult with validation status
        """
        logger.info(f"Starting policy validation for claim {claim.claim_id}")
        
        # Create initial state
        initial_state = PolicyValidationState(claim=claim)
        
        # Execute workflow
        final_state = await asyncio.to_thread(
            self.graph.invoke,
            initial_state
        )
        
        logger.info(f"Policy validation complete for claim {claim.claim_id}")
        
        return final_state.validation_result