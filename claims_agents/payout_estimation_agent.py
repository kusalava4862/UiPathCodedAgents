"""
Healthcare Claim Payout Estimation Agent - Using MCP Server Tools
Estimates payout amounts for validated claims based on fee schedules, coverage, and policy terms
"""

import json
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List, TypedDict
from contextlib import asynccontextmanager
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from uipath_langchain.chat import UiPathAzureChatOpenAI
from uipath import UiPath
from uipath.tracing import traced
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.tools import tool

# Configure logging for tracing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = UiPathAzureChatOpenAI(
    model="gpt-4o-2024-08-06",
    temperature=0,
    max_tokens=4096,
    timeout=None,
    max_retries=2,
)

# Initialize SDK for authentication
sdk = UiPath()

# ========== Configuration Variables ==========
# All configurable values are defined here in global scope
UIPATH_MCP_SERVER_URL = os.getenv("UIPATH_MCP_SERVER_URL")
UIPATH_ACCESS_TOKEN = os.getenv("UIPATH_ACCESS_TOKEN")
UIPATH_EMAIL_PROCESS_NAME = "Send Email"
UIPATH_EMAIL_FOLDER_PATH = "UiPath_SDK_Challenge/Claims Adjudication Processes"


class GraphState(TypedDict):
    """State for the payout estimation graph with tracing support"""
    claim: str
    ProviderEmailAddress: Optional[str]


class GraphOutput(BaseModel):
    html_email_body: str


# ---------------- MCP Server Configuration ----------------

@asynccontextmanager
async def get_mcp_session():
    """MCP session management"""
    mcp_server_url = UIPATH_MCP_SERVER_URL
    
    if not mcp_server_url:
        raise ValueError("UIPATH_MCP_SERVER_URL environment variable is required")
    
    access_token = UIPATH_ACCESS_TOKEN
    
    # Try to get token from SDK's API client
    if hasattr(sdk, 'api_client'):
        if hasattr(sdk.api_client, 'default_headers'):
            auth_header = sdk.api_client.default_headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                access_token = auth_header.replace('Bearer ', '')
                logger.info("Retrieved token from UiPath API client")
    
    # Fallback to environment variables
    if not access_token:
        access_token = os.getenv("UIPATH_ACCESS_TOKEN") or os.getenv("UIPATH_TOKEN")
        if access_token:
            logger.info("Using access token from environment variable")
    
    async with streamablehttp_client(
        url=mcp_server_url,
        headers={"Authorization": f"Bearer {access_token}"} if access_token else {},
        timeout=60,
    ) as (read, write, session_id_callback):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def get_mcp_tools():
    """Load MCP tools for use with agents - context manager keeps session open"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        logger.info(f"Loaded {len(tools)} tools from MCP server: {[tool.name for tool in tools]}")
        yield tools


# Tool: Send Email
@tool
@traced(name="send_payout_notification_email")
async def send_email(email_data: str) -> str:
    """Send email notification via UiPath process. 
    
    Args:
        email_data: JSON string with keys: process_name, to, subject, html_body
    
    Returns:
        Status message indicating if email was sent successfully
    """
    try:
        data = json.loads(email_data)
        process_name = data.get('process_name', UIPATH_EMAIL_PROCESS_NAME)
        to_email = data.get('to')
        subject = data.get('subject', 'Claim Payout Estimation Notification')
        html_body = data.get('html_body', '')
        
        if not to_email:
            return "ERROR: 'to' email address is required"
        
        logger.info(f"Sending email to {to_email} via process {process_name}")
        
        job = await sdk.processes.invoke_async(
            name=process_name,
            input_arguments={
                'to': to_email,
                'subject': subject,
                'html_body': html_body
            },
            folder_path=UIPATH_EMAIL_FOLDER_PATH
        )
        
        logger.info(f"Email job created: {job.key}")
        return f"EMAIL_SENT: Notification sent to {to_email} via process {process_name}. Job ID: {job.key}"
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in email_data: {str(e)}")
        return f"ERROR: Invalid JSON format - {str(e)}"
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return f"ERROR: Could not send email - {str(e)}"


# System prompt for agent reasoning
SYSTEM_PROMPT = """You are a healthcare claim payout estimation expert. Your task is to process structured JSON inputs for validated claims, policies, coverage, and fee schedules to estimate payout amounts for both the payer (insurance company) and the patient.

AVAILABLE MCP TOOLS:
1. getAllowedAmountForCPTCodes(cpt_codes: str)
   - Input: cpt_codes (comma-separated string, e.g., "99213,36415,87880")
   - Output: Returns allowed_amount for each CPT code from fee schedule
   - Use this to get the allowed amount for CPT codes in the claim

2. getCoverageForCPTCodes(cpt_codes: str)
   - Input: cpt_codes (comma-separated string, e.g., "99213,36415,87880")
   - Output: Returns coverage information for each CPT code including coverage_limit and note
   - Use this to check coverage limits for CPT codes

3. getPolicyDetailsForPolicyID(policy_id: str)
   - Input: policy_id (string, e.g., "POL-US-8088")
   - Output: Returns policy details including deductible, max_out_of_pocket, effective_date, expiry_date, plan_name, status
   - Use this to get policy-level financial parameters

4. getCodesExeclusion(cpt_codes: str, icd_codes: str, patient_id: str)
   - Input: cpt_codes (comma-separated string), icd_codes (comma-separated string), patient_id (string)
   - Output: Returns exclusion details if any CPT/ICD combinations are excluded
   - Use this to check for exclusions that would make services not payable

AVAILABLE LOCAL TOOLS:
- send_email(email_data): Send HTML email via UiPath process (JSON with: process_name, to, subject, html_body)

PAYOUT ESTIMATION PROCESS - FOLLOW THESE STEPS:

1. Parse the claim JSON to extract:
   - claim_id
   - policy_id
   - service_date
   - claim_lines (each with line_id, cpt, icd)

2. Extract all CPT codes and ICD codes from claim_lines and create comma-separated strings:
   - Collect all CPT codes: "99213,36415,87880" (example)
   - Collect all ICD codes: "J02.9,K80.12,Z41.1" (example)
   - Extract patient_id from claim JSON

3. Call MCP tools with comma-separated codes:
   a. Call getAllowedAmountForCPTCodes(cpt_codes) with all CPT codes
      - Returns allowed_amount for each CPT code
      - Validate response is proper JSON format
      - If not, attempt to parse or mark for escalation
   
   b. Call getCoverageForCPTCodes(cpt_codes) with all CPT codes
      - Returns coverage information for each CPT code including coverage_limit and note
      - Validate response is proper JSON format
      - If CPT is not found in coverage → mark line for escalation
   
   c. Call getPolicyDetailsForPolicyID(policy_id) with the policy_id
      - Returns deductible, max_out_of_pocket, effective_date, expiry_date, plan_name, status
      - Validate response is proper JSON format
      - Store these values for calculations
   
   d. Call getCodesExeclusion(cpt_codes, icd_codes, patient_id) to check for exclusions
      - Returns exclusion details if any CPT/ICD combinations are excluded
      - If exclusions are found, mark those lines as "Not Payable"

4. Apply payout logic for each CPT code:
   - If CPT is not found in policy_coverage → mark line for escalation (Not Payable)
   - If allowed_amount > coverage_limit → cap at coverage_limit
   - Patient pays deductible first (if deductible_remaining > 0)
   - Once deductible is met, insurer pays remaining allowed amount up to max_out_of_pocket
   - Calculate: patient_responsibility = min(deductible_remaining, allowed_amount * 0.2)
   - Calculate: payer_liability = allowed_amount - patient_responsibility
   - Include a "reason" field for each line item explaining the calculation

5. Sum all values:
   - Total allowed amounts
   - Total patient_responsibility
   - Total payer_liability

6. Generate a clear, professional HTML email to the provider:
   - Summarize payout results with a line-by-line breakdown
   - Include an overall summary (total allowed, patient responsibility, payer liability)
   - Explain each claim line and provide a final disposition (accepted / rejected / partially approved)
   - Clearly mark any excluded or invalid lines as "Not Payable" with an explanation
   - Use proper HTML structure with tables, headers, and professional styling
   - Conclude with a courteous closing and contact information

7. Send the email using send_email tool with:
   - to: ProviderEmailAddress from input
   - subject: "Claim Payout Estimation - [claim_id]"
   - html_body: The generated HTML email body

CRITICAL CALCULATION RULES:
- Track deductible_remaining across all line items (deductible applies once per claim, not per line)
- Patient responsibility includes deductible and 20% coinsurance
- Payer liability = allowed_amount - patient_responsibility
- If allowed_amount exceeds coverage_limit, use coverage_limit as the cap
- Lines marked for escalation should show $0.00 for all amounts and "Not Payable" status

Use the ReAct format: Thought → Action → Observation → Repeat until you have completed all calculations and sent the email.

Your final output should be the HTML email body as a string. Return it in this JSON format:
{"html_email_body": "<html>...</html>"}
"""


@traced(name="payout_estimation_agent")
async def estimate_payout(state: GraphState) -> GraphOutput:
    """Main payout estimation function using LangGraph ReAct agent with MCP server tools"""
    
    # Initialize tracing
    agent_messages = state.get('agent_messages', [])
    validation_results = state.get('validation_results', {})
    trace_spans = []
    
    try:
        logger.info("Starting payout estimation with ReAct agent")
        trace_spans.append({
            "name": "payout_estimation_start",
            "status": "start",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Get MCP tools (context manager keeps session open)
        async with get_mcp_tools() as mcp_tools:
            logger.info(f"Using {len(mcp_tools)} MCP tools for payout estimation")
            
            # Combine MCP tools with local send_email tool
            tools = list(mcp_tools) + [send_email]
            
            # Create ReAct agent using LangGraph prebuilt
            agent_graph = create_react_agent(llm, tools)
            
            # Prepare input for the agent
            agent_input = f"""You need to estimate payout for the following healthcare claim:

CLAIM DATA:
{state['claim']}

PROVIDER EMAIL: {state.get('ProviderEmailAddress', 'provider@example.com')}

TASK:
1. Parse the claim JSON to extract claim_id, policy_id, service_date, patient_id, and all claim_lines
2. Extract all CPT codes and ICD codes from claim_lines and create comma-separated strings
3. Call MCP tools:
   - getAllowedAmountForCPTCodes(cpt_codes) to get allowed amounts
   - getCoverageForCPTCodes(cpt_codes) to get coverage limits
   - getPolicyDetailsForPolicyID(policy_id) to get deductible and max_out_of_pocket
   - getCodesExeclusion(cpt_codes, icd_codes, patient_id) to check for exclusions
4. Calculate patient_responsibility and payer_liability for each line
5. Sum totals across all lines
6. Generate professional HTML email with line-by-line breakdown
7. Send email to provider using send_email tool
8. Return the HTML email body

Provide your final answer in this JSON format:
{{"html_email_body": "<html>...</html>"}}
"""
            
            # Initialize state for agent with system prompt
            initial_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=agent_input)
            ]
            
            initial_state = {"messages": initial_messages}
            
            # Run the agent with streaming to capture dynamic nodes and edges
            config = {
                "configurable": {
                    "thread_id": f"payout-estimation-{hash(state['claim'])}"
                }
            }
            logger.info("Invoking ReAct agent for payout estimation with streaming")
            
            executed_nodes = []
            executed_edges = []
            
            # Stream events to capture dynamic nodes and create traces
            async for event in agent_graph.astream(initial_state, config, stream_mode="updates"):
                # Track each node execution
                for node_name, node_data in event.items():
                    if node_name not in executed_nodes:
                        executed_nodes.append(node_name)
                        logger.info(f"Executing node: {node_name}")
                        trace_spans.append({
                            "name": f"react_agent_node_{node_name}",
                            "type": "node",
                            "status": "start",
                            "timestamp": asyncio.get_event_loop().time()
                        })
                        
                        # Track edges based on node execution order
                        if len(executed_nodes) > 1:
                            edge = (executed_nodes[-2], node_name)
                            if edge not in executed_edges:
                                executed_edges.append(edge)
                                logger.info(f"Executing edge: {edge[0]} -> {edge[1]}")
                        
                        # Extract message content for tracing
                        if isinstance(node_data, dict) and "messages" in node_data:
                            messages = node_data["messages"]
                            if messages:
                                last_msg = messages[-1]
                                msg_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                                agent_messages.append({
                                    "node": node_name,
                                    "message": msg_content[:200] if len(str(msg_content)) > 200 else str(msg_content),
                                    "timestamp": asyncio.get_event_loop().time()
                                })
                                logger.debug(f"Node {node_name} message: {msg_content[:100]}...")
                        
                        # Mark node as completed
                        trace_spans.append({
                            "name": f"react_agent_node_{node_name}",
                            "type": "node",
                            "status": "end",
                            "timestamp": asyncio.get_event_loop().time()
                        })
            
            # Get final state
            final_state = await agent_graph.ainvoke(initial_state, config)
            
            # Store trace information in state
            validation_results = {
                "executed_nodes": executed_nodes,
                "executed_edges": executed_edges,
                "trace_spans": trace_spans,
                "total_nodes": len(executed_nodes),
                "total_edges": len(executed_edges)
            }
            
            # Update state with trace information
            state['validation_results'] = validation_results
            state['agent_messages'] = agent_messages
            
            logger.info(f"Agent execution completed. Nodes: {executed_nodes}, Edges: {executed_edges}")
            
                        
            # Extract final answer from agent messages
            if final_state and final_state.get("messages"):
                last_message = final_state["messages"][-1]
                output = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                logger.info(f"Agent completed. Extracting HTML email body from: {output[:200]}...")
                
                # Try to parse JSON from output
                import re
                # Look for JSON with html_email_body field
                json_match = re.search(r'\{[^{}]*"html_email_body"[^{}]*\}', output, re.DOTALL)
                
                if not json_match:
                    # Try to find HTML directly
                    html_match = re.search(r'<html>.*</html>', output, re.DOTALL | re.IGNORECASE)
                    if html_match:
                        html_body = html_match.group()
                        result = GraphOutput(html_email_body=html_body)
                        logger.info("Extracted HTML email body from output")
                        
                        # Complete trace
                        trace_spans.append({
                            "name": "payout_estimation_complete",
                            "status": "end",
                            "timestamp": asyncio.get_event_loop().time(),
                            "result": {"html_extracted": True}
                        })
                        
                        return result
                
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        html_body = parsed.get("html_email_body", "")
                        if not html_body:
                            # Try to extract HTML from the full output
                            html_match = re.search(r'<html>.*</html>', output, re.DOTALL | re.IGNORECASE)
                            if html_match:
                                html_body = html_match.group()
                            else:
                                html_body = output
                        
                        result = GraphOutput(html_email_body=html_body)
                        logger.info("Extracted HTML email body from JSON")
                        
                        # Complete trace
                        trace_spans.append({
                            "name": "payout_estimation_complete",
                            "status": "end",
                            "timestamp": asyncio.get_event_loop().time(),
                            "result": {"html_extracted": True}
                        })
                        
                        return result
                    except json.JSONDecodeError:
                        logger.warning("Could not parse JSON, trying to extract HTML directly")
                
                # Fallback: try to extract HTML from output
                html_match = re.search(r'<html>.*</html>', output, re.DOTALL | re.IGNORECASE)
                if html_match:
                    html_body = html_match.group()
                    result = GraphOutput(html_email_body=html_body)
                    
                    # Complete trace
                    trace_spans.append({
                        "name": "estimate_payout",
                        "status": "end",
                        "timestamp": asyncio.get_event_loop().time(),
                        "result": {"html_extracted": True}
                    })
                    
                    return result
                
                # Last resort: use full output as HTML
                result = GraphOutput(html_email_body=output)
                
                # Complete trace
                trace_spans.append({
                    "name": "payout_estimation_complete",
                    "status": "end",
                    "timestamp": asyncio.get_event_loop().time(),
                    "result": {"html_extracted": False, "used_full_output": True}
                })
                
                return result
            else:
                logger.error("Agent did not return any messages")
                
                # Complete trace with error
                trace_spans.append({
                    "name": "payout_estimation_no_messages_error",
                    "status": "error",
                    "timestamp": asyncio.get_event_loop().time(),
                    "error": "Agent did not return any messages"
                })
                
                return GraphOutput(
                    html_email_body="<html><body><p>Error: Agent did not return payout estimation result</p></body></html>"
                )
        
    except Exception as e:
        logger.error(f"Error during payout estimation: {str(e)}", exc_info=True)
        
        # Complete trace with exception
        trace_spans.append({
            "name": "payout_estimation_exception",
            "status": "error",
            "timestamp": asyncio.get_event_loop().time(),
            "error": str(e),
            "exception_type": type(e).__name__
        })
        
        return GraphOutput(
            html_email_body=f"<html><body><p>Error during payout estimation: {str(e)}</p></body></html>"
        )


# Build the graph with dynamic structure
# The ReAct agent handles its own internal flow with dynamic nodes/edges
builder = StateGraph(GraphState, output_schema=GraphOutput)
builder.add_node("estimate_payout", estimate_payout)
builder.add_edge(START, "estimate_payout")
builder.add_edge("estimate_payout", END)

# Compile graph with tracing support
graph = builder.compile()


# Function to get dynamic graph structure from execution
def get_dynamic_graph_structure(state: GraphState) -> Dict[str, Any]:
    """Extract dynamic graph structure from state"""
    validation_results = state.get('validation_results', {})
    return {
        "nodes": validation_results.get('executed_nodes', []),
        "edges": validation_results.get('executed_edges', []),
        "trace_spans": validation_results.get('trace_spans', []),
        "total_nodes": validation_results.get('total_nodes', 0),
        "total_edges": validation_results.get('total_edges', 0)
    }

