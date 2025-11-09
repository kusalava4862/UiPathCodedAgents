"""
Proper ReAct Agent Implementation - Using Agent Reasoning for Medical Claim Validation
"""

import json
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List, TypedDict
from langchain_core.tools import tool
from uipath_langchain.chat import UiPathAzureChatOpenAI
from uipath import UiPath
from pydantic import BaseModel
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from uipath.tracing import traced

# Configure logging for tracing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize
llm = UiPathAzureChatOpenAI(
    model="gpt-4o-2024-08-06",
    temperature=0,
    max_tokens=2048,
    timeout=None,
    max_retries=2,
)

sdk = UiPath()

# ========== Configuration Variables ==========
# All configurable values are defined here in global scope

UIPATH_ICD_CODES_INDEX_NAME = "ICD_Codes"
UIPATH_CPT_CODES_INDEX_NAME = "CPT_Codes"
UIPATH_ICD_CODES_INDEX_FOLDER_NAME = "UiPath_SDK_Challenge"
UIPATH_CPT_CODES_INDEX_FOLDER_NAME = "UiPath_SDK_Challenge"

class GraphState(TypedDict):
    """State for the validation graph with tracing support"""
    claim_data: str
    discharge_summary: str
    provider_email: Optional[str]


class GraphOutput(BaseModel):
    claim_valid: bool
    justification: str


# Tool 1: CPT Code Lookup
@tool
@traced(name="cpt_code_database_lookup")
async def cpt_lookup(code: str) -> str:
    """Lookup CPT code and description from index. Returns code and full description for agent reasoning.
    
    Args:
        code: CPT code to lookup (e.g., '99213')
    
    Returns:
        String containing CPT code and full description from the index
    """
    try:
        logger.info(f"Looking up CPT code: {code}")
        results = await sdk.context_grounding.search_async(
            name=UIPATH_CPT_CODES_INDEX_NAME,
            query=code,
            number_of_results=3,
            folder_path=UIPATH_CPT_CODES_INDEX_FOLDER_NAME
        )
        
        if results and len(results) > 0:
            # Return the most relevant result with full description
            best_match = results[0]
            # Handle different response structures
            if hasattr(best_match, 'content'):
                content = best_match.content
            elif hasattr(best_match, 'text'):
                content = best_match.text
            elif hasattr(best_match, 'description'):
                content = best_match.description
            else:
                content = str(best_match)
            logger.info(f"Found CPT code {code} in index")
            return f"CPT Code: {code}\nDescription: {content}"
        else:
            logger.warning(f"CPT code {code} not found in index")
            return f"CPT Code: {code}\nDescription: NOT FOUND in database"
            
    except Exception as e:
        logger.error(f"Error looking up CPT code {code}: {str(e)}")
        return f"ERROR: Could not lookup CPT {code} - {str(e)}"


# Tool 2: ICD Code Lookup
@tool
@traced(name="icd_code_database_lookup")
async def icd_lookup(code: str) -> str:
    """Lookup ICD code and description from index. Returns code and full description for agent reasoning.
    
    Args:
        code: ICD code to lookup (e.g., 'J020')
    
    Returns:
        String containing ICD code and full description from the index
    """
    try:
        logger.info(f"Looking up ICD code: {code}")
        results = await sdk.context_grounding.search_async(
            name=UIPATH_ICD_CODES_INDEX_NAME,
            query=code,
            number_of_results=3,
            folder_path=UIPATH_ICD_CODES_INDEX_FOLDER_NAME
        )
        
        if results and len(results) > 0:
            # Return the most relevant result with full description
            best_match = results[0]
            # Handle different response structures
            if hasattr(best_match, 'content'):
                content = best_match.content
            elif hasattr(best_match, 'text'):
                content = best_match.text
            elif hasattr(best_match, 'description'):
                content = best_match.description
            else:
                content = str(best_match)
            logger.info(f"Found ICD code {code} in index")
            return f"ICD Code: {code}\nDescription: {content}"
        else:
            logger.warning(f"ICD code {code} not found in index")
            return f"ICD Code: {code}\nDescription: NOT FOUND in database"
            
    except Exception as e:
        logger.error(f"Error looking up ICD code {code}: {str(e)}")
        return f"ERROR: Could not lookup ICD {code} - {str(e)}"


# Tool 3: Send Email
@tool
@traced(name="send_validation_notification_email")
async def send_email(email_data: str) -> str:
    """Send email notification via UiPath process. 
    
    Args:
        email_data: JSON string with keys: process_name, to, subject, html_body
    
    Returns:
        Status message indicating if email was sent successfully
    """
    try:
        data = json.loads(email_data)
        process_name = data.get('process_name', 'SendEmail')
        to_email = data.get('to')
        subject = data.get('subject', 'Claim Validation Notification')
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
            folder_path=UIPATH_FOLDER_ASSET_NAME
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
SYSTEM_PROMPT = """You are a medical claim validation expert. Your task is to validate medical claims by reasoning about CPT and ICD codes.

AVAILABLE TOOLS:
- cpt_lookup(code): Lookup CPT code and description from database
- icd_lookup(code): Lookup ICD code and description from database  
- send_email(email_data): Send HTML email via UiPath process (JSON with: process_name, to, subject, html_body)

VALIDATION PROCESS - USE YOUR REASONING:
1. Extract all CPT and ICD codes from the claim data
2. For each code, use the lookup tools to retrieve the official description
3. COMPARE the retrieved description with the claim description - reason if they match
4. REASON about CPT-ICD medical alignment - are the procedures appropriate for the diagnoses?
5. VERIFY codes match the discharge summary - reason if the codes are supported by the clinical documentation
6. If claim is invalid, prepare a detailed HTML email with:
   - Claim ID and provider information
   - List of invalid codes with explanations
   - Specific reasons for each validation failure
   - Professional HTML formatting

CRITICAL REASONING GUIDELINES:
- A claim is valid ONLY if:
  * ALL codes exist in the database
  * Code descriptions match claim descriptions
  * CPT-ICD pairs are medically appropriate
  * Codes are supported by the discharge summary
- Use your medical knowledge to reason about alignment - don't rely on hardcoded rules
- When preparing HTML email, use proper HTML structure with tables, headers, and styling
- Always provide detailed justification for your decisions

Use the ReAct format: Thought → Action → Observation → Repeat until you have a final answer.
"""

@traced(name="cpt_icd_validation_agent")
async def validate_cpt_icd_codes(state: GraphState) -> GraphOutput:
    """Main validation function using LangGraph ReAct agent with dynamic reasoning and tracing"""
    
    # Initialize tracing
    agent_messages = state.get('agent_messages', [])
    validation_results = state.get('validation_results', {})
    trace_spans = []
    
    try:
        logger.info("Starting claim validation with ReAct agent")
        trace_spans.append({
            "name": "cpt_icd_validation_start",
            "status": "start",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Create tools list
        tools = [cpt_lookup, icd_lookup, send_email]
        
        # Create ReAct agent using LangGraph prebuilt
        from langchain_core.messages import SystemMessage
        
        agent_graph = create_react_agent(
            llm,
            tools
        )
        
        # Prepare input for the agent
        agent_input = f"""You need to validate the following medical claim:

CLAIM DATA:
{state['claim_data']}

DISCHARGE SUMMARY:
{state['discharge_summary']}

PROVIDER EMAIL: {state.get('provider_email', 'provider@example.com')}

TASK:
1. Extract all CPT and ICD codes from the claim data
2. Lookup each code to get official descriptions
3. Reason about code validity by comparing descriptions
4. Reason about CPT-ICD medical alignment
5. Reason if codes match the discharge summary
6. Determine if claim is valid
7. If invalid, prepare and send HTML email with detailed findings

Provide your final answer in this JSON format:
{{"claim_valid": true/false, "justification": "detailed reasoning"}}
"""
        
        # Initialize state for agent with system prompt
        from langchain_core.messages import SystemMessage, HumanMessage
        
        initial_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=agent_input)
        ]
        
        initial_state = {"messages": initial_messages}
        
        # create_react_agent returns a compiled graph
        app = agent_graph
        
        # Run the agent with streaming to capture dynamic nodes and edges
        config = {
            "configurable": {
                "thread_id": f"validation-{hash(state['claim_data'])}"
            }
        }
        logger.info("Invoking ReAct agent for validation with streaming")
        
        final_state = None
        executed_nodes = []
        executed_edges = []
        
        # Stream events to capture dynamic nodes and create traces
        async for event in app.astream(initial_state, config, stream_mode="updates"):
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
        final_state = await app.ainvoke(initial_state, config)
        
        # Store trace information in state (will be accessible via result)
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
            
            logger.info(f"Agent completed. Extracting final answer from: {output[:200]}...")
            
            # Try to parse JSON from output
            import re
            json_match = re.search(r'\{[^{}]*"claim_valid"[^{}]*\}', output, re.DOTALL)
            
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    result = GraphOutput(
                        claim_valid=parsed.get("claim_valid", False),
                        justification=parsed.get("justification", output)
                    )
                    logger.info(f"Validation complete: Claim valid={result.claim_valid}")
                    
                    # Complete trace
                    trace_spans.append({
                        "name": "cpt_icd_validation_complete",
                        "status": "end",
                        "timestamp": asyncio.get_event_loop().time(),
                        "result": {"claim_valid": result.claim_valid}
                    })
                    
                    return result
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON, using full output as justification")
            
            # Fallback: extract from text
            if "claim_valid" in output.lower() or "valid" in output.lower():
                # Try to infer from text
                is_valid = "true" in output.lower() or "valid" in output.lower() and "invalid" not in output.lower()
                result = GraphOutput(
                    claim_valid=is_valid,
                    justification=output
                )
                
                # Complete trace
                trace_spans.append({
                    "name": "cpt_icd_validation_complete",
                    "status": "end",
                    "timestamp": asyncio.get_event_loop().time(),
                    "result": {"claim_valid": result.claim_valid}
                })
                
                return result
            
            result = GraphOutput(
                claim_valid=False,
                justification=f"Could not parse result. Agent output: {output}"
            )
            
            # Complete trace with error
            trace_spans.append({
                "name": "cpt_icd_validation_parse_error",
                "status": "error",
                "timestamp": asyncio.get_event_loop().time(),
                "error": "Could not parse result"
            })
            
            return result
        else:
            logger.error("Agent did not return any messages")
            
            # Complete trace with error
            trace_spans.append({
                "name": "cpt_icd_validation_no_messages_error",
                "status": "error",
                "timestamp": asyncio.get_event_loop().time(),
                "error": "Agent did not return any messages"
            })
            
            return GraphOutput(
                claim_valid=False,
                justification="Agent did not return any validation result"
            )
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}", exc_info=True)
        
        # Complete trace with exception
        trace_spans.append({
            "name": "cpt_icd_validation_exception",
            "status": "error",
            "timestamp": asyncio.get_event_loop().time(),
            "error": str(e),
            "exception_type": type(e).__name__
        })
        
        return GraphOutput(
            claim_valid=False,
            justification=f"Error during validation: {str(e)}"
        )


# Build the graph with dynamic structure
# The ReAct agent handles its own internal flow with dynamic nodes/edges
builder = StateGraph(GraphState, output_schema=GraphOutput)
builder.add_node("validate_cpt_icd_codes", validate_cpt_icd_codes)
builder.add_edge(START, "validate_cpt_icd_codes")
builder.add_edge("validate_cpt_icd_codes", END)

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

