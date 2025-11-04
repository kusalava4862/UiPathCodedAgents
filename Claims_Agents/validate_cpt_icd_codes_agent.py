"""
Agent 1: CPT/ICD Code Validation Agent
Validates CPT and ICD codes against context grounding indexes
Uses LangChain and LangGraph for multi-step validation workflow
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import ASSETS, CLAIM_CONFIG, LANGCHAIN_CONFIG, EMAIL_CONFIG
from uipath_utils import get_cpt_icd_context, sdk_manager
from tracing_utils import traced, async_traced, NodeTracer

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ClaimLineItem(BaseModel):
    """Represents a single line item in a claim"""
    line_number: int
    cpt_code: str
    icd_codes: List[str] = Field(default_factory=list)
    description: str
    units: float
    amount: float


class Claim(BaseModel):
    """Represents a complete claim for validation"""
    claim_id: str
    patient_id: str
    provider_npi: str
    provider_email: str
    discharge_summary: str
    line_items: List[ClaimLineItem] = Field(default_factory=list)
    total_amount: float = 0.0


class ValidationError(BaseModel):
    """Represents a validation error for a specific line item"""
    line_number: int
    cpt_code: str
    error_type: str  # "INVALID_CODE", "INCORRECT_MAPPING", "MISSING_ICD"
    severity: str  # "ERROR", "WARNING"
    message: str
    recommended_action: str


class ValidationResult(BaseModel):
    """Result of CPT/ICD validation"""
    claim_id: str
    claim_valid: bool
    validation_errors: List[ValidationError] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    validation_timestamp: str = ""
    validation_notes: str = ""


# ============================================================================
# LangGraph State
# ============================================================================

class ValidationState(BaseModel):
    """State for the validation workflow"""
    claim: Claim
    validation_result: Optional[ValidationResult] = None
    context_data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    current_step: str = "initialized"


# ============================================================================
# Node Functions with Tracing
# ============================================================================

@async_traced()
async def validate_cpt_codes_node(state: ValidationState) -> ValidationState:
    """
    Node: Validate CPT codes against context grounding index
    """
    node_tracer = NodeTracer("validate_cpt_codes")
    state.current_step = "validate_cpt_codes"
    
    try:
        logger.info(f"Starting CPT code validation for claim {state.claim.claim_id}")
        node_tracer.log_data("claim_id", state.claim.claim_id)
        node_tracer.log_data("line_items_count", len(state.claim.line_items))
        
        # Build CPT codes query
        cpt_codes = [item.cpt_code for item in state.claim.line_items]
        query = f"Validate CPT codes: {', '.join(cpt_codes)}"
        
        # Search context grounding
        context_result = await get_cpt_icd_context(query)
        
        if context_result.success:
            state.context_data["cpt_codes_context"] = context_result.results
            node_tracer.log_data("cpt_context_matches", context_result.total_matches)
            logger.info(f"Found {context_result.total_matches} CPT code matches")
        else:
            state.errors.append(f"CPT context search failed: {context_result.error_message}")
            node_tracer.log_data("cpt_context_error", context_result.error_message)
        
        logger.info(f"CPT validation node completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error in validate_cpt_codes_node: {str(e)}")
        state.errors.append(f"CPT validation error: {str(e)}")
    
    return state


@async_traced()
async def validate_icd_codes_node(state: ValidationState) -> ValidationState:
    """
    Node: Validate ICD codes against context grounding index
    """
    node_tracer = NodeTracer("validate_icd_codes")
    state.current_step = "validate_icd_codes"
    
    try:
        logger.info(f"Starting ICD code validation for claim {state.claim.claim_id}")
        
        # Build ICD codes query including discharge summary for context
        all_icd_codes = []
        for item in state.claim.line_items:
            all_icd_codes.extend(item.icd_codes)
        
        query = f"Validate ICD codes: {', '.join(all_icd_codes)} for conditions: {state.claim.discharge_summary[:200]}"
        
        # Search context grounding
        context_result = await get_cpt_icd_context(query)
        
        if context_result.success:
            state.context_data["icd_codes_context"] = context_result.results
            node_tracer.log_data("icd_context_matches", context_result.total_matches)
            logger.info(f"Found {context_result.total_matches} ICD code matches")
        else:
            state.errors.append(f"ICD context search failed: {context_result.error_message}")
        
        logger.info(f"ICD validation node completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error in validate_icd_codes_node: {str(e)}")
        state.errors.append(f"ICD validation error: {str(e)}")
    
    return state


@async_traced()
async def validate_code_mappings_node(state: ValidationState) -> ValidationState:
    """
    Node: Validate CPT-ICD code mappings and relationships
    Uses LLM to analyze mappings based on discharge summary
    """
    node_tracer = NodeTracer("validate_code_mappings")
    state.current_step = "validate_code_mappings"
    
    try:
        logger.info(f"Starting code mapping validation for claim {state.claim.claim_id}")
        
        # Initialize LLM
        llm = ChatAnthropic(
            model=LANGCHAIN_CONFIG.MODEL_NAME,
            temperature=LANGCHAIN_CONFIG.TEMPERATURE,
            max_tokens=LANGCHAIN_CONFIG.MAX_TOKENS
        )
        
        # Build validation prompt
        system_prompt = """You are a healthcare coding expert. Analyze the CPT and ICD code mappings.
        
Validation rules:
1. CPT codes must be 5 digits (00000-99999)
2. ICD codes must be 3-7 characters (e.g., A00, I10.9)
3. ICD codes must match the procedures indicated by CPT codes
4. Multiple ICD codes are appropriate for complex conditions
5. All codes must be within the discharge summary context

Respond with a JSON object containing:
- valid_mappings: list of line numbers with correct mappings
- invalid_mappings: list of objects with {line_number, issue, recommendation}
- overall_valid: boolean indicating if claim passes validation"""
        
        # Format claim data for LLM
        claim_text = f"""
Claim ID: {state.claim.claim_id}
Patient ID: {state.claim.patient_id}
Provider NPI: {state.claim.provider_npi}

Discharge Summary:
{state.claim.discharge_summary}

Line Items:
"""
        for item in state.claim.line_items:
            claim_text += f"\nLine {item.line_number}: CPT={item.cpt_code}, ICD={','.join(item.icd_codes)}, Description={item.description}"
        
        # Add context grounding results to prompt
        if "cpt_codes_context" in state.context_data:
            claim_text += f"\n\nCPT Code Reference Data: {json.dumps(state.context_data['cpt_codes_context'][:3], default=str)}"
        
        if "icd_codes_context" in state.context_data:
            claim_text += f"\n\nICD Code Reference Data: {json.dumps(state.context_data['icd_codes_context'][:3], default=str)}"
        
        # Call LLM for validation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=claim_text)
        ]
        
        response = await llm.ainvoke(messages)
        
        # Parse LLM response
        try:
            validation_data = json.loads(response.content)
            node_tracer.log_data("llm_validation_result", validation_data)
            state.context_data["mapping_validation"] = validation_data
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            state.errors.append("LLM response parsing failed")
        
        logger.info(f"Code mapping validation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error in validate_code_mappings_node: {str(e)}")
        state.errors.append(f"Code mapping validation error: {str(e)}")
    
    return state


@async_traced()
async def generate_validation_result_node(state: ValidationState) -> ValidationState:
    """
    Node: Generate final validation result with errors and recommendations
    """
    node_tracer = NodeTracer("generate_validation_result")
    state.current_step = "generate_validation_result"
    
    try:
        logger.info(f"Generating validation result for claim {state.claim.claim_id}")
        
        validation_errors: List[ValidationError] = []
        
        # Extract errors from LLM validation
        if "mapping_validation" in state.context_data:
            mapping_data = state.context_data["mapping_validation"]
            
            if "invalid_mappings" in mapping_data:
                for invalid in mapping_data["invalid_mappings"]:
                    # Find matching line item
                    line_number = invalid.get("line_number")
                    matching_item = next(
                        (item for item in state.claim.line_items if item.line_number == line_number),
                        None
                    )
                    
                    if matching_item:
                        validation_errors.append(
                            ValidationError(
                                line_number=line_number,
                                cpt_code=matching_item.cpt_code,
                                error_type="INCORRECT_MAPPING",
                                severity="ERROR",
                                message=invalid.get("issue", "Code mapping invalid"),
                                recommended_action=invalid.get("recommendation", "Review codes")
                            )
                        )
        
        # Create validation result
        claim_valid = len(validation_errors) == 0 and len(state.errors) == 0
        
        state.validation_result = ValidationResult(
            claim_id=state.claim.claim_id,
            claim_valid=claim_valid,
            validation_errors=validation_errors,
            error_count=sum(1 for e in validation_errors if e.severity == "ERROR"),
            warning_count=sum(1 for e in validation_errors if e.severity == "WARNING"),
            validation_timestamp=datetime.now().isoformat(),
            validation_notes="; ".join(state.errors) if state.errors else "Validation completed successfully"
        )
        
        node_tracer.log_data("claim_valid", claim_valid)
        node_tracer.log_data("error_count", state.validation_result.error_count)
        node_tracer.log_data("warning_count", state.validation_result.warning_count)
        
        logger.info(f"Validation result generated: valid={claim_valid}, errors={len(validation_errors)}")
        logger.info(f"Result generation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error generating validation result: {str(e)}")
        state.errors.append(f"Result generation error: {str(e)}")
    
    return state


@async_traced()
async def send_validation_email_node(state: ValidationState) -> ValidationState:
    """
    Node: Send validation report email to provider if errors found
    Uses UiPath process for email delivery
    """
    node_tracer = NodeTracer("send_validation_email")
    state.current_step = "send_validation_email"
    
    try:
        if not state.validation_result:
            logger.warning("No validation result to send")
            return state
        
        # Only send email if there are errors
        if not state.validation_result.validation_errors:
            logger.info("No errors found, skipping email notification")
            return state
        
        logger.info(f"Sending validation email to {state.claim.provider_email}")
        node_tracer.log_data("provider_email", state.claim.provider_email)
        node_tracer.log_data("error_count", state.validation_result.error_count)
        
        # Generate HTML email content
        html_content = generate_validation_email_html(state.claim, state.validation_result)
        
        # Invoke UiPath process to send email
        email_input = {
            "to_email": state.claim.provider_email,
            "subject": f"Claim Validation Report - Claim ID: {state.claim.claim_id}",
            "html_body": html_content,
            "claim_id": state.claim.claim_id
        }
        
        result = await sdk_manager.invoke_process(
            process_name=ASSETS.SEND_EMAIL_PROCESS_NAME,
            input_arguments=email_input
        )
        
        if result.get("success"):
            logger.info(f"Email sent successfully for claim {state.claim.claim_id}")
            node_tracer.log_data("email_sent", True)
        else:
            logger.error(f"Failed to send email: {result.get('error')}")
            state.errors.append(f"Email send failed: {result.get('error')}")
        
        logger.info(f"Email node completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error sending validation email: {str(e)}")
        state.errors.append(f"Email send error: {str(e)}")
    
    return state


# ============================================================================
# Helper Functions
# ============================================================================

def generate_validation_email_html(claim: Claim, result: ValidationResult) -> str:
    """Generate HTML email for validation report"""
    error_rows = ""
    for error in result.validation_errors:
        error_rows += f"""
        <tr>
            <td>{error.line_number}</td>
            <td>{error.cpt_code}</td>
            <td>{error.error_type}</td>
            <td style="color: {'red' if error.severity == 'ERROR' else 'orange'}">{error.severity}</td>
            <td>{error.message}</td>
            <td>{error.recommended_action}</td>
        </tr>
        """
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .header {{ background-color: #f2f2f2; padding: 10px; margin-bottom: 20px; }}
            .footer {{ margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Claim Validation Report</h2>
            <p><strong>Claim ID:</strong> {claim.claim_id}</p>
            <p><strong>Patient ID:</strong> {claim.patient_id}</p>
            <p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Status:</strong> <span style="color: {'green' if result.claim_valid else 'red'};">
                {'✓ VALID' if result.claim_valid else '✗ REQUIRES REVIEW'}</span></p>
        </div>
        
        <h3>Validation Summary</h3>
        <p>Total Errors: <strong>{result.error_count}</strong></p>
        <p>Total Warnings: <strong>{result.warning_count}</strong></p>
        
        <h3>Line Item Issues</h3>
        <table>
            <tr>
                <th>Line</th>
                <th>CPT Code</th>
                <th>Issue Type</th>
                <th>Severity</th>
                <th>Message</th>
                <th>Recommended Action</th>
            </tr>
            {error_rows}
        </table>
        
        <div class="footer">
            <p>Please review the issues above and resubmit the claim with corrections.</p>
            <p>For questions, contact the claims processing team.</p>
        </div>
    </body>
    </html>
    """
    return html


# ============================================================================
# Graph Construction
# ============================================================================

def build_validation_graph() -> StateGraph:
    """Build the CPT/ICD validation workflow graph"""
    
    workflow = StateGraph(ValidationState)
    
    # Add nodes
    workflow.add_node("validate_cpt_codes", validate_cpt_codes_node)
    workflow.add_node("validate_icd_codes", validate_icd_codes_node)
    workflow.add_node("validate_code_mappings", validate_code_mappings_node)
    workflow.add_node("generate_result", generate_validation_result_node)
    workflow.add_node("send_email", send_validation_email_node)
    
    # Add edges
    workflow.set_entry_point("validate_cpt_codes")
    workflow.add_edge("validate_cpt_codes", "validate_icd_codes")
    workflow.add_edge("validate_icd_codes", "validate_code_mappings")
    workflow.add_edge("validate_code_mappings", "generate_result")
    workflow.add_edge("generate_result", "send_email")
    workflow.add_edge("send_email", END)
    
    return workflow


# ============================================================================
# Main Agent Interface
# ============================================================================

class CPTICDValidationAgent:
    """Agent for CPT/ICD code validation"""
    
    def __init__(self):
        self.graph = build_validation_graph().compile()
    
    @traced()
    async def validate_claim(self, claim: Claim) -> ValidationResult:
        """
        Validate a claim's CPT and ICD codes
        
        Args:
            claim: Claim object with line items to validate
            
        Returns:
            ValidationResult with validation status and errors
        """
        logger.info(f"Starting validation for claim {claim.claim_id}")
        
        # Create initial state
        initial_state = ValidationState(claim=claim)
        
        # Execute workflow
        final_state = await asyncio.to_thread(
            self.graph.invoke,
            initial_state
        )
        
        logger.info(f"Validation complete for claim {claim.claim_id}: valid={final_state.validation_result.claim_valid}")
        
        return final_state.validation_result