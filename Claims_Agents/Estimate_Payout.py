"""
Agent 4: Claim Payout Agent
Calculates claim payout based on fee schedule, copay, and deductibles
Generates and sends payout notifications to provider
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

from config import ASSETS, CLAIM_CONFIG, LANGCHAIN_CONFIG
from uipath_utils import get_fee_schedule_context, sdk_manager
from tracing_utils import traced, async_traced, NodeTracer
from agent_1_cpt_icd_validation import Claim

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class LineItemPayout(BaseModel):
    """Payout details for a single line item"""
    line_number: int
    cpt_code: str
    units: float
    charged_amount: float
    allowed_amount: float
    copay_amount: float
    deductible_amount: float
    insurance_pays: float
    patient_pays: float
    write_off: float


class PayoutCalculation(BaseModel):
    """Complete payout calculation for a claim"""
    claim_id: str
    patient_id: str
    provider_npi: str
    provider_email: str
    total_charged: float
    total_allowed: float
    total_copay: float
    total_deductible: float
    total_insurance_pays: float
    total_patient_pays: float
    total_write_off: float
    line_items: List[LineItemPayout] = Field(default_factory=list)
    calculation_timestamp: str = ""
    notes: str = ""


# ============================================================================
# LangGraph State
# ============================================================================

class PayoutState(BaseModel):
    """State for the payout calculation workflow"""
    claim: Claim
    payout_calculation: Optional[PayoutCalculation] = None
    fee_schedule_data: Dict[str, Any] = Field(default_factory=dict)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    current_step: str = "initialized"


# ============================================================================
# Node Functions with Tracing
# ============================================================================

@async_traced()
async def retrieve_fee_schedule_node(state: PayoutState) -> PayoutState:
    """
    Node: Retrieve fee schedule from context grounding
    """
    node_tracer = NodeTracer("retrieve_fee_schedule")
    state.current_step = "retrieve_fee_schedule"
    
    try:
        logger.info(f"Retrieving fee schedule for claim {state.claim.claim_id}")
        node_tracer.log_data("claim_id", state.claim.claim_id)
        node_tracer.log_data("provider_npi", state.claim.provider_npi)
        
        # Build query for fee schedule
        cpt_codes = [item.cpt_code for item in state.claim.line_items]
        query = f"Fee schedule for CPT codes: {', '.join(cpt_codes)}, provider: {state.claim.provider_npi}"
        
        # Search context grounding
        context_result = await get_fee_schedule_context(query)
        
        if context_result.success and context_result.results:
            state.context_data["fee_schedule"] = context_result.results
            node_tracer.log_data("fee_schedule_matches", context_result.total_matches)
            logger.info(f"Found {context_result.total_matches} fee schedule entries")
        else:
            logger.warning(f"No fee schedule found: {context_result.error_message}")
            state.errors.append(f"Fee schedule retrieval: {context_result.error_message}")
        
        logger.info(f"Fee schedule retrieval completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error retrieving fee schedule: {str(e)}")
        state.errors.append(f"Fee schedule retrieval error: {str(e)}")
    
    return state


@async_traced()
async def calculate_line_item_payout_node(state: PayoutState) -> PayoutState:
    """
    Node: Calculate payout for each line item using LLM analysis
    """
    node_tracer = NodeTracer("calculate_line_item_payout")
    state.current_step = "calculate_line_item_payout"
    
    try:
        logger.info(f"Calculating line item payouts for claim {state.claim.claim_id}")
        
        # Initialize LLM
        llm = ChatAnthropic(
            model=LANGCHAIN_CONFIG.MODEL_NAME,
            temperature=LANGCHAIN_CONFIG.TEMPERATURE,
            max_tokens=LANGCHAIN_CONFIG.MAX_TOKENS
        )
        
        # Build calculation prompt
        system_prompt = """You are a healthcare claims payment calculator. Calculate payout for each line item.

Rules:
1. Allowed Amount = fee schedule amount for CPT code (use claim amount if not in fee schedule)
2. Write-off = Charged Amount - Allowed Amount
3. Copay = fixed amount per service (typically $25-50)
4. Deductible = remaining deductible amount (if any)
5. Insurance Pays = Allowed Amount - Copay - Deductible
6. Patient Pays = Copay + Deductible + Write-off

Respond with JSON:
{
    "line_items": [
        {
            "line_number": 1,
            "allowed_amount": 100.00,
            "copay_amount": 25.00,
            "deductible_amount": 0.00,
            "insurance_pays": 75.00,
            "patient_pays": 25.00,
            "write_off": 0.00
        }
    ],
    "totals": {
        "total_allowed": 0.00,
        "total_copay": 0.00,
        "total_deductible": 0.00,
        "total_insurance_pays": 0.00,
        "total_patient_pays": 0.00,
        "total_write_off": 0.00
    }
}"""
        
        # Format claim data
        claim_text = f"""
Claim Information:
- Claim ID: {state.claim.claim_id}
- Patient ID: {state.claim.patient_id}
- Total Charged: ${state.claim.total_amount}

Line Items to Process:
"""
        for item in state.claim.line_items:
            claim_text += f"\nLine {item.line_number}: CPT={item.cpt_code}, Units={item.units}, Charged=${item.amount}"
        
        # Add fee schedule context
        if "fee_schedule" in state.context_data:
            claim_text += f"\n\nFee Schedule Data:\n{json.dumps(state.context_data['fee_schedule'][:5], default=str)}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=claim_text)
        ]
        
        response = await llm.ainvoke(messages)
        
        try:
            payout_data = json.loads(response.content)
            node_tracer.log_data("payout_calculation", payout_data)
            state.context_data["line_item_payouts"] = payout_data
            logger.info(f"Line item calculations completed")
        except json.JSONDecodeError:
            logger.warning("Failed to parse payout calculation as JSON")
            state.errors.append("Payout calculation parsing failed")
        
        logger.info(f"Line item payout calculation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error calculating line item payouts: {str(e)}")
        state.errors.append(f"Payout calculation error: {str(e)}")
    
    return state


@async_traced()
async def generate_payout_result_node(state: PayoutState) -> PayoutState:
    """
    Node: Generate final payout calculation result
    """
    node_tracer = NodeTracer("generate_payout_result")
    state.current_step = "generate_payout_result"
    
    try:
        logger.info(f"Generating payout result for claim {state.claim.claim_id}")
        
        # Extract line item payouts from LLM response
        line_items_list: List[LineItemPayout] = []
        
        if "line_item_payouts" in state.context_data:
            payout_data = state.context_data["line_item_payouts"]
            
            for idx, line_item in enumerate(state.claim.line_items):
                # Match with LLM calculation
                llm_line = None
                if "line_items" in payout_data:
                    llm_lines = payout_data["line_items"]
                    llm_line = next(
                        (item for item in llm_lines if item.get("line_number") == line_item.line_number),
                        None
                    )
                
                if llm_line:
                    line_payout = LineItemPayout(
                        line_number=line_item.line_number,
                        cpt_code=line_item.cpt_code,
                        units=line_item.units,
                        charged_amount=line_item.amount,
                        allowed_amount=llm_line.get("allowed_amount", line_item.amount),
                        copay_amount=llm_line.get("copay_amount", 0.0),
                        deductible_amount=llm_line.get("deductible_amount", 0.0),
                        insurance_pays=llm_line.get("insurance_pays", 0.0),
                        patient_pays=llm_line.get("patient_pays", 0.0),
                        write_off=llm_line.get("write_off", 0.0)
                    )
                else:
                    # Fallback calculation
                    line_payout = LineItemPayout(
                        line_number=line_item.line_number,
                        cpt_code=line_item.cpt_code,
                        units=line_item.units,
                        charged_amount=line_item.amount,
                        allowed_amount=line_item.amount,
                        copay_amount=25.0,
                        deductible_amount=0.0,
                        insurance_pays=line_item.amount - 25.0,
                        patient_pays=25.0,
                        write_off=0.0
                    )
                
                line_items_list.append(line_payout)
        
        # Calculate totals
        total_charged = sum(item.charged_amount for item in line_items_list)
        total_allowed = sum(item.allowed_amount for item in line_items_list)
        total_copay = sum(item.copay_amount for item in line_items_list)
        total_deductible = sum(item.deductible_amount for item in line_items_list)
        total_insurance_pays = sum(item.insurance_pays for item in line_items_list)
        total_patient_pays = sum(item.patient_pays for item in line_items_list)
        total_write_off = sum(item.write_off for item in line_items_list)
        
        # Create payout calculation
        state.payout_calculation = PayoutCalculation(
            claim_id=state.claim.claim_id,
            patient_id=state.claim.patient_id,
            provider_npi=state.claim.provider_npi,
            provider_email=state.claim.provider_email,
            total_charged=total_charged,
            total_allowed=total_allowed,
            total_copay=total_copay,
            total_deductible=total_deductible,
            total_insurance_pays=total_insurance_pays,
            total_patient_pays=total_patient_pays,
            total_write_off=total_write_off,
            line_items=line_items_list,
            calculation_timestamp=datetime.now().isoformat(),
            notes="; ".join(state.errors) if state.errors else "Payout calculated successfully"
        )
        
        node_tracer.log_data("total_insurance_pays", total_insurance_pays)
        node_tracer.log_data("total_patient_pays", total_patient_pays)
        node_tracer.log_data("line_items_count", len(line_items_list))
        
        logger.info(f"Payout result generated: insurance_pays=${total_insurance_pays:.2f}, patient_pays=${total_patient_pays:.2f}")
        logger.info(f"Result generation completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error generating payout result: {str(e)}")
        state.errors.append(f"Result generation error: {str(e)}")
    
    return state


@async_traced()
async def send_payout_email_node(state: PayoutState) -> PayoutState:
    """
    Node: Send payout notification email to provider
    Uses UiPath process for email delivery
    """
    node_tracer = NodeTracer("send_payout_email")
    state.current_step = "send_payout_email"
    
    try:
        if not state.payout_calculation:
            logger.warning("No payout calculation to send")
            return state
        
        logger.info(f"Sending payout email to {state.claim.provider_email}")
        node_tracer.log_data("provider_email", state.claim.provider_email)
        node_tracer.log_data("payout_amount", state.payout_calculation.total_insurance_pays)
        
        # Generate HTML email content
        html_content = generate_payout_email_html(state.payout_calculation)
        
        # Invoke UiPath process to send email
        email_input = {
            "to_email": state.claim.provider_email,
            "subject": f"Claim Payment Notification - Claim ID: {state.payout_calculation.claim_id}",
            "html_body": html_content,
            "claim_id": state.payout_calculation.claim_id,
            "payout_amount": state.payout_calculation.total_insurance_pays
        }
        
        result = await sdk_manager.invoke_process(
            process_name=ASSETS.SEND_EMAIL_PROCESS_NAME,
            input_arguments=email_input
        )
        
        if result.get("success"):
            logger.info(f"Payout email sent successfully for claim {state.claim.claim_id}")
            node_tracer.log_data("email_sent", True)
        else:
            logger.error(f"Failed to send payout email: {result.get('error')}")
            state.errors.append(f"Email send failed: {result.get('error')}")
        
        logger.info(f"Payout email node completed in {node_tracer.get_duration():.2f}s")
        
    except Exception as e:
        logger.error(f"Error sending payout email: {str(e)}")
        state.errors.append(f"Email send error: {str(e)}")
    
    return state


# ============================================================================
# Helper Functions
# ============================================================================

def generate_payout_email_html(payout: PayoutCalculation) -> str:
    """Generate HTML email for payout notification"""
    
    line_rows = ""
    for line in payout.line_items:
        line_rows += f"""
        <tr>
            <td>{line.line_number}</td>
            <td>{line.cpt_code}</td>
            <td>{line.units}</td>
            <td>${line.charged_amount:.2f}</td>
            <td>${line.allowed_amount:.2f}</td>
            <td>${line.copay_amount:.2f}</td>
            <td>${line.insurance_pays:.2f}</td>
            <td>${line.patient_pays:.2f}</td>
            <td>${line.write_off:.2f}</td>
        </tr>
        """
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
            th {{ background-color: #2196F3; color: white; text-align: left; }}
            .summary-box {{ background-color: #f2f2f2; padding: 15px; margin: 20px 0; border-radius: 5px; }}
            .summary-row {{ display: flex; justify-content: space-between; margin: 8px 0; font-size: 14px; }}
            .total-row {{ font-weight: bold; font-size: 16px; color: #2196F3; }}
            .header {{ background-color: #e3f2fd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
            .footer {{ margin-top: 20px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Claim Payment Notification</h2>
            <p><strong>Claim ID:</strong> {payout.claim_id}</p>
            <p><strong>Patient ID:</strong> {payout.patient_id}</p>
            <p><strong>Payment Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        
        <h3>Payment Summary</h3>
        <div class="summary-box">
            <div class="summary-row">
                <span>Total Charged Amount:</span>
                <span>${payout.total_charged:.2f}</span>
            </div>
            <div class="summary-row">
                <span>Total Allowed Amount:</span>
                <span>${payout.total_allowed:.2f}</span>
            </div>
            <div class="summary-row">
                <span>Total Patient Copay:</span>
                <span>${payout.total_copay:.2f}</span>
            </div>
            <div class="summary-row">
                <span>Total Deductible Applied:</span>
                <span>${payout.total_deductible:.2f}</span>
            </div>
            <div class="summary-row">
                <span>Total Write-off:</span>
                <span>${payout.total_write_off:.2f}</span>
            </div>
            <div class="summary-row total-row">
                <span>INSURANCE PAYMENT AMOUNT:</span>
                <span>${payout.total_insurance_pays:.2f}</span>
            </div>
            <div class="summary-row">
                <span>Total Patient Responsibility:</span>
                <span>${payout.total_patient_pays:.2f}</span>
            </div>
        </div>
        
        <h3>Line Item Breakdown</h3>
        <table>
            <tr>
                <th>Line</th>
                <th>CPT Code</th>
                <th>Units</th>
                <th>Charged</th>
                <th>Allowed</th>
                <th>Copay</th>
                <th>Insurance Pays</th>
                <th>Patient Pays</th>
                <th>Write-off</th>
            </tr>
            {line_rows}
        </table>
        
        <div class="footer">
            <p>Payment Instructions: Please allow 5-10 business days for payment processing.</p>
            <p>Questions? Contact claims@healthcare.com or call 1-800-XXX-XXXX</p>
            <p><strong>Note:</strong> Patient is responsible for copay and any deductible amounts.</p>
        </div>
    </body>
    </html>
    """
    return html


# ============================================================================
# Graph Construction
# ============================================================================

def build_payout_graph() -> StateGraph:
    """Build the claim payout workflow graph"""
    
    workflow = StateGraph(PayoutState)
    
    # Add nodes
    workflow.add_node("retrieve_fee_schedule", retrieve_fee_schedule_node)
    workflow.add_node("calculate_payouts", calculate_line_item_payout_node)
    workflow.add_node("generate_result", generate_payout_result_node)
    workflow.add_node("send_email", send_payout_email_node)
    
    # Add edges
    workflow.set_entry_point("retrieve_fee_schedule")
    workflow.add_edge("retrieve_fee_schedule", "calculate_payouts")
    workflow.add_edge("calculate_payouts", "generate_result")
    workflow.add_edge("generate_result", "send_email")
    workflow.add_edge("send_email", END)
    
    return workflow


# ============================================================================
# Main Agent Interface
# ============================================================================

class ClaimPayoutAgent:
    """Agent for claim payout calculation and notification"""
    
    def __init__(self):
        self.graph = build_payout_graph().compile()
    
    @traced()
    async def calculate_and_notify_payout(self, claim: Claim) -> PayoutCalculation:
        """
        Calculate claim payout and send notification to provider
        
        Args:
            claim: Approved claim with all validations completed
            
        Returns:
            PayoutCalculation with detailed payout breakdown
        """
        logger.info(f"Starting payout calculation for claim {claim.claim_id}")
        
        # Create initial state
        initial_state = PayoutState(claim=claim)
        
        # Execute workflow
        final_state = await asyncio.to_thread(
            self.graph.invoke,
            initial_state
        )
        
        logger.info(f"Payout calculation complete for claim {claim.claim_id}: amount=${final_state.payout_calculation.total_insurance_pays if final_state.payout_calculation else 0:.2f}")
        
        return final_state.payout_calculation