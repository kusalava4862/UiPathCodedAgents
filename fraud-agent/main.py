import json

from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.workflow import (
    Context,
    Event,
    HumanResponseEvent,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from uipath import UiPath
from uipath.tracing import traced, wait_for_tracers
from uipath_llamaindex.llms import UiPathOpenAI
from uipath_llamaindex.models import CreateActionEvent
from uipath_llamaindex.query_engines import ContextGroundingQueryEngine

# ============================================================================
# CONFIGURATION VARIABLES - Set all details here
# ============================================================================

# Model configuration
MODEL_NAME = "gpt-4o-2024-11-20"

# RAG Configuration
INDEX_FOLDER_PATH = "UiPath_SDK_Challenge"
HISTORICAL_CLAIMS_INDEX_NAME = "Historical_Claims"
#HISTORICAL_CLAIMS_FILES_DIRECTORY = "sample_data/historical_claims"

# HITL Configuration
HITL_APP_NAME = "Claims.Adjudication.Processes.webApp.Escalation.APP.for.Fraud.Investigator"
HITL_APP_VERSION = 1
HITL_APP_FOLDER_PATH = "UiPath_SDK_Challenge/Claims Adjudication Processes"
AGENT_NAME = "Healthcare Fraud Detection AI Agent"

# Email Process Configuration
EMAIL_PROCESS_NAME = "Send Email"
EMAIL_PROCESS_FOLDER_PATH = "UiPath_SDK_Challenge/Claims Adjudication Processes"

# Risk Thresholds
LOW_RISK_MAX = 0.3
MEDIUM_RISK_MAX = 0.7
HIGH_RISK_MAX = 1.0

# ============================================================================
# INITIALIZATION
# ============================================================================

llm = UiPathOpenAI(model=MODEL_NAME)
uipath = UiPath()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


@traced(name="generate_historical_claims_query_engine", run_type="fraud_detection")
def generate_historical_claims_query_engine(
    response_mode: ResponseMode = ResponseMode.SIMPLE_SUMMARIZE,
) -> ContextGroundingQueryEngine:
    """Generate a query engine for historical claims data."""
    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode, llm=llm
    )
    query_engine = ContextGroundingQueryEngine(
        index_name=HISTORICAL_CLAIMS_INDEX_NAME,
        folder_path=INDEX_FOLDER_PATH,
        response_synthesizer=response_synthesizer,
    )
    return query_engine


@traced(name="analyze_claim_fraud_risk", run_type="fraud_detection")
async def analyze_claim_fraud_risk(
    claim_data: dict, historical_context: str
) -> dict:
    """
    Analyze a claim for fraud risk by comparing with historical data.
    
    Args:
        claim_data: The claim data to analyze
        historical_context: Historical claims context from RAG
        
    Returns:
        Dictionary with fraud_risk_score, risk_classification, reasons, etc.
    """
    system_prompt = """You are a healthcare fraud detection AI agent. Your primary function is to analyze insurance claims by comparing them with historical claim records to identify unusual or suspicious patterns that may indicate fraudulent activity. You will:

1. Detect anomalies such as repeated procedures, same-day services, or unusually high-cost procedures.
2. Identify patterns of frequent or clustered CPT codes for the same provider or patient.
3. Calculate a fraud risk score between 0 and 1 for each claim.
4. Classify claims as low, medium, or high risk based on the fraud risk score.
5. Provide structured output with reasons for the risk classification and an escalation flag for high-risk or ambiguous cases.

Maintain a neutral and objective tone in your analysis. Your role is to identify potential fraud indicators, not to make final determinations of fraud. Always base your assessments on the data provided and avoid making assumptions beyond the available information."""

    analysis_prompt = f"""
Analyze the following claim using the historical claims data provided:

Claim: {json.dumps(claim_data, indent=2)}

Historical Claims Context:
{historical_context}

Follow these steps:

1. Compare the claim with historical data to identify any anomalies or suspicious patterns.
2. Calculate a fraud_risk_score between 0 and 1 based on your findings.
3. Classify the claim as low (0-0.3), medium (0.31-0.7), or high (0.71-1) risk.
4. Provide a JSON output with the following structure:
   {{
     "fraud_risk_score": <calculated score>,
     "risk_classification": "<low|medium|high>",
     "reasons": [<list of reasons for the classification>],
     "escalation_flag": <true if high / medium risk or ambiguous, false otherwise>,
     "human_review_required": <true for low and medium risk, false for high risk>
   }}

5. Ensure all reasons are specific and data-driven.

Process the claim and provide your analysis. Return ONLY valid JSON, no additional text.
"""

    response = await llm.acomplete(analysis_prompt)
    response_text = str(response).strip()

    # Try to extract JSON from response
    try:
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        analysis_result = json.loads(response_text)
        return analysis_result
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object from text
        import re

        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text)
        if json_match:
            analysis_result = json.loads(json_match.group())
            return analysis_result
        else:
            # Default response if parsing fails
            return {
                "fraud_risk_score": 0.5,
                "risk_classification": "medium",
                "reasons": ["Unable to parse analysis response"],
                "escalation_flag": True,
                "human_review_required": True,
            }


@traced(name="escalate_to_hitl", run_type="fraud_detection")
async def escalate_to_hitl(
    ctx: Context,
    claim_data: dict,
    analysis_result: dict,
    provider_emailaddress: str,
) -> bool:
    """
    Escalate to human-in-the-loop for review.
    
    Args:
        ctx: Workflow context
        claim_data: The claim data
        analysis_result: The fraud analysis result
        provider_emailaddress: Provider email address
        
    Returns:
        True if approved, False if rejected
    """
    # Create HITL escalation event
    ctx.write_event_to_stream(
        CreateActionEvent(
            prefix="hitl escalation for fraud detection",
            app_name=HITL_APP_NAME,
            title=f"Fraud Detection Review Required - {analysis_result.get('risk_classification', 'unknown').upper()} Risk",
            data={
                "AgentOutput": f"""
Fraud Detection Analysis Results:

Risk Classification: {analysis_result.get('risk_classification', 'unknown').upper()}
Fraud Risk Score: {analysis_result.get('fraud_risk_score', 0):.2f}

Reasons:
{chr(10).join(f"- {reason}" for reason in analysis_result.get('reasons', []))}

Claim Data:
{json.dumps(claim_data, indent=2)}

Provider Email: {provider_emailaddress}

Please review this claim and approve or reject it.
                """.strip(),
                "AgentName": AGENT_NAME,
                "ClaimData": claim_data,
                "AnalysisResult": analysis_result,
                "ProviderEmail": provider_emailaddress,
            },
            app_version=HITL_APP_VERSION,
            app_folder_path=HITL_APP_FOLDER_PATH,
        )
    )

    # Wait for human response
    hitl_response = await ctx.wait_for_event(HumanResponseEvent)
    feedback = json.loads(hitl_response.response)

    # Check if approved
    if isinstance(feedback.get("Answer"), bool) and feedback["Answer"] is True:
        return True
    elif isinstance(feedback.get("approved"), bool) and feedback["approved"] is True:
        return True
    else:
        return False


@traced(name="send_rejection_email", run_type="fraud_detection", hide_input=True)
async def send_rejection_email(
    provider_emailaddress: str,
    claim_data: dict,
    analysis_result: dict,
) -> None:
    """
    Send rejection email to provider using process calling.
    
    Args:
        provider_emailaddress: Provider email address
        claim_data: The claim data
        analysis_result: The fraud analysis result
    """
    # Generate HTML email content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #d32f2f; color: white; padding: 20px; text-align: center; }}
            .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; }}
            .risk-badge {{ display: inline-block; padding: 5px 15px; border-radius: 5px; font-weight: bold; }}
            .risk-high {{ background-color: #d32f2f; color: white; }}
            .risk-medium {{ background-color: #f57c00; color: white; }}
            .risk-low {{ background-color: #388e3c; color: white; }}
            .claim-details {{ background-color: white; padding: 15px; margin: 15px 0; border-left: 4px solid #1976d2; }}
            .reasons {{ background-color: white; padding: 15px; margin: 15px 0; }}
            .reasons ul {{ margin: 10px 0; padding-left: 20px; }}
            .footer {{ margin-top: 20px; padding: 15px; background-color: #e0e0e0; font-size: 12px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Claim Rejection Notice</h1>
            </div>
            <div class="content">
                <p>Dear Provider,</p>
                <p>We regret to inform you that the following claim has been rejected after fraud detection analysis:</p>
                
                <div class="claim-details">
                    <h3>Claim Details</h3>
                    <pre>{json.dumps(claim_data, indent=2)}</pre>
                </div>
                
                <div class="reasons">
                    <h3>Fraud Risk Analysis</h3>
                    <p><strong>Risk Classification:</strong> 
                        <span class="risk-badge risk-{analysis_result.get('risk_classification', 'medium')}">
                            {analysis_result.get('risk_classification', 'medium').upper()}
                        </span>
                    </p>
                    <p><strong>Fraud Risk Score:</strong> {analysis_result.get('fraud_risk_score', 0):.2f}</p>
                    <h4>Reasons for Rejection:</h4>
                    <ul>
                        {''.join(f'<li>{reason}</li>' for reason in analysis_result.get('reasons', []))}
                    </ul>
                </div>
                
                <p>If you believe this rejection is in error, please contact our support team for further review.</p>
                
                <div class="footer">
                    <p>This is an automated message from the Healthcare Fraud Detection System.</p>
                    <p>Please do not reply to this email.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    # Call UiPath process to send email
    # This uses process calling instead of MCP
    try:
        # Execute the email sending process
        process_input = {
            "To": provider_emailaddress,
            "Subject": f"Claim Rejection - Fraud Detection Alert - Risk: {analysis_result.get('risk_classification', 'unknown').upper()}",
            "Body": html_content,
            "IsBodyHtml": True,
        }

        # Use UiPath process execution
        # Note: Adjust process name and parameters based on your actual email process
        job = await uipath.jobs.start_job_async(
            process_name=EMAIL_PROCESS_NAME,
            folder_path=EMAIL_PROCESS_FOLDER_PATH,
            input_arguments=process_input,
        )

        print(f"Email process job started: {job.id}")
        # Wait for job completion if needed
        # await uipath.jobs.wait_for_job_completion_async(job.id)

    except Exception as e:
        print(f"Error sending email via process: {e}")
        # Fallback: You could implement direct email sending here if process fails
        raise


# ============================================================================
# WORKFLOW EVENTS
# ============================================================================


class ClaimAnalysisEvent(StartEvent):
    """Event representing a claim to be analyzed."""

    claim_data: dict
    provider_emailaddress: str


class HistoricalContextEvent(Event):
    """Event containing historical claims context."""

    historical_context: str


class FraudAnalysisEvent(Event):
    """Event containing fraud analysis results."""

    analysis_result: dict


class HITLEscalationEvent(Event):
    """Event for HITL escalation."""

    approved: bool


class EmailSentEvent(Event):
    """Event indicating email was sent."""

    pass


class ClaimProcessingResultEvent(StopEvent):
    """Final event with claim processing result."""

    continue_claim_processing: bool
    analysis_result: dict
    message: str


# ============================================================================
# FRAUD DETECTION WORKFLOW
# ============================================================================


class FraudDetectionWorkflow(Workflow):
    """Workflow for fraud detection in insurance claims."""

    @step
    @traced(name="retrieve_historical_context", run_type="fraud_detection")
    async def retrieve_historical_context(
        self, ctx: Context, ev: ClaimAnalysisEvent
    ) -> HistoricalContextEvent:
        """Retrieve historical claims context using RAG."""
        query_engine = generate_historical_claims_query_engine()

        # Create a query to find similar historical claims
        claim_summary = json.dumps(ev.claim_data)
        query = f"""
        Find historical claims that are similar to this claim or show patterns that might be relevant for fraud detection:
        {claim_summary}
        
        Focus on:
        - Similar CPT codes
        - Same provider patterns
        - Same patient patterns
        - Unusual billing patterns
        - High-cost procedures
        - Same-day services
        """

        response = await query_engine.aquery(query)
        historical_context = str(response)

        await ctx.store.set("claim_data", ev.claim_data)
        await ctx.store.set("provider_emailaddress", ev.provider_emailaddress)

        return HistoricalContextEvent(historical_context=historical_context)

    @step
    @traced(name="analyze_fraud_risk", run_type="fraud_detection")
    async def analyze_fraud_risk(
        self, ctx: Context, ev: HistoricalContextEvent
    ) -> FraudAnalysisEvent | ClaimProcessingResultEvent:
        """Analyze the claim for fraud risk."""
        claim_data = await ctx.store.get("claim_data")
        analysis_result = await analyze_claim_fraud_risk(
            claim_data, ev.historical_context
        )

        risk_classification = analysis_result.get("risk_classification", "medium")
        fraud_risk_score = analysis_result.get("fraud_risk_score", 0.5)

        # If low risk, automatically approve
        if risk_classification == "low" and fraud_risk_score <= LOW_RISK_MAX:
            return ClaimProcessingResultEvent(
                continue_claim_processing=True,
                analysis_result=analysis_result,
                message="Claim approved automatically - Low risk detected",
            )

        # Otherwise, escalate to HITL
        return FraudAnalysisEvent(analysis_result=analysis_result)

    @step
    @traced(name="escalate_hitl", run_type="fraud_detection")
    async def escalate_hitl(
        self, ctx: Context, ev: FraudAnalysisEvent
    ) -> HITLEscalationEvent:
        """Escalate to human-in-the-loop for review."""
        claim_data = await ctx.store.get("claim_data")
        provider_emailaddress = await ctx.store.get("provider_emailaddress")

        approved = await escalate_to_hitl(
            ctx, claim_data, ev.analysis_result, provider_emailaddress
        )

        await ctx.store.set("hitl_approved", approved)
        await ctx.store.set("analysis_result", ev.analysis_result)

        return HITLEscalationEvent(approved=approved)

    @step
    @traced(name="handle_hitl_result", run_type="fraud_detection")
    async def handle_hitl_result(
        self, ctx: Context, ev: HITLEscalationEvent
    ) -> ClaimProcessingResultEvent | EmailSentEvent:
        """Handle the result from HITL escalation."""
        claim_data = await ctx.store.get("claim_data")
        provider_emailaddress = await ctx.store.get("provider_emailaddress")
        analysis_result = await ctx.store.get("analysis_result")

        if ev.approved:
            # Approved - continue processing
            return ClaimProcessingResultEvent(
                continue_claim_processing=True,
                analysis_result=analysis_result,
                message="Claim approved after human review",
            )
        else:
            # Rejected - send email
            return EmailSentEvent()

    @step
    @traced(name="send_rejection_email_step", run_type="fraud_detection")
    async def send_rejection_email_step(
        self, ctx: Context, ev: EmailSentEvent
    ) -> ClaimProcessingResultEvent:
        """Send rejection email to provider."""
        claim_data = await ctx.store.get("claim_data")
        provider_emailaddress = await ctx.store.get("provider_emailaddress")
        analysis_result = await ctx.store.get("analysis_result")

        await send_rejection_email(provider_emailaddress, claim_data, analysis_result)

        return ClaimProcessingResultEvent(
            continue_claim_processing=False,
            analysis_result=analysis_result,
            message="Claim rejected - Rejection email sent to provider",
        )


# ============================================================================
# WORKFLOW INSTANCE
# ============================================================================

workflow = FraudDetectionWorkflow(timeout=300, verbose=True)

