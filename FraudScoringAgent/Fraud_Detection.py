"""
Agent 3: Fraud Scoring Agent
Analyzes claims for fraud risk using historical data via LlamaIndex
Implements human-in-the-loop for medium/high risk escalation
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import ASSETS, CLAIM_CONFIG, LLAMAINDEX_CONFIG
from uipath_utils import get_fraud_context, sdk_manager
from tracing_utils import traced, async_traced, NodeTracer
from agent_1_cpt_icd_validation import Claim

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class FraudRiskLevel(str, Enum):
    """Fraud risk classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FraudScoringResult(BaseModel):
    """Result of fraud risk assessment"""
    claim_id: str
    fraud_risk_level: FraudRiskLevel
    fraud_score: float  # 0.0 to 1.0
    is_fraudulent: bool  # True if HIGH risk
    risk_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    escalation_required: bool = False
    action_id: Optional[str] = None
    scoring_timestamp: str = ""
    historical_context: Dict[str, Any] = Field(default_factory=dict)


class FraudIndicator(BaseModel):
    """Individual fraud indicator found"""
    indicator_type: str  # "BILLING_PATTERN", "CODE_ABUSE", "FREQUENCY_ANOMALY", etc.
    severity: float  # 0.0 to 1.0
    description: str
    evidence: str


# ============================================================================
# LlamaIndex Integration
# ============================================================================

class FraudHistoryAnalyzer:
    """Analyzer using LlamaIndex for fraud history retrieval"""
    
    def __init__(self):
        self.llm = ChatAnthropic(
            model=LLAMAINDEX_CONFIG.MODEL_NAME,
            temperature=0.1
        )
    
    @async_traced()
    async def analyze_provider_history(
        self,
        provider_npi: str,
        patient_id: str,
        claim_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze provider and patient history for fraud indicators
        
        Uses LlamaIndex to query historical fraud rules and patterns
        """
        node_tracer = NodeTracer("analyze_provider_history")
        
        try:
            logger.info(f"Analyzing fraud history for provider {provider_npi}, patient {patient_id}")
            
            # Build query for fraud rules context
            query = f"""
            Historical fraud patterns for:
            - Provider NPI: {provider_npi}
            - Patient ID: {patient_id}
            - Claim codes: {claim_data.get('cpt_codes', [])}
            - Claim amount: ${claim_data.get('total_amount', 0)}
            
            Look for: billing pattern anomalies, unusual code combinations, frequency issues, 
            provider red flags, patient red flags
            """
            
            # Search fraud context grounding
            context_result = await get_fraud_context(query)
            node_tracer.log_data("context_matches", context_result.total_matches)
            
            if context_result.success:
                return {
                    "success": True,
                    "historical_data": context_result.results,
                    "data_quality": "GOOD" if context_result.total_matches > 0 else "LIMITED"
                }
            else:
                logger.warning(f"Fraud context search failed: {context_result.error_message}")
                return {
                    "success": False,
                    "error": context_result.error_message,
                    "data_quality": "UNAVAILABLE"
                }
        
        except Exception as e:
            logger.error(f"Error analyzing provider history: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data_quality": "ERROR"
            }
    
    @async_traced()
    async def identify_fraud_indicators(
        self,
        claim: Claim,
        historical_data: List[Dict[str, Any]]
    ) -> List[FraudIndicator]:
        """
        Use LLM to identify fraud indicators in claim
        """
        node_tracer = NodeTracer("identify_fraud_indicators")
        
        try:
            # Build analysis prompt
            system_prompt = """You are a healthcare fraud detection expert. Analyze this claim for fraud indicators.

            Look for:
            1. BILLING_PATTERN - unusual combinations or quantities
            2. CODE_ABUSE - codes used outside normal parameters
            3. FREQUENCY_ANOMALY - services more frequent than typical
            4. PROVIDER_RED_FLAGS - provider has history of fraud
            5. PATIENT_RED_FLAGS - patient has history of fraud abuse
            6. AMOUNT_ANOMALY - charges unusually high or low
            7. TEMPORAL_ANOMALY - timing of claims seems suspicious
            
            Respond with JSON:
            {
                "indicators": [
                    {
                        "type": "indicator_type",
                        "severity": 0.0-1.0,
                        "description": "brief description",
                        "evidence": "supporting evidence"
                    }
                ],
                "overall_risk_score": 0.0-1.0,
                "recommendations": ["recommendation1", "recommendation2"]
            }"""
            
            claim_text = f"""
Claim Analysis:
- Claim ID: {claim.claim_id}
- Provider NPI: {claim.provider_npi}
- Patient ID: {claim.patient_id}
- Total Amount: ${claim.total_amount}
- Line Items: {len(claim.line_items)}

Line Items:
"""
            for item in claim.line_items:
                claim_text += f"\n- CPT: {item.cpt_code}, ICD: {','.join(item.icd_codes)}, Amount: ${item.amount}, Units: {item.units}"
            
            if historical_data:
                claim_text += f"\n\nHistorical Context:\n{json.dumps(historical_data[:5], default=str)}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=claim_text)
            ]
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            # Parse LLM response
            try:
                response_data = json.loads(response.content)
                indicators = [
                    FraudIndicator(**indicator)
                    for indicator in response_data.get("indicators", [])
                ]
                node_tracer.log_data("indicators_found", len(indicators))
                return indicators
            except json.JSONDecodeError:
                logger.warning("Failed to parse fraud indicator response as JSON")
                return []
        
        except Exception as e:
            logger.error(f"Error identifying fraud indicators: {str(e)}")
            return []


# ============================================================================
# Fraud Scoring Engine
# ============================================================================

class FraudScoringEngine:
    """Engine for fraud risk scoring"""
    
    def __init__(self):
        self.analyzer = FraudHistoryAnalyzer()
    
    @staticmethod
    def _calculate_fraud_score(indicators: List[FraudIndicator]) -> float:
        """
        Calculate overall fraud score from indicators
        
        Score: 0.0 (no fraud) to 1.0 (definite fraud)
        """
        if not indicators:
            return 0.0
        
        # Weight indicators by severity and type
        weights = {
            "BILLING_PATTERN": 0.15,
            "CODE_ABUSE": 0.20,
            "FREQUENCY_ANOMALY": 0.15,
            "PROVIDER_RED_FLAGS": 0.25,
            "PATIENT_RED_FLAGS": 0.15,
            "AMOUNT_ANOMALY": 0.15,
            "TEMPORAL_ANOMALY": 0.10
        }
        
        total_score = 0.0
        for indicator in indicators:
            weight = weights.get(indicator.indicator_type, 0.10)
            total_score += indicator.severity * weight
        
        # Normalize to 0.0-1.0 range
        return min(max(total_score, 0.0), 1.0)
    
    @staticmethod
    def _determine_risk_level(fraud_score: float) -> FraudRiskLevel:
        """Determine risk level from fraud score"""
        if fraud_score >= 0.7:
            return FraudRiskLevel.HIGH
        elif fraud_score >= 0.4:
            return FraudRiskLevel.MEDIUM
        else:
            return FraudRiskLevel.LOW
    
    @async_traced()
    async def score_claim(self, claim: Claim) -> FraudScoringResult:
        """
        Score claim for fraud risk
        """
        node_tracer = NodeTracer("score_claim")
        
        try:
            logger.info(f"Starting fraud scoring for claim {claim.claim_id}")
            node_tracer.log_data("claim_id", claim.claim_id)
            node_tracer.log_data("provider_npi", claim.provider_npi)
            node_tracer.log_data("patient_id", claim.patient_id)
            
            # Prepare claim data for analysis
            claim_data = {
                "cpt_codes": [item.cpt_code for item in claim.line_items],
                "total_amount": claim.total_amount,
                "line_count": len(claim.line_items),
                "claim_date": datetime.now().isoformat()
            }
            
            # Get historical data
            history_result = await self.analyzer.analyze_provider_history(
                provider_npi=claim.provider_npi,
                patient_id=claim.patient_id,
                claim_data=claim_data
            )
            
            historical_data = history_result.get("historical_data", []) if history_result.get("success") else []
            
            # Identify fraud indicators
            indicators = await self.analyzer.identify_fraud_indicators(
                claim=claim,
                historical_data=historical_data
            )
            
            # Calculate fraud score
            fraud_score = self._calculate_fraud_score(indicators)
            risk_level = self._determine_risk_level(fraud_score)
            
            node_tracer.log_data("fraud_score", fraud_score)
            node_tracer.log_data("risk_level", risk_level.value)
            node_tracer.log_data("indicators_count", len(indicators))
            
            # Create result
            result = FraudScoringResult(
                claim_id=claim.claim_id,
                fraud_risk_level=risk_level,
                fraud_score=fraud_score,
                is_fraudulent=(risk_level == FraudRiskLevel.HIGH),
                risk_factors=[f"{ind.indicator_type}: {ind.description}" for ind in indicators],
                recommendations=[],
                escalation_required=(risk_level in [FraudRiskLevel.MEDIUM, FraudRiskLevel.HIGH]),
                scoring_timestamp=datetime.now().isoformat(),
                historical_context={
                    "data_quality": history_result.get("data_quality", "UNKNOWN"),
                    "provider_npi": claim.provider_npi,
                    "patient_id": claim.patient_id
                }
            )
            
            # Add recommendations based on risk level
            if risk_level == FraudRiskLevel.HIGH:
                result.recommendations = [
                    "ESCALATE TO FRAUD INVESTIGATION TEAM",
                    "DENY CLAIM PENDING REVIEW",
                    "REQUEST ADDITIONAL DOCUMENTATION FROM PROVIDER",
                    "CONSIDER PROVIDER ENROLLMENT REVIEW"
                ]
            elif risk_level == FraudRiskLevel.MEDIUM:
                result.recommendations = [
                    "REQUEST HUMAN REVIEW",
                    "REQUEST ADDITIONAL DOCUMENTATION",
                    "VERIFY PATIENT CONTACT",
                    "MONITOR PROVIDER FOR PATTERNS"
                ]
            else:
                result.recommendations = ["APPROVE CLAIM"]
            
            logger.info(f"Fraud scoring complete for claim {claim.claim_id}: score={fraud_score:.2f}, level={risk_level.value}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error scoring claim for fraud: {str(e)}")
            # Return high-risk result on error for safety
            return FraudScoringResult(
                claim_id=claim.claim_id,
                fraud_risk_level=FraudRiskLevel.HIGH,
                fraud_score=0.5,
                is_fraudulent=False,
                risk_factors=[f"Scoring error: {str(e)}"],
                escalation_required=True,
                scoring_timestamp=datetime.now().isoformat()
            )


# ============================================================================
# Human Escalation
# ============================================================================

@async_traced()
async def escalate_to_human(
    claim: Claim,
    fraud_result: FraudScoringResult,
    assignee_email: Optional[str] = None
) -> bool:
    """
    Escalate medium/high risk claim to human for approval/rejection
    Uses UiPath Action Center
    """
    node_tracer = NodeTracer("escalate_to_human")
    
    try:
        logger.info(f"Escalating claim {claim.claim_id} for fraud review (risk: {fraud_result.fraud_risk_level.value})")
        
        # Prepare escalation data
        action_data = {
            "claim_id": claim.claim_id,
            "patient_id": claim.patient_id,
            "provider_npi": claim.provider_npi,
            "fraud_score": fraud_result.fraud_score,
            "risk_level": fraud_result.fraud_risk_level.value,
            "risk_factors": fraud_result.risk_factors,
            "claim_amount": claim.total_amount,
            "recommendations": fraud_result.recommendations,
            "escalation_reason": f"Fraud risk score: {fraud_result.fraud_score:.2f}"
        }
        
        # Create action in Action Center
        action_result = await sdk_manager.create_action(
            app_name="ClaimFraudReviewApp",
            title=f"Fraud Review Required - Claim {claim.claim_id}",
            action_data=action_data,
            assignee_email=assignee_email
        )
        
        if action_result.get("success"):
            fraud_result.action_id = action_result.get("action_id")
            node_tracer.log_data("action_created", True)
            node_tracer.log_data("action_id", fraud_result.action_id)
            logger.info(f"Action created: {fraud_result.action_id}")
            return True
        else:
            logger.error(f"Failed to create action: {action_result.get('error')}")
            return False
    
    except Exception as e:
        logger.error(f"Error escalating to human: {str(e)}")
        return False


# ============================================================================
# Main Agent Interface
# ============================================================================

class FraudScoringAgent:
    """Agent for fraud risk assessment and scoring"""
    
    def __init__(self):
        self.scoring_engine = FraudScoringEngine()
    
    @traced()
    async def score_and_escalate(
        self,
        claim: Claim,
        escalation_assignee: Optional[str] = None
    ) -> FraudScoringResult:
        """
        Score claim for fraud and escalate if needed
        
        Args:
            claim: Claim to score
            escalation_assignee: Email of user to assign escalation to
            
        Returns:
            FraudScoringResult with risk assessment
        """
        logger.info(f"Starting fraud scoring and escalation for claim {claim.claim_id}")
        
        # Score the claim
        result = await self.scoring_engine.score_claim(claim)
        
        # Escalate if needed
        if result.escalation_required:
            escalation_success = await escalate_to_human(
                claim=claim,
                fraud_result=result,
                assignee_email=escalation_assignee
            )
            
            if escalation_success:
                logger.info(f"Claim {claim.claim_id} escalated to human for review")
            else:
                logger.warning(f"Failed to escalate claim {claim.claim_id}")
        else:
            logger.info(f"No escalation required for claim {claim.claim_id} (LOW risk)")
        
        return result