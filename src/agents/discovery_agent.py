"""
Discovery Agent for market research and requirement gathering.
Specializes in analyzing market opportunities and gathering user requirements.
"""

import logging
from typing import Dict, Any, Optional, List

from pydantic_ai import RunContext

from .base_agent import BaseSubAgent
from ..models.agent_models import AgentDependencies, AgentCapability
from ..models.context_models import ContextSnapshot
from ..models.workflow_models import AgentType

logger = logging.getLogger(__name__)


class DiscoveryAgent(BaseSubAgent):
    """
    Discovery Agent specializing in market research and requirement gathering.
    """
    
    def __init__(self):
        """Initialize the Discovery Agent."""
        super().__init__(AgentType.DISCOVERY)
        self._setup_capabilities()
    
    def _setup_capabilities(self):
        """Setup agent capabilities."""
        self.capabilities = [
            AgentCapability(
                capability_id="market_research",
                name="Market Research",
                description="Analyze market opportunities and competitive landscape",
                input_schema={"project_description": "str", "target_market": "str"},
                output_schema={"market_analysis": "dict", "competitive_analysis": "dict"},
                required_tools=["web_research", "data_analysis"]
            ),
            AgentCapability(
                capability_id="requirement_gathering",
                name="Requirement Gathering",  
                description="Gather and analyze user requirements and constraints",
                input_schema={"stakeholders": "list", "project_goals": "str"},
                output_schema={"user_requirements": "list", "constraints": "list"},
                required_tools=["stakeholder_analysis", "requirement_modeling"]
            )
        ]
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Discovery Agent."""
        return """
        You are a Discovery Agent specializing in market research and requirement gathering.
        Your role is to analyze market opportunities, gather user requirements, and identify
        project constraints and objectives.
        
        Core Responsibilities:
        1. **Market Research**: Analyze target markets, identify opportunities, assess competition
        2. **User Research**: Understand user needs, behaviors, and pain points
        3. **Requirement Elicitation**: Gather functional and non-functional requirements
        4. **Stakeholder Analysis**: Identify key stakeholders and their interests
        5. **Constraint Identification**: Identify technical, business, and regulatory constraints
        6. **Opportunity Assessment**: Evaluate market potential and business viability
        
        Capabilities:
        - Conduct comprehensive market analysis
        - Perform competitive research and benchmarking
        - Analyze user personas and journey mapping
        - Facilitate requirement gathering sessions
        - Identify risks and constraints
        - Create project feasibility assessments
        
        Always provide:
        - Evidence-based insights with sources
        - Structured analysis with clear recommendations
        - Actionable requirements with priorities
        - Risk assessment with mitigation strategies
        - Clear documentation for handoff to next stage
        
        Focus on understanding the "why" behind the project and ensuring all requirements
        are properly captured and prioritized before moving to specification phase.
        """
    
    def get_required_tools(self) -> List[str]:
        """Get required tools for the Discovery Agent."""
        return [
            "web_research",
            "market_analysis", 
            "competitive_analysis",
            "stakeholder_mapping",
            "requirement_modeling",
            "user_persona_analysis"
        ]
    
    def _get_required_output_keys(self) -> List[str]:
        """Get required output keys for Discovery Agent."""
        return [
            "market_analysis",
            "user_requirements",
            "stakeholder_analysis", 
            "constraints",
            "project_objectives",
            "success_metrics"
        ]
    
    async def validate_input(self, task: str, context: Optional[ContextSnapshot] = None) -> bool:
        """Validate input for the Discovery Agent."""
        if not task or len(task.strip()) < 10:
            logger.error("Task description too short for discovery analysis")
            return False
        
        # Check if task contains basic project information
        required_elements = ["project", "product", "application", "system", "platform"]
        has_project_element = any(element in task.lower() for element in required_elements)
        
        if not has_project_element:
            logger.warning("Task may not contain clear project description")
        
        return True
    
    async def process_task(
        self,
        task: str,
        context: Optional[ContextSnapshot] = None,
        dependencies: Optional[AgentDependencies] = None
    ) -> Dict[str, Any]:
        """Process the discovery task."""
        logger.info(f"Processing discovery task: {task[:100]}...")
        
        # Use the Pydantic AI agent to process the task
        if dependencies:
            result = await self.agent.run(
                f"""
                Conduct comprehensive discovery analysis for the following project:
                
                Project Description: {task}
                
                Please provide a thorough analysis including:
                1. Market analysis and opportunity assessment
                2. User requirements and personas
                3. Stakeholder identification and analysis
                4. Technical and business constraints
                5. Project objectives and success metrics
                6. Risk assessment and mitigation strategies
                
                Format your response as structured data that can be used by the specification agent.
                """,
                deps=dependencies
            )
            
            # Process the result into structured format
            return await self._structure_discovery_results(result.data if hasattr(result, 'data') else str(result))
        else:
            # Fallback processing without dependencies
            return await self._process_task_standalone(task)
    
    async def _structure_discovery_results(self, raw_result: Any) -> Dict[str, Any]:
        """Structure the raw discovery results into standardized format."""
        # This would parse and structure the LLM output
        # For now, providing a structured template
        
        structured_result = {
            "market_analysis": {
                "target_market": "Analysis of target market",
                "market_size": "Market size estimation", 
                "growth_trends": "Market growth trends",
                "key_opportunities": ["Opportunity 1", "Opportunity 2"]
            },
            "competitive_analysis": {
                "direct_competitors": ["Competitor 1", "Competitor 2"],
                "indirect_competitors": ["Alternative 1", "Alternative 2"],
                "competitive_advantages": ["Advantage 1", "Advantage 2"],
                "market_gaps": ["Gap 1", "Gap 2"]
            },
            "user_requirements": {
                "functional_requirements": ["Requirement 1", "Requirement 2"],
                "non_functional_requirements": ["Performance", "Security"],
                "user_personas": ["Persona 1", "Persona 2"],
                "user_journeys": ["Journey 1", "Journey 2"]
            },
            "stakeholder_analysis": {
                "primary_stakeholders": ["Stakeholder 1", "Stakeholder 2"],
                "secondary_stakeholders": ["Stakeholder 3", "Stakeholder 4"],
                "stakeholder_interests": {"Stakeholder 1": "Interest 1"},
                "influence_power_matrix": {}
            },
            "constraints": {
                "technical_constraints": ["Constraint 1", "Constraint 2"],
                "business_constraints": ["Budget", "Timeline"],
                "regulatory_constraints": ["Compliance 1", "Compliance 2"],
                "resource_constraints": ["Skills", "Tools"]
            },
            "project_objectives": {
                "primary_objectives": ["Objective 1", "Objective 2"],
                "secondary_objectives": ["Objective 3", "Objective 4"],
                "success_criteria": ["Criteria 1", "Criteria 2"]
            },
            "success_metrics": {
                "business_metrics": ["Revenue", "User adoption"],
                "technical_metrics": ["Performance", "Availability"],
                "user_metrics": ["Satisfaction", "Engagement"]
            },
            "risk_assessment": {
                "high_risks": ["Risk 1", "Risk 2"],
                "medium_risks": ["Risk 3", "Risk 4"],
                "mitigation_strategies": {"Risk 1": "Mitigation 1"}
            },
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2",
                "Recommendation 3"
            ]
        }
        
        # In a real implementation, this would parse the raw_result
        # and populate the structured format
        if isinstance(raw_result, str):
            structured_result["raw_analysis"] = raw_result
        elif isinstance(raw_result, dict):
            structured_result.update(raw_result)
        
        return structured_result
    
    async def _process_task_standalone(self, task: str) -> Dict[str, Any]:
        """Process task without external dependencies (fallback mode)."""
        logger.info("Processing discovery task in standalone mode")
        
        # Basic analysis based on task description
        return {
            "market_analysis": {
                "analysis_method": "standalone",
                "project_description": task,
                "estimated_complexity": self._estimate_complexity(task)
            },
            "user_requirements": {
                "inferred_requirements": self._infer_requirements(task)
            },
            "constraints": {
                "assumed_constraints": ["Budget limitations", "Time constraints"]
            },
            "project_objectives": {
                "primary_objective": self._extract_primary_objective(task)
            },
            "recommendations": [
                "Conduct user interviews for detailed requirements",
                "Perform comprehensive market research",
                "Validate assumptions with stakeholders"
            ]
        }
    
    def _register_agent_tools(self):
        """Register Discovery Agent specific tools."""
        
        @self.agent.tool
        async def conduct_market_research(
            ctx: RunContext[AgentDependencies],
            market_segment: str,
            research_depth: str = "comprehensive"
        ) -> Dict[str, Any]:
            """Conduct market research for a specific segment."""
            logger.info(f"Conducting {research_depth} market research for: {market_segment}")
            
            # This would integrate with actual research tools
            return {
                "segment": market_segment,
                "research_depth": research_depth,
                "findings": f"Market research results for {market_segment}",
                "data_sources": ["Industry reports", "Market surveys", "Competitor analysis"]
            }
        
        @self.agent.tool  
        async def analyze_user_personas(
            ctx: RunContext[AgentDependencies],
            target_users: str,
            analysis_method: str = "behavioral"
        ) -> Dict[str, Any]:
            """Analyze user personas and behaviors."""
            logger.info(f"Analyzing user personas: {target_users}")
            
            return {
                "target_users": target_users,
                "analysis_method": analysis_method,
                "personas": f"User personas for {target_users}",
                "behavioral_insights": f"Behavioral analysis results"
            }
        
        @self.agent.tool
        async def identify_stakeholders(
            ctx: RunContext[AgentDependencies],
            project_scope: str
        ) -> List[Dict[str, Any]]:
            """Identify project stakeholders and their interests."""
            logger.info(f"Identifying stakeholders for: {project_scope}")
            
            # This would use actual stakeholder identification methods
            return [
                {
                    "name": "End Users",
                    "type": "primary",
                    "interest": "Usable and valuable product",
                    "influence": "high"
                },
                {
                    "name": "Product Owner",
                    "type": "primary", 
                    "interest": "Business success",
                    "influence": "high"
                }
            ]
    
    def _estimate_complexity(self, task: str) -> str:
        """Estimate project complexity based on task description."""
        task_lower = task.lower()
        
        # Simple heuristics for complexity estimation
        complex_indicators = [
            "enterprise", "scalable", "distributed", "microservices",
            "machine learning", "ai", "big data", "real-time"
        ]
        
        simple_indicators = [
            "simple", "basic", "crud", "website", "blog", "portfolio"
        ]
        
        complex_count = sum(1 for indicator in complex_indicators if indicator in task_lower)
        simple_count = sum(1 for indicator in simple_indicators if indicator in task_lower)
        
        if complex_count > simple_count:
            return "high"
        elif simple_count > 0:
            return "low"
        else:
            return "medium"
    
    def _infer_requirements(self, task: str) -> List[str]:
        """Infer basic requirements from task description."""
        requirements = []
        task_lower = task.lower()
        
        # Common requirement patterns
        if "user" in task_lower or "customer" in task_lower:
            requirements.append("User management and authentication")
        
        if "data" in task_lower or "database" in task_lower:
            requirements.append("Data storage and management")
        
        if "web" in task_lower or "website" in task_lower:
            requirements.append("Web interface")
        
        if "mobile" in task_lower or "app" in task_lower:
            requirements.append("Mobile compatibility")
        
        if "api" in task_lower or "integration" in task_lower:
            requirements.append("API and integration capabilities")
        
        return requirements if requirements else ["Core functionality based on project description"]
    
    def _extract_primary_objective(self, task: str) -> str:
        """Extract primary objective from task description."""
        # Simple extraction - would be more sophisticated in practice
        if "build" in task.lower():
            return f"Build {task.split('build', 1)[1].strip()}"
        elif "create" in task.lower():
            return f"Create {task.split('create', 1)[1].strip()}"
        elif "develop" in task.lower():
            return f"Develop {task.split('develop', 1)[1].strip()}"
        else:
            return f"Implement solution for: {task[:100]}..."