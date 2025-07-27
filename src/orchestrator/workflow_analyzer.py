"""
Workflow Analysis Engine using LLM reasoning.
Analyzes project descriptions to generate appropriate agent sequences with parallel execution opportunities.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import re

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider, OpenAIModel

from ..models.workflow_models import (
    ProjectWorkflow,
    WorkflowNode,
    WorkflowStage,
    AgentType,
    QualityGateType,
    WorkflowTemplate,
    WorkflowStatus
)
from ..config.settings import settings

logger = logging.getLogger(__name__)


WORKFLOW_ANALYSIS_PROMPT = """
You are an expert product development workflow analyzer. Your task is to analyze project descriptions and generate optimal multi-agent workflows.

Given a project description, you must:

1. **Identify Project Complexity**: Determine the scope, technical requirements, and complexity level
2. **Select Required Stages**: Choose which workflow stages are needed from: discovery, specification, design, implementation, validation, deployment
3. **Determine Dependencies**: Identify which stages must complete before others can start
4. **Identify Parallel Opportunities**: Find stages that can run simultaneously without conflicts
5. **Estimate Duration**: Provide realistic time estimates for each stage
6. **Set Quality Gates**: Determine appropriate validation requirements

**Project Types & Patterns:**
- **Simple Applications**: Basic CRUD apps, utilities, simple websites
- **Complex Applications**: Multi-service platforms, enterprise software, complex integrations
- **Research Projects**: AI/ML models, data analysis, research implementations  
- **Infrastructure Projects**: DevOps, cloud architecture, system administration
- **Design-Heavy Projects**: UI/UX focused, creative applications, consumer products

**Parallel Execution Rules:**
- Discovery must complete before other stages
- Specification can start once discovery is 50% complete
- Design and technical architecture can run in parallel after specification
- Implementation can have parallel streams for independent components
- Validation can start when implementation reaches 70% completion
- Deployment preparation can start during validation

**Quality Gate Selection:**
- automated: Code quality, tests, basic validation
- manual: Design reviews, stakeholder approval, architectural decisions
- security: Security scans, vulnerability assessment, compliance checks
- performance: Load testing, optimization validation, scalability checks
- compliance: Regulatory requirements, accessibility, standards adherence

Respond with a JSON object containing the workflow analysis.
"""


class WorkflowAnalyzer:
    """
    Analyzes project descriptions using LLM reasoning to generate optimal workflows.
    """
    
    def __init__(self):
        """Initialize the workflow analyzer."""
        self.llm_model = self._create_llm_model()
        self.analysis_agent = self._create_analysis_agent()
        self.templates = self._load_workflow_templates()
        
    def _create_llm_model(self) -> OpenAIModel:
        """Create LLM model for workflow analysis."""
        provider = OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key
        )
        return OpenAIModel(settings.llm_model, provider=provider)
    
    def _create_analysis_agent(self) -> Agent:
        """Create the workflow analysis agent."""
        return Agent(
            self.llm_model,
            system_prompt=WORKFLOW_ANALYSIS_PROMPT
        )
    
    def _load_workflow_templates(self) -> Dict[str, WorkflowTemplate]:
        """Load predefined workflow templates."""
        templates = {}
        
        # Simple web application template
        templates["simple_web_app"] = WorkflowTemplate(
            template_id="simple_web_app",
            name="Simple Web Application",
            description="Template for basic web applications with standard CRUD functionality",
            project_types=["web app", "crud app", "simple application"],
            node_templates=[
                WorkflowNode(
                    id="discovery",
                    stage=WorkflowStage.DISCOVERY,
                    agent_type=AgentType.DISCOVERY,
                    estimated_duration_minutes=120,
                    quality_gates=[QualityGateType.MANUAL]
                ),
                WorkflowNode(
                    id="specification",
                    stage=WorkflowStage.SPECIFICATION,
                    agent_type=AgentType.SPECIFICATION,
                    dependencies=["discovery"],
                    estimated_duration_minutes=180,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.MANUAL]
                ),
                WorkflowNode(
                    id="design",
                    stage=WorkflowStage.DESIGN,
                    agent_type=AgentType.DESIGN,
                    dependencies=["specification"],
                    parallel_eligible=True,
                    estimated_duration_minutes=240,
                    quality_gates=[QualityGateType.MANUAL]
                ),
                WorkflowNode(
                    id="implementation",
                    stage=WorkflowStage.IMPLEMENTATION,
                    agent_type=AgentType.IMPLEMENTATION,
                    dependencies=["specification"],
                    parallel_eligible=True,
                    estimated_duration_minutes=480,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.SECURITY]
                ),
                WorkflowNode(
                    id="validation",
                    stage=WorkflowStage.VALIDATION,
                    agent_type=AgentType.VALIDATION,
                    dependencies=["design", "implementation"],
                    estimated_duration_minutes=120,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.PERFORMANCE]
                ),
                WorkflowNode(
                    id="deployment",
                    stage=WorkflowStage.DEPLOYMENT,
                    agent_type=AgentType.DEPLOYMENT,
                    dependencies=["validation"],
                    estimated_duration_minutes=90,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.MANUAL]
                )
            ]
        )
        
        # Complex enterprise application template
        templates["enterprise_app"] = WorkflowTemplate(
            template_id="enterprise_app",
            name="Enterprise Application",
            description="Template for complex enterprise applications with microservices",
            project_types=["enterprise app", "microservices", "complex platform"],
            node_templates=[
                WorkflowNode(
                    id="discovery",
                    stage=WorkflowStage.DISCOVERY,
                    agent_type=AgentType.DISCOVERY,
                    estimated_duration_minutes=300,
                    quality_gates=[QualityGateType.MANUAL, QualityGateType.COMPLIANCE]
                ),
                WorkflowNode(
                    id="specification_architecture",
                    stage=WorkflowStage.SPECIFICATION,
                    agent_type=AgentType.SPECIFICATION,
                    dependencies=["discovery"],
                    estimated_duration_minutes=360,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.MANUAL]
                ),
                WorkflowNode(
                    id="specification_services",
                    stage=WorkflowStage.SPECIFICATION,
                    agent_type=AgentType.SPECIFICATION,
                    dependencies=["discovery"],
                    parallel_eligible=True,
                    estimated_duration_minutes=240,
                    quality_gates=[QualityGateType.AUTOMATED]
                ),
                WorkflowNode(
                    id="design_ux",
                    stage=WorkflowStage.DESIGN,
                    agent_type=AgentType.DESIGN,
                    dependencies=["specification_architecture"],
                    parallel_eligible=True,
                    estimated_duration_minutes=480,
                    quality_gates=[QualityGateType.MANUAL]
                ),
                WorkflowNode(
                    id="design_system",
                    stage=WorkflowStage.DESIGN,
                    agent_type=AgentType.DESIGN,
                    dependencies=["specification_architecture"],
                    parallel_eligible=True,
                    estimated_duration_minutes=360,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.MANUAL]
                ),
                WorkflowNode(
                    id="implementation_backend",
                    stage=WorkflowStage.IMPLEMENTATION,
                    agent_type=AgentType.IMPLEMENTATION,
                    dependencies=["specification_architecture", "specification_services"],
                    parallel_eligible=True,
                    estimated_duration_minutes=720,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.SECURITY]
                ),
                WorkflowNode(
                    id="implementation_frontend",
                    stage=WorkflowStage.IMPLEMENTATION,
                    agent_type=AgentType.IMPLEMENTATION,
                    dependencies=["design_ux", "design_system"],
                    parallel_eligible=True,
                    estimated_duration_minutes=600,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.SECURITY]
                ),
                WorkflowNode(
                    id="validation_integration",
                    stage=WorkflowStage.VALIDATION,
                    agent_type=AgentType.VALIDATION,
                    dependencies=["implementation_backend", "implementation_frontend"],
                    estimated_duration_minutes=240,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.PERFORMANCE]
                ),
                WorkflowNode(
                    id="validation_security",
                    stage=WorkflowStage.VALIDATION,
                    agent_type=AgentType.VALIDATION,
                    dependencies=["implementation_backend", "implementation_frontend"],
                    parallel_eligible=True,
                    estimated_duration_minutes=180,
                    quality_gates=[QualityGateType.SECURITY, QualityGateType.COMPLIANCE]
                ),
                WorkflowNode(
                    id="deployment",
                    stage=WorkflowStage.DEPLOYMENT,
                    agent_type=AgentType.DEPLOYMENT,
                    dependencies=["validation_integration", "validation_security"],
                    estimated_duration_minutes=180,
                    quality_gates=[QualityGateType.AUTOMATED, QualityGateType.MANUAL, QualityGateType.COMPLIANCE]
                )
            ]
        )
        
        return templates
    
    async def analyze_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> ProjectWorkflow:
        """
        Analyze a project description and generate an optimal workflow.
        
        Args:
            description: Project description
            project_name: Optional project name
            constraints: Optional constraints (timeline, resources, etc.)
            preferences: Optional preferences (parallel execution, quality gates, etc.)
            
        Returns:
            Generated project workflow
        """
        logger.info(f"Analyzing project description: {description[:100]}...")
        
        # First try template matching
        template_workflow = await self._try_template_matching(description, project_name)
        if template_workflow:
            logger.info(f"Matched project to template: {template_workflow.name}")
            return template_workflow
        
        # Fall back to LLM analysis
        return await self._analyze_with_llm(description, project_name, constraints, preferences)
    
    async def _try_template_matching(
        self,
        description: str,
        project_name: Optional[str] = None
    ) -> Optional[ProjectWorkflow]:
        """Try to match project to existing templates."""
        description_lower = description.lower()
        
        # Simple keyword matching for templates
        template_keywords = {
            "simple_web_app": ["simple", "basic", "crud", "web app", "website", "blog"],
            "enterprise_app": ["enterprise", "complex", "microservices", "platform", "scalable", "multi-service"]
        }
        
        best_match = None
        best_score = 0
        
        for template_id, keywords in template_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > best_score:
                best_score = score
                best_match = template_id
        
        # Require at least 30% keyword match
        if best_match and best_score >= len(template_keywords[best_match]) * 0.3:
            template = self.templates[best_match]
            project_id = f"proj_{int(time.time())}"
            
            return template.create_workflow(
                project_id=project_id,
                project_name=project_name or "Generated Project",
                project_description=description
            )
        
        return None
    
    async def _analyze_with_llm(
        self,
        description: str,
        project_name: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> ProjectWorkflow:
        """Analyze project using LLM reasoning."""
        
        # Prepare analysis prompt
        analysis_input = {
            "project_description": description,
            "project_name": project_name,
            "constraints": constraints or {},
            "preferences": preferences or {},
            "available_stages": [stage.value for stage in WorkflowStage],
            "available_agents": [agent.value for agent in AgentType],
            "quality_gate_types": [gate.value for gate in QualityGateType]
        }
        
        prompt = f"""
        Analyze this project and generate a workflow:
        
        {json.dumps(analysis_input, indent=2)}
        
        Respond with a JSON object in this exact format:
        {{
            "project_type": "description of project type",
            "complexity_level": "simple|medium|complex",
            "estimated_total_duration_hours": 0,
            "required_stages": ["discovery", "specification", ...],
            "nodes": [
                {{
                    "id": "unique_node_id",
                    "stage": "workflow_stage",
                    "agent_type": "agent_type",
                    "dependencies": ["node_id1", "node_id2"],
                    "parallel_eligible": true/false,
                    "estimated_duration_minutes": 0,
                    "priority": 1-10,
                    "quality_gates": ["automated", "manual", ...],
                    "resource_requirements": {{"memory_gb": 2, "cpu_cores": 1}},
                    "description": "what this node does"
                }}
            ],
            "parallel_opportunities": ["node_id1", "node_id2"],
            "critical_path": ["node_id1", "node_id2", ...],
            "risk_factors": ["factor1", "factor2", ...],
            "recommendations": ["recommendation1", "recommendation2", ...]
        }}
        """
        
        try:
            # Run LLM analysis
            result = await self.analysis_agent.run(prompt)
            
            # Parse LLM response
            analysis_data = json.loads(result.data)
            
            # Create workflow from analysis
            return await self._create_workflow_from_analysis(
                description, project_name, analysis_data
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Fallback to simple template
            return await self._create_fallback_workflow(description, project_name)
    
    async def _create_workflow_from_analysis(
        self,
        description: str,
        project_name: Optional[str],
        analysis_data: Dict[str, Any]
    ) -> ProjectWorkflow:
        """Create workflow from LLM analysis results."""
        
        project_id = f"proj_{int(time.time())}"
        
        # Create workflow nodes from analysis
        nodes = []
        for node_data in analysis_data.get("nodes", []):
            try:
                # Parse quality gates
                quality_gates = []
                for gate_str in node_data.get("quality_gates", []):
                    try:
                        quality_gates.append(QualityGateType(gate_str))
                    except ValueError:
                        logger.warning(f"Unknown quality gate type: {gate_str}")
                
                node = WorkflowNode(
                    id=node_data["id"],
                    stage=WorkflowStage(node_data["stage"]),
                    agent_type=AgentType(node_data["agent_type"]),
                    dependencies=node_data.get("dependencies", []),
                    parallel_eligible=node_data.get("parallel_eligible", False),
                    estimated_duration_minutes=node_data.get("estimated_duration_minutes", 60),
                    priority=node_data.get("priority", 1),
                    quality_gates=quality_gates,
                    resource_requirements=node_data.get("resource_requirements", {})
                )
                nodes.append(node)
                
            except Exception as e:
                logger.error(f"Failed to create node from analysis data: {e}")
                continue
        
        # Validate workflow has required nodes
        if not nodes:
            logger.warning("No valid nodes created from analysis, using fallback")
            return await self._create_fallback_workflow(description, project_name)
        
        # Calculate total estimated duration
        total_duration = sum(node.estimated_duration_minutes for node in nodes)
        
        workflow = ProjectWorkflow(
            project_id=project_id,
            name=project_name or "Generated Project",
            description=description,
            nodes=nodes,
            status=WorkflowStatus.INITIALIZED,
            total_estimated_duration_minutes=total_duration,
            tags=[analysis_data.get("project_type", "unknown")],
            external_refs={
                "complexity_level": analysis_data.get("complexity_level", "medium"),
                "parallel_opportunities": analysis_data.get("parallel_opportunities", []),
                "critical_path": analysis_data.get("critical_path", []),
                "risk_factors": analysis_data.get("risk_factors", []),
                "recommendations": analysis_data.get("recommendations", [])
            }
        )
        
        logger.info(f"Created workflow {project_id} with {len(nodes)} nodes")
        return workflow
    
    async def _create_fallback_workflow(
        self,
        description: str,
        project_name: Optional[str]
    ) -> ProjectWorkflow:
        """Create a basic fallback workflow when analysis fails."""
        
        project_id = f"proj_{int(time.time())}"
        
        # Basic workflow with all stages
        nodes = [
            WorkflowNode(
                id="discovery",
                stage=WorkflowStage.DISCOVERY,
                agent_type=AgentType.DISCOVERY,
                estimated_duration_minutes=120,
                quality_gates=[QualityGateType.MANUAL]
            ),
            WorkflowNode(
                id="specification",
                stage=WorkflowStage.SPECIFICATION,
                agent_type=AgentType.SPECIFICATION,
                dependencies=["discovery"],
                estimated_duration_minutes=180,
                quality_gates=[QualityGateType.AUTOMATED]
            ),
            WorkflowNode(
                id="design",
                stage=WorkflowStage.DESIGN,
                agent_type=AgentType.DESIGN,
                dependencies=["specification"],
                estimated_duration_minutes=240,
                quality_gates=[QualityGateType.MANUAL]
            ),
            WorkflowNode(
                id="implementation",
                stage=WorkflowStage.IMPLEMENTATION,
                agent_type=AgentType.IMPLEMENTATION,
                dependencies=["specification", "design"],
                estimated_duration_minutes=480,
                quality_gates=[QualityGateType.AUTOMATED, QualityGateType.SECURITY]
            ),
            WorkflowNode(
                id="validation",
                stage=WorkflowStage.VALIDATION,
                agent_type=AgentType.VALIDATION,
                dependencies=["implementation"],
                estimated_duration_minutes=120,
                quality_gates=[QualityGateType.AUTOMATED]
            ),
            WorkflowNode(
                id="deployment",
                stage=WorkflowStage.DEPLOYMENT,
                agent_type=AgentType.DEPLOYMENT,
                dependencies=["validation"],
                estimated_duration_minutes=90,
                quality_gates=[QualityGateType.MANUAL]
            )
        ]
        
        return ProjectWorkflow(
            project_id=project_id,
            name=project_name or "Fallback Project",
            description=description,
            nodes=nodes,
            status=WorkflowStatus.INITIALIZED,
            total_estimated_duration_minutes=sum(node.estimated_duration_minutes for node in nodes),
            tags=["fallback"],
            external_refs={"analysis_method": "fallback"}
        )
    
    async def optimize_workflow(self, workflow: ProjectWorkflow) -> ProjectWorkflow:
        """
        Optimize an existing workflow for better performance and parallel execution.
        
        Args:
            workflow: Existing workflow to optimize
            
        Returns:
            Optimized workflow
        """
        logger.info(f"Optimizing workflow {workflow.project_id}")
        
        # Find additional parallel opportunities
        optimized_nodes = []
        for node in workflow.nodes:
            optimized_node = node.copy()
            
            # Check if node can be made parallel
            if not node.parallel_eligible:
                can_be_parallel = await self._can_run_parallel(node, workflow.nodes)
                optimized_node.parallel_eligible = can_be_parallel
            
            optimized_nodes.append(optimized_node)
        
        # Update workflow
        workflow.nodes = optimized_nodes
        workflow.updated_at = datetime.now()
        
        # Recalculate critical path
        critical_path = workflow.get_critical_path()
        workflow.external_refs["optimized_critical_path"] = [node.id for node in critical_path]
        
        logger.info(f"Optimized workflow {workflow.project_id}")
        return workflow
    
    async def _can_run_parallel(self, node: WorkflowNode, all_nodes: List[WorkflowNode]) -> bool:
        """Check if a node can run in parallel with others."""
        
        # Get nodes at the same stage
        same_stage_nodes = [n for n in all_nodes if n.stage == node.stage and n.id != node.id]
        
        # Check for resource conflicts
        for other_node in same_stage_nodes:
            if self._has_resource_conflict(node, other_node):
                return False
        
        # Check dependencies - node can be parallel if it doesn't block other nodes
        dependent_nodes = [n for n in all_nodes if node.id in n.dependencies]
        
        # If many nodes depend on this one, it shouldn't be parallel
        if len(dependent_nodes) > 2:
            return False
        
        return True
    
    def _has_resource_conflict(self, node1: WorkflowNode, node2: WorkflowNode) -> bool:
        """Check if two nodes have conflicting resource requirements."""
        
        req1 = node1.resource_requirements
        req2 = node2.resource_requirements
        
        # Simple check - if both require significant resources, they conflict
        cpu1 = req1.get("cpu_cores", 1)
        cpu2 = req2.get("cpu_cores", 1)
        memory1 = req1.get("memory_gb", 1)
        memory2 = req2.get("memory_gb", 1)
        
        # Conflict if combined resources exceed system limits
        max_cpu = settings.max_parallel_agents * 2  # Rough estimate
        max_memory = settings.max_parallel_agents * 4  # GB
        
        return (cpu1 + cpu2 > max_cpu) or (memory1 + memory2 > max_memory)
    
    def get_workflow_summary(self, workflow: ProjectWorkflow) -> Dict[str, Any]:
        """Get a summary of workflow characteristics."""
        
        parallel_nodes = [node for node in workflow.nodes if node.parallel_eligible]
        critical_path = workflow.get_critical_path()
        
        return {
            "project_id": workflow.project_id,
            "total_nodes": len(workflow.nodes),
            "parallel_nodes": len(parallel_nodes),
            "estimated_duration_hours": workflow.total_estimated_duration_minutes / 60,
            "critical_path_length": len(critical_path),
            "stages_covered": list(set(node.stage for node in workflow.nodes)),
            "quality_gates_count": sum(len(node.quality_gates) for node in workflow.nodes),
            "complexity_indicators": {
                "has_parallel_execution": len(parallel_nodes) > 0,
                "multi_stage_implementation": len([n for n in workflow.nodes if n.stage == WorkflowStage.IMPLEMENTATION]) > 1,
                "comprehensive_validation": len([n for n in workflow.nodes if n.stage == WorkflowStage.VALIDATION]) > 1
            }
        }