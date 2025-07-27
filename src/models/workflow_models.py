"""
Workflow orchestration data models.
Core structures for agent workflows and orchestration.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class WorkflowStage(str, Enum):
    """Product development workflow stages."""
    DISCOVERY = "discovery"
    SPECIFICATION = "specification"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"


class AgentType(str, Enum):
    """Types of specialized agents."""
    DISCOVERY = "discovery"
    SPECIFICATION = "specification"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityGateType(str, Enum):
    """Types of quality gates."""
    AUTOMATED = "automated"
    MANUAL = "manual"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


class WorkflowNode(BaseModel):
    """Individual node in the workflow graph."""
    id: str = Field(..., description="Unique node identifier")
    stage: WorkflowStage = Field(..., description="Workflow stage")
    agent_type: AgentType = Field(..., description="Required agent type")
    dependencies: List[str] = Field(default_factory=list, description="Node dependencies")
    parallel_eligible: bool = Field(False, description="Can run in parallel")
    quality_gates: List[QualityGateType] = Field(default_factory=list, description="Required validations")
    estimated_duration_minutes: int = Field(default=60, ge=1, le=1440, description="Estimated execution time")
    priority: int = Field(default=1, ge=1, le=10, description="Execution priority (1=highest)")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource needs")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "id": "discovery_001",
                "stage": "discovery",
                "agent_type": "discovery",
                "dependencies": [],
                "parallel_eligible": False,
                "quality_gates": ["automated"],
                "estimated_duration_minutes": 120,
                "priority": 1,
                "resource_requirements": {"memory_gb": 2, "cpu_cores": 1}
            }
        }


class ProjectWorkflow(BaseModel):
    """Complete workflow definition for a project."""
    project_id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    nodes: List[WorkflowNode] = Field(..., description="Workflow nodes")
    current_stage: Optional[WorkflowStage] = Field(None, description="Current active stage")
    status: WorkflowStatus = Field(WorkflowStatus.INITIALIZED, description="Workflow status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    # Workflow metadata
    created_by: Optional[str] = Field(None, description="Creator identifier")
    tags: List[str] = Field(default_factory=list, description="Project tags")
    external_refs: Dict[str, str] = Field(default_factory=dict, description="External references")
    
    # Progress tracking
    completed_nodes: List[str] = Field(default_factory=list, description="Completed node IDs")
    failed_nodes: List[str] = Field(default_factory=list, description="Failed node IDs")
    active_nodes: List[str] = Field(default_factory=list, description="Currently active node IDs")
    
    # Resource tracking
    total_estimated_duration_minutes: int = Field(default=0, ge=0, description="Total estimated duration")
    actual_duration_minutes: int = Field(default=0, ge=0, description="Actual execution time")
    
    def get_next_nodes(self) -> List[WorkflowNode]:
        """Get nodes that are ready to execute."""
        next_nodes = []
        for node in self.nodes:
            if node.id not in self.completed_nodes and node.id not in self.failed_nodes:
                # Check if all dependencies are completed
                dependencies_met = all(
                    dep_id in self.completed_nodes for dep_id in node.dependencies
                )
                if dependencies_met:
                    next_nodes.append(node)
        return next_nodes
    
    def get_parallel_nodes(self) -> List[WorkflowNode]:
        """Get nodes that can run in parallel."""
        next_nodes = self.get_next_nodes()
        return [node for node in next_nodes if node.parallel_eligible]
    
    def calculate_progress(self) -> float:
        """Calculate workflow completion percentage."""
        if not self.nodes:
            return 0.0
        return len(self.completed_nodes) / len(self.nodes) * 100.0
    
    def get_critical_path(self) -> List[WorkflowNode]:
        """Calculate critical path through workflow."""
        # Simple implementation - could be enhanced with proper critical path algorithm
        critical_nodes = []
        for node in self.nodes:
            if not node.parallel_eligible:
                critical_nodes.append(node)
        return sorted(critical_nodes, key=lambda x: x.priority)


class WorkflowTemplate(BaseModel):
    """Template for creating workflows for specific project types."""
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    project_types: List[str] = Field(..., description="Applicable project types")
    node_templates: List[WorkflowNode] = Field(..., description="Template nodes")
    default_config: Dict[str, Any] = Field(default_factory=dict, description="Default configuration")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    def create_workflow(self, project_id: str, project_name: str, project_description: str) -> ProjectWorkflow:
        """Create a workflow instance from this template."""
        return ProjectWorkflow(
            project_id=project_id,
            name=project_name,
            description=project_description,
            nodes=self.node_templates.copy(),
            external_refs=self.default_config.copy()
        )


class WorkflowEvent(BaseModel):
    """Event in workflow execution history."""
    event_id: str = Field(..., description="Unique event identifier")
    project_id: str = Field(..., description="Associated project")
    event_type: str = Field(..., description="Type of event")
    node_id: Optional[str] = Field(None, description="Associated node")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    actor: Optional[str] = Field(None, description="Who/what triggered the event")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "event_id": "evt_001",
                "project_id": "proj_001",
                "event_type": "node_started",
                "node_id": "discovery_001",
                "data": {"agent_type": "discovery", "start_time": "2024-01-01T10:00:00Z"},
                "actor": "orchestrator"
            }
        }