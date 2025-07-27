"""
Agent communication and state management models.
Models for inter-agent communication and agent lifecycle management.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

from .workflow_models import AgentType, WorkflowStage


class AgentState(str, Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class MessageType(str, Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class AgentMessage(BaseModel):
    """Message passed between agents."""
    message_id: str = Field(..., description="Unique message identifier")
    sender_id: str = Field(..., description="Sending agent identifier")
    receiver_id: Optional[str] = Field(None, description="Receiving agent identifier")
    message_type: MessageType = Field(..., description="Type of message")
    
    # Content
    subject: str = Field(..., description="Message subject")
    content: Dict[str, Any] = Field(..., description="Message content")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Message attachments")
    
    # Metadata
    priority: int = Field(default=1, ge=1, le=10, description="Message priority (1=highest)")
    expires_at: Optional[datetime] = Field(None, description="Message expiration")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    sent_at: Optional[datetime] = Field(None, description="Send timestamp")
    received_at: Optional[datetime] = Field(None, description="Receipt timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    
    # Delivery tracking
    delivery_attempts: int = Field(default=0, ge=0, description="Number of delivery attempts")
    max_delivery_attempts: int = Field(default=3, ge=1, le=10, description="Maximum delivery attempts")
    last_error: Optional[str] = Field(None, description="Last delivery error")
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def can_retry_delivery(self) -> bool:
        """Check if message can be retried for delivery."""
        return self.delivery_attempts < self.max_delivery_attempts and not self.is_expired()


class AgentCapability(BaseModel):
    """Agent capability definition."""
    capability_id: str = Field(..., description="Unique capability identifier")
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    input_schema: Dict[str, Any] = Field(..., description="Input data schema")
    output_schema: Dict[str, Any] = Field(..., description="Output data schema")
    
    # Requirements
    required_tools: List[str] = Field(default_factory=list, description="Required tool names")
    required_permissions: List[str] = Field(default_factory=list, description="Required permissions")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    
    # Quality metrics
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Historical success rate")
    average_execution_time: float = Field(default=0.0, ge=0.0, description="Average execution time in seconds")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality score")
    
    # Metadata
    version: str = Field(default="1.0.0", description="Capability version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class AgentInstance(BaseModel):
    """Running agent instance."""
    instance_id: str = Field(..., description="Unique instance identifier")
    agent_type: AgentType = Field(..., description="Type of agent")
    project_id: str = Field(..., description="Associated project")
    workflow_stage: WorkflowStage = Field(..., description="Current workflow stage")
    
    # State
    state: AgentState = Field(AgentState.INITIALIZING, description="Current agent state")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    tools: List[str] = Field(default_factory=list, description="Available tool names")
    permissions: List[str] = Field(default_factory=list, description="Granted permissions")
    
    # Resource tracking
    cpu_usage: float = Field(default=0.0, ge=0.0, description="CPU usage percentage")
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Memory usage in MB")
    token_usage: int = Field(default=0, ge=0, description="LLM token usage")
    
    # Lifecycle timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    
    # Error tracking
    error_count: int = Field(default=0, ge=0, description="Number of errors encountered")
    last_error: Optional[str] = Field(None, description="Last error message")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    
    def is_healthy(self, heartbeat_timeout_seconds: int = 300) -> bool:
        """Check if agent instance is healthy."""
        if self.state in [AgentState.FAILED, AgentState.TERMINATED]:
            return False
        
        if self.last_heartbeat is None:
            return self.state == AgentState.INITIALIZING
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < heartbeat_timeout_seconds
    
    def get_uptime_seconds(self) -> float:
        """Get agent uptime in seconds."""
        if self.started_at is None:
            return 0.0
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.now()


class AgentResult(BaseModel):
    """Result of agent execution."""
    result_id: str = Field(..., description="Unique result identifier")
    agent_instance_id: str = Field(..., description="Associated agent instance")
    agent_type: AgentType = Field(..., description="Type of agent")
    
    # Execution details
    success: bool = Field(..., description="Whether execution was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # Quality metrics
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Result quality score")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="Result completeness")
    
    # Performance metrics
    execution_time_seconds: float = Field(..., ge=0.0, description="Execution time")
    token_usage: int = Field(default=0, ge=0, description="LLM tokens used")
    tool_calls: List[str] = Field(default_factory=list, description="Tools called during execution")
    
    # Validation
    validation_status: str = Field(default="pending", description="Validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Generated artifacts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def is_valid(self) -> bool:
        """Check if result is valid."""
        return self.success and self.validation_status == "passed"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        return {
            "success": self.success,
            "quality_score": self.quality_score,
            "completeness_score": self.completeness_score,
            "execution_time": self.execution_time_seconds,
            "token_usage": self.token_usage,
            "tool_calls_count": len(self.tool_calls),
            "artifacts_count": len(self.artifacts),
            "validation_status": self.validation_status,
            "has_errors": bool(self.validation_errors),
            "has_warnings": bool(self.validation_warnings)
        }


class AgentDependencies(BaseModel):
    """Dependencies for agent initialization."""
    project_id: str = Field(..., description="Associated project ID")
    workflow_stage: WorkflowStage = Field(..., description="Workflow stage")
    
    # External tool configurations
    external_tools: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="External tool configs")
    
    # Authentication and permissions
    permissions: List[str] = Field(default_factory=list, description="Granted permissions")
    api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for external services")
    
    # Resource limits
    max_execution_time: int = Field(default=1800, ge=60, le=7200, description="Max execution time in seconds")
    max_token_usage: int = Field(default=10000, ge=100, description="Max LLM tokens")
    max_memory_mb: int = Field(default=1024, ge=100, description="Max memory usage in MB")
    
    # Session information
    session_id: Optional[str] = Field(None, description="Session identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")
    
    class Config:
        """Pydantic configuration."""
        # Exclude sensitive data from serialization by default
        fields = {
            'api_keys': {'exclude': True}
        }