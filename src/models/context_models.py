"""
Context preservation data models.
Models for maintaining context across agent handoffs with versioning and rollback.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import hashlib
import json

from .workflow_models import WorkflowStage, AgentType


class ContextVersion(BaseModel):
    """Version information for context snapshots."""
    version: int = Field(..., ge=1, description="Version number")
    parent_version: Optional[int] = Field(None, description="Parent version number")
    checksum: str = Field(..., description="Data integrity checksum")
    size_bytes: int = Field(..., ge=0, description="Context size in bytes")
    created_at: datetime = Field(default_factory=datetime.now, description="Version creation time")
    created_by: str = Field(..., description="Agent that created this version")
    
    @classmethod
    def create_checksum(cls, data: Dict[str, Any]) -> str:
        """Create a checksum for context data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class ContextSnapshot(BaseModel):
    """Immutable snapshot of context at a specific point in time."""
    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    project_id: str = Field(..., description="Associated project")
    stage: WorkflowStage = Field(..., description="Workflow stage")
    agent_type: AgentType = Field(..., description="Agent type that created snapshot")
    
    # Core context data
    data: Dict[str, Any] = Field(..., description="Context data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Versioning
    version_info: ContextVersion = Field(..., description="Version information")
    
    # Relationships
    derived_from: Optional[str] = Field(None, description="Parent snapshot ID")
    children: List[str] = Field(default_factory=list, description="Child snapshot IDs")
    
    # Quality metrics
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Context quality score")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="Context completeness")
    drift_detected: bool = Field(False, description="Whether context drift was detected")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    archived: bool = Field(False, description="Whether snapshot is archived")
    
    def __init__(self, **data):
        # Auto-generate version info if not provided
        if 'version_info' not in data and 'data' in data:
            version_data = data.get('version_info', {})
            if 'checksum' not in version_data:
                version_data['checksum'] = ContextVersion.create_checksum(data['data'])
            if 'size_bytes' not in version_data:
                version_data['size_bytes'] = len(json.dumps(data['data']).encode())
            if 'created_by' not in version_data:
                version_data['created_by'] = data.get('agent_type', 'unknown')
            if 'version' not in version_data:
                version_data['version'] = 1
            
            data['version_info'] = ContextVersion(**version_data)
        
        super().__init__(**data)
    
    def validate_integrity(self) -> bool:
        """Validate data integrity using checksum."""
        expected_checksum = ContextVersion.create_checksum(self.data)
        return expected_checksum == self.version_info.checksum
    
    def get_size_mb(self) -> float:
        """Get context size in megabytes."""
        return self.version_info.size_bytes / (1024 * 1024)
    
    def is_expired(self) -> bool:
        """Check if snapshot has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def extract_key_data(self, keys: List[str]) -> Dict[str, Any]:
        """Extract specific keys from context data."""
        return {key: self.data.get(key) for key in keys if key in self.data}


class HandoffArtifact(BaseModel):
    """Artifact passed between agents during handoff."""
    artifact_id: str = Field(..., description="Unique artifact identifier")
    artifact_type: str = Field(..., description="Type of artifact")
    name: str = Field(..., description="Artifact name")
    description: Optional[str] = Field(None, description="Artifact description")
    data: Dict[str, Any] = Field(..., description="Artifact data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Artifact metadata")
    created_by: str = Field(..., description="Creating agent")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    # Quality indicators
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Artifact quality score")
    validation_status: str = Field(default="pending", description="Validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")


class HandoffPackage(BaseModel):
    """Complete package for agent-to-agent handoff."""
    handoff_id: str = Field(..., description="Unique handoff identifier")
    project_id: str = Field(..., description="Associated project")
    
    # Agent information
    source_agent: AgentType = Field(..., description="Source agent type")
    target_agent: AgentType = Field(..., description="Target agent type")
    
    # Context and artifacts
    context: ContextSnapshot = Field(..., description="Preserved context")
    artifacts: List[HandoffArtifact] = Field(default_factory=list, description="Stage deliverables")
    
    # Quality and approval
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall quality score")
    approved: bool = Field(False, description="Manual approval status")
    approved_by: Optional[str] = Field(None, description="Who approved the handoff")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    
    # Execution tracking
    created_at: datetime = Field(default_factory=datetime.now, description="Handoff creation time")
    completed_at: Optional[datetime] = Field(None, description="Handoff completion time")
    status: str = Field(default="pending", description="Handoff status")
    
    # Quality gates
    required_gates: List[str] = Field(default_factory=list, description="Required quality gates")
    passed_gates: List[str] = Field(default_factory=list, description="Passed quality gates")
    failed_gates: List[str] = Field(default_factory=list, description="Failed quality gates")
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score based on context and artifacts."""
        if not self.artifacts:
            return self.context.quality_score
        
        artifact_scores = [artifact.quality_score for artifact in self.artifacts]
        avg_artifact_score = sum(artifact_scores) / len(artifact_scores)
        
        # Weighted average: 60% context, 40% artifacts
        return (self.context.quality_score * 0.6) + (avg_artifact_score * 0.4)
    
    def is_ready_for_handoff(self, quality_threshold: float = 0.8) -> bool:
        """Check if handoff package meets quality requirements."""
        if self.quality_score < quality_threshold:
            return False
        
        # Check that all required gates have passed
        if self.required_gates:
            return all(gate in self.passed_gates for gate in self.required_gates)
        
        return True
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation status."""
        return {
            "quality_score": self.quality_score,
            "ready_for_handoff": self.is_ready_for_handoff(),
            "context_valid": self.context.validate_integrity(),
            "artifacts_count": len(self.artifacts),
            "failed_artifacts": len([a for a in self.artifacts if a.validation_status == "failed"]),
            "passed_gates": len(self.passed_gates),
            "failed_gates": len(self.failed_gates),
            "total_gates": len(self.required_gates)
        }


class ContextDrift(BaseModel):
    """Detection and measurement of context drift."""
    drift_id: str = Field(..., description="Unique drift detection identifier")
    project_id: str = Field(..., description="Associated project")
    source_snapshot_id: str = Field(..., description="Source snapshot")
    target_snapshot_id: str = Field(..., description="Target snapshot")
    
    # Drift metrics
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift magnitude (0=no drift, 1=complete drift)")
    drift_type: str = Field(..., description="Type of drift detected")
    affected_keys: List[str] = Field(default_factory=list, description="Context keys affected by drift")
    
    # Analysis
    detected_at: datetime = Field(default_factory=datetime.now, description="Drift detection time")
    corrective_action: Optional[str] = Field(None, description="Recommended corrective action")
    auto_corrected: bool = Field(False, description="Whether drift was automatically corrected")
    
    # Thresholds
    warning_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Warning threshold")
    critical_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Critical threshold")
    
    def get_severity(self) -> str:
        """Get drift severity level."""
        if self.drift_score >= self.critical_threshold:
            return "critical"
        elif self.drift_score >= self.warning_threshold:
            return "warning"
        else:
            return "normal"
    
    def requires_intervention(self) -> bool:
        """Check if drift requires manual intervention."""
        return self.drift_score >= self.critical_threshold and not self.auto_corrected