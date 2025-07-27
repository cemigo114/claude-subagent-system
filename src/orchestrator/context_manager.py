"""
Context Management System with persistent storage and versioning.
Handles context preservation across agent handoffs with drift detection and rollback capabilities.
"""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib
import pickle
from dataclasses import dataclass

from ..models.context_models import (
    ContextSnapshot,
    ContextVersion,
    ContextDrift,
    HandoffPackage
)
from ..models.workflow_models import WorkflowStage, AgentType
from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ContextSearchQuery:
    """Query parameters for context search."""
    project_id: Optional[str] = None
    stage: Optional[WorkflowStage] = None
    agent_type: Optional[AgentType] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_quality_score: Optional[float] = None
    limit: int = 100


class ContextStorageBackend:
    """Abstract base for context storage backends."""
    
    async def store_snapshot(self, snapshot: ContextSnapshot) -> bool:
        """Store a context snapshot."""
        raise NotImplementedError
    
    async def retrieve_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Retrieve a context snapshot by ID."""
        raise NotImplementedError
    
    async def list_snapshots(self, query: ContextSearchQuery) -> List[ContextSnapshot]:
        """List snapshots matching query criteria."""
        raise NotImplementedError
    
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a context snapshot."""
        raise NotImplementedError
    
    async def cleanup_expired(self, retention_days: int) -> int:
        """Clean up expired snapshots."""
        raise NotImplementedError


class SQLiteContextStorage(ContextStorageBackend):
    """SQLite-based context storage implementation."""
    
    def __init__(self, db_path: str = "context.db"):
        """Initialize SQLite storage."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    metadata_json TEXT,
                    version INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    completeness_score REAL DEFAULT 0.0,
                    drift_detected BOOLEAN DEFAULT FALSE,
                    derived_from TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    archived BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (derived_from) REFERENCES context_snapshots(snapshot_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project_stage 
                ON context_snapshots(project_id, stage)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON context_snapshots(created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_score 
                ON context_snapshots(quality_score)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    async def store_snapshot(self, snapshot: ContextSnapshot) -> bool:
        """Store a context snapshot in SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO context_snapshots (
                        snapshot_id, project_id, stage, agent_type, data_json, metadata_json,
                        version, checksum, size_bytes, quality_score, completeness_score,
                        drift_detected, derived_from, created_at, expires_at, archived
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.snapshot_id,
                    snapshot.project_id,
                    snapshot.stage,
                    snapshot.agent_type,
                    json.dumps(snapshot.data),
                    json.dumps(snapshot.metadata),
                    snapshot.version_info.version,
                    snapshot.version_info.checksum,
                    snapshot.version_info.size_bytes,
                    snapshot.quality_score,
                    snapshot.completeness_score,
                    snapshot.drift_detected,
                    snapshot.derived_from,
                    snapshot.created_at.isoformat(),
                    snapshot.expires_at.isoformat() if snapshot.expires_at else None,
                    snapshot.archived
                ))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to store context snapshot {snapshot.snapshot_id}: {e}")
            return False
    
    async def retrieve_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Retrieve a context snapshot from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute("""
                    SELECT * FROM context_snapshots WHERE snapshot_id = ?
                """, (snapshot_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_snapshot(row)
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to retrieve context snapshot {snapshot_id}: {e}")
            return None
    
    async def list_snapshots(self, query: ContextSearchQuery) -> List[ContextSnapshot]:
        """List snapshots matching query criteria."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                where_clauses = []
                params = []
                
                if query.project_id:
                    where_clauses.append("project_id = ?")
                    params.append(query.project_id)
                
                if query.stage:
                    where_clauses.append("stage = ?")
                    params.append(query.stage)
                
                if query.agent_type:
                    where_clauses.append("agent_type = ?")
                    params.append(query.agent_type)
                
                if query.created_after:
                    where_clauses.append("created_at >= ?")
                    params.append(query.created_after.isoformat())
                
                if query.created_before:
                    where_clauses.append("created_at <= ?")
                    params.append(query.created_before.isoformat())
                
                if query.min_quality_score:
                    where_clauses.append("quality_score >= ?")
                    params.append(query.min_quality_score)
                
                where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                
                sql = f"""
                    SELECT * FROM context_snapshots {where_clause}
                    ORDER BY created_at DESC LIMIT ?
                """
                params.append(query.limit)
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                return [self._row_to_snapshot(row) for row in rows]
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to list context snapshots: {e}")
            return []
    
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a context snapshot."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                    DELETE FROM context_snapshots WHERE snapshot_id = ?
                """, (snapshot_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to delete context snapshot {snapshot_id}: {e}")
            return False
    
    async def cleanup_expired(self, retention_days: int) -> int:
        """Clean up expired snapshots."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                    DELETE FROM context_snapshots 
                    WHERE created_at < ? OR (expires_at IS NOT NULL AND expires_at < ?)
                """, (cutoff_date.isoformat(), datetime.now().isoformat()))
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to cleanup expired snapshots: {e}")
            return 0
    
    def _row_to_snapshot(self, row: sqlite3.Row) -> ContextSnapshot:
        """Convert database row to ContextSnapshot."""
        version_info = ContextVersion(
            version=row['version'],
            checksum=row['checksum'],
            size_bytes=row['size_bytes'],
            created_by=row['agent_type'],
            created_at=datetime.fromisoformat(row['created_at'])
        )
        
        return ContextSnapshot(
            snapshot_id=row['snapshot_id'],
            project_id=row['project_id'],
            stage=WorkflowStage(row['stage']),
            agent_type=AgentType(row['agent_type']),
            data=json.loads(row['data_json']),
            metadata=json.loads(row['metadata_json']) if row['metadata_json'] else {},
            version_info=version_info,
            derived_from=row['derived_from'],
            quality_score=row['quality_score'],
            completeness_score=row['completeness_score'],
            drift_detected=bool(row['drift_detected']),
            created_at=datetime.fromisoformat(row['created_at']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
            archived=bool(row['archived'])
        )


class ContextManager:
    """
    Context management system with versioning and drift detection.
    Handles persistent storage, rollback, and context preservation across agent handoffs.
    """
    
    def __init__(self, storage_backend: Optional[str] = None):
        """Initialize context manager."""
        self.storage_backend = self._create_storage_backend(storage_backend)
        self.retention_days = settings.context_retention_days
        self.max_size_mb = settings.context_max_size_mb
        self._drift_threshold = 0.3  # Configurable drift detection threshold
        
    def _create_storage_backend(self, backend: Optional[str] = None) -> ContextStorageBackend:
        """Create appropriate storage backend."""
        backend = backend or settings.context_storage_backend
        
        if backend == "local" or backend == "sqlite":
            return SQLiteContextStorage()
        elif backend == "redis":
            # TODO: Implement Redis backend
            raise NotImplementedError("Redis backend not yet implemented")
        elif backend == "postgresql":
            # TODO: Implement PostgreSQL backend  
            raise NotImplementedError("PostgreSQL backend not yet implemented")
        else:
            raise ValueError(f"Unsupported storage backend: {backend}")
    
    async def create_snapshot(
        self,
        project_id: str,
        stage: WorkflowStage,
        agent_type: AgentType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        derived_from: Optional[str] = None
    ) -> ContextSnapshot:
        """
        Create a new context snapshot with versioning.
        
        Args:
            project_id: Project identifier
            stage: Workflow stage
            agent_type: Agent type creating the snapshot
            data: Context data
            metadata: Additional metadata
            derived_from: Parent snapshot ID if applicable
            
        Returns:
            Created context snapshot
        """
        # Validate data size
        data_size_mb = len(json.dumps(data).encode()) / (1024 * 1024)
        if data_size_mb > self.max_size_mb:
            raise ValueError(f"Context data size {data_size_mb:.1f}MB exceeds limit {self.max_size_mb}MB")
        
        # Generate unique snapshot ID
        timestamp = int(datetime.now().timestamp() * 1000)
        snapshot_id = f"ctx_{project_id}_{stage}_{timestamp}"
        
        # Get next version number
        version = await self._get_next_version(project_id, stage)
        
        # Create version info
        version_info = ContextVersion(
            version=version,
            parent_version=await self._get_parent_version(derived_from) if derived_from else None,
            checksum=ContextVersion.create_checksum(data),
            size_bytes=len(json.dumps(data).encode()),
            created_by=agent_type
        )
        
        # Create snapshot
        snapshot = ContextSnapshot(
            snapshot_id=snapshot_id,
            project_id=project_id,
            stage=stage,
            agent_type=agent_type,
            data=data,
            metadata=metadata or {},
            version_info=version_info,
            derived_from=derived_from,
            quality_score=await self._calculate_quality_score(data),
            completeness_score=await self._calculate_completeness_score(data, stage)
        )
        
        # Store snapshot
        success = await self.storage_backend.store_snapshot(snapshot)
        if not success:
            raise RuntimeError(f"Failed to store context snapshot {snapshot_id}")
        
        logger.info(f"Created context snapshot {snapshot_id} for project {project_id}")
        return snapshot
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Retrieve a context snapshot by ID."""
        snapshot = await self.storage_backend.retrieve_snapshot(snapshot_id)
        
        if snapshot and not snapshot.validate_integrity():
            logger.warning(f"Context snapshot {snapshot_id} failed integrity check")
            return None
        
        return snapshot
    
    async def get_latest_snapshot(
        self,
        project_id: str,
        stage: Optional[WorkflowStage] = None,
        agent_type: Optional[AgentType] = None
    ) -> Optional[ContextSnapshot]:
        """Get the most recent snapshot for a project/stage/agent."""
        query = ContextSearchQuery(
            project_id=project_id,
            stage=stage,
            agent_type=agent_type,
            limit=1
        )
        
        snapshots = await self.storage_backend.list_snapshots(query)
        return snapshots[0] if snapshots else None
    
    async def rollback_to_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """
        Rollback to a previous snapshot by creating a new snapshot with the old data.
        
        Args:
            snapshot_id: ID of snapshot to rollback to
            
        Returns:
            New snapshot with rolled-back data
        """
        original_snapshot = await self.get_snapshot(snapshot_id)
        if not original_snapshot:
            logger.error(f"Cannot rollback: snapshot {snapshot_id} not found")
            return None
        
        # Create new snapshot with original data
        rollback_snapshot = await self.create_snapshot(
            project_id=original_snapshot.project_id,
            stage=original_snapshot.stage,
            agent_type=original_snapshot.agent_type,
            data=original_snapshot.data.copy(),
            metadata={
                **original_snapshot.metadata,
                "rollback_from": snapshot_id,
                "rollback_at": datetime.now().isoformat()
            },
            derived_from=original_snapshot.snapshot_id
        )
        
        logger.info(f"Rolled back to snapshot {snapshot_id}, created new snapshot {rollback_snapshot.snapshot_id}")
        return rollback_snapshot
    
    async def detect_context_drift(
        self,
        source_snapshot_id: str,
        target_data: Dict[str, Any]
    ) -> Optional[ContextDrift]:
        """
        Detect context drift between snapshots.
        
        Args:
            source_snapshot_id: ID of source snapshot
            target_data: New context data to compare
            
        Returns:
            ContextDrift object if drift detected, None otherwise
        """
        source_snapshot = await self.get_snapshot(source_snapshot_id)
        if not source_snapshot:
            return None
        
        drift_score = self._calculate_drift_score(source_snapshot.data, target_data)
        
        if drift_score >= self._drift_threshold:
            # Create temporary snapshot for target data to get ID
            temp_snapshot = await self.create_snapshot(
                project_id=source_snapshot.project_id,
                stage=source_snapshot.stage,
                agent_type=source_snapshot.agent_type,
                data=target_data,
                metadata={"temporary": True}
            )
            
            drift = ContextDrift(
                drift_id=f"drift_{source_snapshot_id}_{int(datetime.now().timestamp())}",
                project_id=source_snapshot.project_id,
                source_snapshot_id=source_snapshot_id,
                target_snapshot_id=temp_snapshot.snapshot_id,
                drift_score=drift_score,
                drift_type="data_divergence",
                affected_keys=self._get_changed_keys(source_snapshot.data, target_data)
            )
            
            logger.warning(f"Context drift detected: {drift_score:.2f} for project {source_snapshot.project_id}")
            return drift
        
        return None
    
    async def cleanup_expired_snapshots(self) -> int:
        """Clean up expired context snapshots."""
        count = await self.storage_backend.cleanup_expired(self.retention_days)
        logger.info(f"Cleaned up {count} expired context snapshots")
        return count
    
    async def get_context_history(
        self,
        project_id: str,
        limit: int = 50
    ) -> List[ContextSnapshot]:
        """Get context history for a project."""
        query = ContextSearchQuery(
            project_id=project_id,
            limit=limit
        )
        return await self.storage_backend.list_snapshots(query)
    
    async def _get_next_version(self, project_id: str, stage: WorkflowStage) -> int:
        """Get next version number for a project/stage."""
        query = ContextSearchQuery(
            project_id=project_id,
            stage=stage,
            limit=1
        )
        snapshots = await self.storage_backend.list_snapshots(query)
        
        if not snapshots:
            return 1
        
        return snapshots[0].version_info.version + 1
    
    async def _get_parent_version(self, parent_snapshot_id: str) -> Optional[int]:
        """Get version number of parent snapshot."""
        parent = await self.get_snapshot(parent_snapshot_id)
        return parent.version_info.version if parent else None
    
    async def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate quality score for context data."""
        # Simple quality scoring based on data completeness and structure
        score = 0.0
        
        # Check for required fields
        required_fields = ["requirements", "constraints", "objectives"]
        present_fields = sum(1 for field in required_fields if field in data and data[field])
        score += (present_fields / len(required_fields)) * 0.4
        
        # Check data richness (non-empty values)
        total_values = 0
        non_empty_values = 0
        
        for value in data.values():
            total_values += 1
            if value and (not isinstance(value, (list, dict)) or len(value) > 0):
                non_empty_values += 1
        
        if total_values > 0:
            score += (non_empty_values / total_values) * 0.4
        
        # Check structural consistency
        if isinstance(data, dict) and len(data) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _calculate_completeness_score(self, data: Dict[str, Any], stage: WorkflowStage) -> float:
        """Calculate completeness score based on stage requirements."""
        # Stage-specific completeness requirements
        stage_requirements = {
            WorkflowStage.DISCOVERY: ["market_analysis", "user_requirements", "competitive_analysis"],
            WorkflowStage.SPECIFICATION: ["functional_requirements", "technical_requirements", "architecture"],
            WorkflowStage.DESIGN: ["ui_design", "ux_flow", "design_system"],
            WorkflowStage.IMPLEMENTATION: ["code_structure", "implementation_plan", "dependencies"],
            WorkflowStage.VALIDATION: ["test_results", "quality_metrics", "performance_data"],
            WorkflowStage.DEPLOYMENT: ["deployment_config", "monitoring_setup", "rollback_plan"]
        }
        
        required_keys = stage_requirements.get(stage, [])
        if not required_keys:
            return 1.0  # No specific requirements
        
        present_keys = sum(1 for key in required_keys if key in data and data[key])
        return present_keys / len(required_keys)
    
    def _calculate_drift_score(self, source_data: Dict[str, Any], target_data: Dict[str, Any]) -> float:
        """Calculate drift score between two data sets."""
        all_keys = set(source_data.keys()) | set(target_data.keys())
        if not all_keys:
            return 0.0
        
        changed_keys = 0
        for key in all_keys:
            source_val = source_data.get(key)
            target_val = target_data.get(key)
            
            if source_val != target_val:
                changed_keys += 1
        
        return changed_keys / len(all_keys)
    
    def _get_changed_keys(self, source_data: Dict[str, Any], target_data: Dict[str, Any]) -> List[str]:
        """Get list of keys that changed between datasets."""
        all_keys = set(source_data.keys()) | set(target_data.keys())
        changed_keys = []
        
        for key in all_keys:
            source_val = source_data.get(key)
            target_val = target_data.get(key)
            
            if source_val != target_val:
                changed_keys.append(key)
        
        return changed_keys