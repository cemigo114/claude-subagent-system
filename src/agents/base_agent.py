"""
Base Agent Architecture with common functionality.
Provides logging, error handling, context awareness, and handoff capabilities.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from contextlib import asynccontextmanager

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider, OpenAIModel

from ..models.agent_models import AgentDependencies, AgentResult, AgentCapability
from ..models.context_models import ContextSnapshot, HandoffPackage
from ..models.workflow_models import AgentType, WorkflowStage
from ..config.settings import settings

logger = logging.getLogger(__name__)


class BaseSubAgent(ABC):
    """
    Abstract base class for all specialized sub-agents.
    Provides common functionality for logging, error handling, context management, and handoffs.
    """
    
    def __init__(self, agent_type: AgentType):
        """Initialize the base agent."""
        self.agent_type = agent_type
        self.llm_model = self._create_llm_model()
        self.capabilities: List[AgentCapability] = []
        self.tools: Dict[str, Callable] = {}
        self._agent: Optional[Agent] = None
        self._execution_stats = {
            "tasks_completed": 0,
            "total_execution_time": 0.0,
            "total_tokens_used": 0,
            "errors_encountered": 0
        }
    
    def _create_llm_model(self) -> OpenAIModel:
        """Create LLM model for the agent."""
        provider = OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key
        )
        return OpenAIModel(settings.llm_model, provider=provider)
    
    @property
    def agent(self) -> Agent:
        """Get the Pydantic AI agent instance."""
        if self._agent is None:
            self._agent = Agent(
                self.llm_model,
                deps_type=AgentDependencies,
                system_prompt=self.get_system_prompt()
            )
            self._register_tools()
        return self._agent
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent type."""
        pass
    
    @abstractmethod
    def get_required_tools(self) -> List[str]:
        """Get list of required tool names for this agent."""
        pass
    
    @abstractmethod
    async def validate_input(self, task: str, context: Optional[ContextSnapshot] = None) -> bool:
        """Validate input for the agent's task."""
        pass
    
    @abstractmethod
    async def process_task(
        self,
        task: str,
        context: Optional[ContextSnapshot] = None,
        dependencies: Optional[AgentDependencies] = None
    ) -> Dict[str, Any]:
        """Process the main task for this agent type."""
        pass
    
    def _register_tools(self):
        """Register tools for this agent."""
        # Register common tools
        self._register_common_tools()
        
        # Register agent-specific tools
        self._register_agent_tools()
    
    def _register_common_tools(self):
        """Register common tools available to all agents."""
        
        @self.agent.tool
        async def log_info(ctx: RunContext[AgentDependencies], message: str) -> str:
            """Log an informational message."""
            logger.info(f"[{self.agent_type}] {message}")
            return f"Logged: {message}"
        
        @self.agent.tool
        async def get_context_data(
            ctx: RunContext[AgentDependencies], 
            key: str
        ) -> Optional[Any]:
            """Get data from the current context."""
            # This would integrate with context manager
            logger.debug(f"Retrieving context data for key: {key}")
            return ctx.deps.__dict__.get(key)
        
        @self.agent.tool
        async def update_progress(
            ctx: RunContext[AgentDependencies], 
            progress_percentage: int,
            status_message: str
        ) -> str:
            """Update task progress."""
            logger.info(f"[{self.agent_type}] Progress: {progress_percentage}% - {status_message}")
            return f"Progress updated: {progress_percentage}%"
    
    @abstractmethod
    def _register_agent_tools(self):
        """Register agent-specific tools."""
        pass
    
    async def execute(
        self,
        task: str,
        context: Optional[ContextSnapshot] = None,
        dependencies: Optional[AgentDependencies] = None
    ) -> AgentResult:
        """
        Execute the agent with proper error handling and logging.
        
        Args:
            task: Task description
            context: Optional context snapshot
            dependencies: Agent dependencies
            
        Returns:
            Agent execution result
        """
        start_time = time.time()
        result_id = f"{self.agent_type}_{int(start_time)}"
        
        logger.info(f"Starting {self.agent_type} execution: {task[:100]}...")
        
        try:
            # Validate input
            if not await self.validate_input(task, context):
                raise ValueError("Input validation failed")
            
            # Process the task
            result_data = await self.process_task(task, context, dependencies)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update stats
            self._execution_stats["tasks_completed"] += 1
            self._execution_stats["total_execution_time"] += execution_time
            
            # Create result
            result = AgentResult(
                result_id=result_id,
                agent_instance_id=f"{self.agent_type}_instance",
                agent_type=self.agent_type,
                success=True,
                data=result_data,
                execution_time_seconds=execution_time,
                quality_score=await self._calculate_quality_score(result_data),
                completeness_score=await self._calculate_completeness_score(result_data)
            )
            
            logger.info(f"{self.agent_type} completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._execution_stats["errors_encountered"] += 1
            
            logger.error(f"{self.agent_type} execution failed: {e}")
            
            return AgentResult(
                result_id=result_id,
                agent_instance_id=f"{self.agent_type}_instance",
                agent_type=self.agent_type,
                success=False,
                error=str(e),
                execution_time_seconds=execution_time
            )
    
    async def create_handoff_package(
        self,
        target_agent: AgentType,
        context: ContextSnapshot,
        result: AgentResult
    ) -> HandoffPackage:
        """
        Create a handoff package for transitioning to another agent.
        
        Args:
            target_agent: Target agent type
            context: Current context snapshot
            result: Execution result
            
        Returns:
            Handoff package
        """
        from ..models.context_models import HandoffArtifact
        
        # Create artifacts from result
        artifacts = []
        if result.success and result.data:
            for key, value in result.data.items():
                artifact = HandoffArtifact(
                    artifact_id=f"{result.result_id}_{key}",
                    artifact_type=key,
                    name=f"{self.agent_type} {key}",
                    description=f"Output from {self.agent_type} agent",
                    data={"content": value},
                    created_by=str(self.agent_type),
                    quality_score=result.quality_score
                )
                artifacts.append(artifact)
        
        # Create handoff package
        handoff_package = HandoffPackage(
            handoff_id=f"handoff_{self.agent_type}_{target_agent}_{int(time.time())}",
            project_id=context.project_id,
            source_agent=self.agent_type,
            target_agent=target_agent,
            context=context,
            artifacts=artifacts,
            quality_score=result.quality_score
        )
        
        logger.info(f"Created handoff package from {self.agent_type} to {target_agent}")
        return handoff_package
    
    async def receive_handoff(self, handoff_package: HandoffPackage) -> bool:
        """
        Receive and process a handoff package from another agent.
        
        Args:
            handoff_package: Handoff package from source agent
            
        Returns:
            Success status
        """
        logger.info(f"Receiving handoff from {handoff_package.source_agent} to {self.agent_type}")
        
        try:
            # Validate handoff package
            if not handoff_package.is_ready_for_handoff():
                logger.error("Handoff package not ready")
                return False
            
            # Process artifacts
            for artifact in handoff_package.artifacts:
                await self._process_handoff_artifact(artifact)
            
            logger.info(f"Successfully received handoff from {handoff_package.source_agent}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to receive handoff: {e}")
            return False
    
    async def _process_handoff_artifact(self, artifact):
        """Process an individual handoff artifact."""
        logger.debug(f"Processing artifact: {artifact.artifact_type}")
        # Agent-specific artifact processing would be implemented in subclasses
    
    async def _calculate_quality_score(self, result_data: Dict[str, Any]) -> float:
        """Calculate quality score for result data."""
        # Basic quality scoring - can be overridden in subclasses
        if not result_data:
            return 0.0
        
        score = 0.0
        
        # Check data completeness
        non_empty_values = sum(1 for v in result_data.values() if v)
        total_values = len(result_data)
        
        if total_values > 0:
            score += (non_empty_values / total_values) * 0.5
        
        # Check data structure quality
        structured_data = sum(1 for v in result_data.values() 
                            if isinstance(v, (dict, list)) and len(v) > 0)
        
        if total_values > 0:
            score += (structured_data / total_values) * 0.3
        
        # Base score for having any data
        score += 0.2
        
        return min(score, 1.0)
    
    async def _calculate_completeness_score(self, result_data: Dict[str, Any]) -> float:
        """Calculate completeness score for result data."""
        # Agent-specific completeness requirements
        required_keys = self._get_required_output_keys()
        
        if not required_keys:
            return 1.0
        
        present_keys = sum(1 for key in required_keys if key in result_data and result_data[key])
        return present_keys / len(required_keys)
    
    def _get_required_output_keys(self) -> List[str]:
        """Get required output keys for this agent type."""
        # Override in subclasses to specify required outputs
        return []
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        return self.capabilities
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self._execution_stats.copy()
        
        if stats["tasks_completed"] > 0:
            stats["average_execution_time"] = stats["total_execution_time"] / stats["tasks_completed"]
            stats["success_rate"] = 1.0 - (stats["errors_encountered"] / stats["tasks_completed"])
        else:
            stats["average_execution_time"] = 0.0
            stats["success_rate"] = 0.0
        
        return stats
    
    @asynccontextmanager
    async def execution_context(self, task: str):
        """Context manager for agent execution with automatic cleanup."""
        logger.debug(f"Entering execution context for {self.agent_type}")
        
        try:
            yield self
        except Exception as e:
            logger.error(f"Error in execution context: {e}")
            raise
        finally:
            logger.debug(f"Exiting execution context for {self.agent_type}")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(type={self.agent_type}, tasks_completed={self._execution_stats['tasks_completed']})"