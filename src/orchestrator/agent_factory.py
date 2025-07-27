"""
Agent Factory System for dynamic agent generation.
Creates and manages specialized agents following Pydantic AI patterns with lifecycle management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Type, Callable
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager

from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider, OpenAIModel

from ..models.agent_models import (
    AgentInstance, 
    AgentDependencies,
    AgentState,
    AgentCapability,
    AgentResult
)
from ..models.workflow_models import AgentType, WorkflowStage
from ..config.settings import settings

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for agent types and their configurations."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agent_configs: Dict[AgentType, Dict[str, Any]] = {}
        self._agent_classes: Dict[AgentType, Type] = {}
        self._capabilities: Dict[AgentType, List[AgentCapability]] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for all agent types."""
        
        # Discovery Agent Configuration
        self._agent_configs[AgentType.DISCOVERY] = {
            "system_prompt": """
            You are a Discovery Agent specializing in market research and requirement gathering.
            Your role is to analyze market opportunities, gather user requirements, and identify
            project constraints and objectives.
            
            Capabilities:
            - Market research and competitive analysis
            - User interview and survey analysis
            - Requirement elicitation and prioritization
            - Stakeholder identification and analysis
            - Risk and constraint identification
            
            Always provide structured, actionable insights with supporting evidence.
            """,
            "tools": ["web_research", "survey_analysis", "stakeholder_mapping"],
            "required_permissions": ["web_search", "external_data_access"],
            "max_execution_time": 7200,  # 2 hours
            "resource_requirements": {"memory_gb": 2, "cpu_cores": 1}
        }
        
        # Specification Agent Configuration  
        self._agent_configs[AgentType.SPECIFICATION] = {
            "system_prompt": """
            You are a Specification Agent specializing in technical requirements and architecture.
            Your role is to translate business requirements into detailed technical specifications
            and create system architecture designs.
            
            Capabilities:
            - Technical requirement specification
            - System architecture design
            - API and interface definition
            - Database schema design
            - Performance and scalability planning
            
            Always ensure specifications are clear, complete, and technically feasible.
            """,
            "tools": ["architecture_design", "api_modeling", "database_design"],
            "required_permissions": ["technical_analysis", "system_design"],
            "max_execution_time": 10800,  # 3 hours
            "resource_requirements": {"memory_gb": 3, "cpu_cores": 2}
        }
        
        # Design Agent Configuration
        self._agent_configs[AgentType.DESIGN] = {
            "system_prompt": """
            You are a Design Agent specializing in UI/UX design and prototyping.
            Your role is to create user-centered designs, interfaces, and interactive prototypes
            that meet user needs and business objectives.
            
            Capabilities:
            - User experience design
            - User interface design
            - Interactive prototyping
            - Design system creation
            - Accessibility and usability optimization
            
            Always prioritize user experience and accessibility in your designs.
            """,
            "tools": ["figma_integration", "design_system", "prototyping"],
            "required_permissions": ["figma_access", "design_tools"],
            "max_execution_time": 14400,  # 4 hours
            "resource_requirements": {"memory_gb": 4, "cpu_cores": 2}
        }
        
        # Implementation Agent Configuration
        self._agent_configs[AgentType.IMPLEMENTATION] = {
            "system_prompt": """
            You are an Implementation Agent specializing in code generation and development.
            Your role is to write high-quality, secure, and maintainable code that implements
            the specified requirements and designs.
            
            Capabilities:
            - Code generation and development
            - Security compliance implementation
            - Testing and quality assurance
            - Documentation generation
            - Integration and deployment preparation
            
            Always follow security best practices and write clean, well-documented code.
            """,
            "tools": ["code_generation", "security_scanner", "test_generator", "github_integration"],
            "required_permissions": ["code_generation", "security_scanning", "github_access"],
            "max_execution_time": 21600,  # 6 hours
            "resource_requirements": {"memory_gb": 6, "cpu_cores": 4}
        }
        
        # Validation Agent Configuration
        self._agent_configs[AgentType.VALIDATION] = {
            "system_prompt": """
            You are a Validation Agent specializing in QA testing and performance validation.
            Your role is to ensure quality, performance, and reliability of the implemented
            solution through comprehensive testing and validation.
            
            Capabilities:
            - Automated testing and test suite generation
            - Performance testing and optimization
            - Security vulnerability assessment
            - User acceptance testing coordination
            - Quality metrics collection and analysis
            
            Always ensure thorough validation before approving releases.
            """,
            "tools": ["test_automation", "performance_testing", "security_scanner", "quality_metrics"],
            "required_permissions": ["testing_tools", "performance_monitoring", "security_scanning"],
            "max_execution_time": 10800,  # 3 hours
            "resource_requirements": {"memory_gb": 4, "cpu_cores": 3}
        }
        
        # Deployment Agent Configuration
        self._agent_configs[AgentType.DEPLOYMENT] = {
            "system_prompt": """
            You are a Deployment Agent specializing in release coordination and monitoring.
            Your role is to manage deployments, setup monitoring, and ensure smooth
            production operations.
            
            Capabilities:
            - Deployment automation and orchestration
            - Infrastructure provisioning and configuration
            - Monitoring and alerting setup
            - Rollback and disaster recovery planning
            - Production health monitoring
            
            Always ensure reliable, monitorable deployments with proper rollback procedures.
            """,
            "tools": ["deployment_automation", "infrastructure_management", "monitoring_setup"],
            "required_permissions": ["deployment_access", "infrastructure_management", "monitoring_tools"],
            "max_execution_time": 7200,  # 2 hours
            "resource_requirements": {"memory_gb": 3, "cpu_cores": 2}
        }
    
    def register_agent_class(self, agent_type: AgentType, agent_class: Type):
        """Register an agent class for a specific type."""
        self._agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent class for type: {agent_type}")
    
    def get_agent_config(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get configuration for an agent type."""
        return self._agent_configs.get(agent_type, {})
    
    def get_agent_class(self, agent_type: AgentType) -> Optional[Type]:
        """Get registered agent class for a type."""
        return self._agent_classes.get(agent_type)
    
    def update_agent_config(self, agent_type: AgentType, config_updates: Dict[str, Any]):
        """Update configuration for an agent type."""
        if agent_type in self._agent_configs:
            self._agent_configs[agent_type].update(config_updates)
            logger.info(f"Updated configuration for agent type: {agent_type}")


class AgentLifecycleManager:
    """Manages agent lifecycle and resource allocation."""
    
    def __init__(self):
        """Initialize the lifecycle manager."""
        self._active_instances: Dict[str, AgentInstance] = {}
        self._resource_usage = {
            "cpu_cores": 0,
            "memory_gb": 0,
            "active_count": 0
        }
        self._max_resources = {
            "cpu_cores": 16,  # Configurable
            "memory_gb": 32,   # Configurable
            "active_count": settings.max_parallel_agents
        }
    
    async def create_instance(
        self,
        agent_type: AgentType,
        project_id: str,
        workflow_stage: WorkflowStage,
        config: Dict[str, Any],
        dependencies: AgentDependencies
    ) -> AgentInstance:
        """Create a new agent instance."""
        
        # Check resource availability
        required_resources = config.get("resource_requirements", {})
        await self._check_resource_availability(required_resources)
        
        # Generate instance ID
        instance_id = f"{agent_type}_{project_id}_{uuid.uuid4().hex[:8]}"
        
        # Create instance
        instance = AgentInstance(
            instance_id=instance_id,
            agent_type=agent_type,
            project_id=project_id,
            workflow_stage=workflow_stage,
            config=config,
            tools=config.get("tools", []),
            permissions=dependencies.permissions
        )
        
        # Reserve resources
        await self._reserve_resources(instance_id, required_resources)
        
        # Register instance
        self._active_instances[instance_id] = instance
        
        logger.info(f"Created agent instance: {instance_id}")
        return instance
    
    async def start_instance(self, instance_id: str) -> bool:
        """Start an agent instance."""
        if instance_id not in self._active_instances:
            logger.error(f"Instance not found: {instance_id}")
            return False
        
        instance = self._active_instances[instance_id]
        instance.state = AgentState.RUNNING
        instance.started_at = datetime.now()
        instance.update_heartbeat()
        
        logger.info(f"Started agent instance: {instance_id}")
        return True
    
    async def stop_instance(self, instance_id: str, reason: str = "completed") -> bool:
        """Stop an agent instance."""
        if instance_id not in self._active_instances:
            return False
        
        instance = self._active_instances[instance_id]
        instance.state = AgentState.COMPLETED if reason == "completed" else AgentState.TERMINATED
        instance.completed_at = datetime.now()
        
        # Release resources
        await self._release_resources(instance_id)
        
        # Remove from active instances
        del self._active_instances[instance_id]
        
        logger.info(f"Stopped agent instance: {instance_id} (reason: {reason})")
        return True
    
    async def get_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """Get an agent instance by ID."""
        return self._active_instances.get(instance_id)
    
    async def list_active_instances(self, project_id: Optional[str] = None) -> List[AgentInstance]:
        """List active agent instances."""
        instances = list(self._active_instances.values())
        
        if project_id:
            instances = [i for i in instances if i.project_id == project_id]
        
        return instances
    
    async def cleanup_stale_instances(self, timeout_seconds: int = 3600) -> int:
        """Clean up stale instances that haven't sent heartbeats."""
        cutoff_time = datetime.now() - timedelta(seconds=timeout_seconds)
        stale_instances = []
        
        for instance_id, instance in self._active_instances.items():
            if instance.last_heartbeat and instance.last_heartbeat < cutoff_time:
                stale_instances.append(instance_id)
        
        # Stop stale instances
        for instance_id in stale_instances:
            await self.stop_instance(instance_id, "timeout")
        
        logger.info(f"Cleaned up {len(stale_instances)} stale instances")
        return len(stale_instances)
    
    async def _check_resource_availability(self, required: Dict[str, Any]):
        """Check if required resources are available."""
        required_cpu = required.get("cpu_cores", 1)
        required_memory = required.get("memory_gb", 1)
        
        if (self._resource_usage["cpu_cores"] + required_cpu > self._max_resources["cpu_cores"] or
            self._resource_usage["memory_gb"] + required_memory > self._max_resources["memory_gb"] or
            self._resource_usage["active_count"] >= self._max_resources["active_count"]):
            
            raise RuntimeError("Insufficient resources to create agent instance")
    
    async def _reserve_resources(self, instance_id: str, required: Dict[str, Any]):
        """Reserve resources for an agent instance."""
        self._resource_usage["cpu_cores"] += required.get("cpu_cores", 1)
        self._resource_usage["memory_gb"] += required.get("memory_gb", 1)
        self._resource_usage["active_count"] += 1
    
    async def _release_resources(self, instance_id: str):
        """Release resources from an agent instance."""
        instance = self._active_instances.get(instance_id)
        if instance:
            required = instance.config.get("resource_requirements", {})
            self._resource_usage["cpu_cores"] -= required.get("cpu_cores", 1)
            self._resource_usage["memory_gb"] -= required.get("memory_gb", 1)
            self._resource_usage["active_count"] -= 1


class AgentFactory:
    """
    Factory for creating and managing specialized agents.
    Handles dynamic agent generation, configuration, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the agent factory."""
        self.registry = AgentRegistry()
        self.lifecycle_manager = AgentLifecycleManager()
        self.llm_model = self._create_llm_model()
        
    def _create_llm_model(self) -> OpenAIModel:
        """Create LLM model for agents."""
        provider = OpenAIProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key
        )
        return OpenAIModel(settings.llm_model, provider=provider)
    
    async def create_agent(
        self,
        agent_type: AgentType,
        project_id: str,
        workflow_stage: WorkflowStage,
        dependencies: AgentDependencies,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Agent, AgentInstance]:
        """
        Create a specialized agent instance.
        
        Args:
            agent_type: Type of agent to create
            project_id: Associated project ID
            workflow_stage: Current workflow stage
            dependencies: Agent dependencies
            custom_config: Optional custom configuration
            
        Returns:
            Tuple of (Agent, AgentInstance)
        """
        # Get base configuration
        base_config = self.registry.get_agent_config(agent_type)
        
        # Merge with custom configuration
        config = {**base_config}
        if custom_config:
            config.update(custom_config)
        
        # Create agent instance
        instance = await self.lifecycle_manager.create_instance(
            agent_type=agent_type,
            project_id=project_id,
            workflow_stage=workflow_stage,
            config=config,
            dependencies=dependencies
        )
        
        # Create Pydantic AI agent
        agent = Agent(
            self.llm_model,
            deps_type=type(dependencies),
            system_prompt=config.get("system_prompt", "")
        )
        
        # Register tools (placeholder - actual tools would be registered here)
        await self._register_agent_tools(agent, config.get("tools", []))
        
        logger.info(f"Created agent {agent_type} for project {project_id}")
        return agent, instance
    
    async def start_agent(self, instance_id: str) -> bool:
        """Start an agent instance."""
        return await self.lifecycle_manager.start_instance(instance_id)
    
    async def stop_agent(self, instance_id: str, reason: str = "completed") -> bool:
        """Stop an agent instance."""
        return await self.lifecycle_manager.stop_instance(instance_id, reason)
    
    async def get_agent_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """Get an agent instance by ID."""
        return await self.lifecycle_manager.get_instance(instance_id)
    
    async def list_active_agents(self, project_id: Optional[str] = None) -> List[AgentInstance]:
        """List active agent instances."""
        return await self.lifecycle_manager.list_active_instances(project_id)
    
    async def execute_agent(
        self,
        agent: Agent,
        instance: AgentInstance,
        task: str,
        dependencies: AgentDependencies
    ) -> AgentResult:
        """
        Execute an agent with proper lifecycle management.
        
        Args:
            agent: Pydantic AI agent
            instance: Agent instance
            task: Task description
            dependencies: Agent dependencies
            
        Returns:
            Agent execution result
        """
        start_time = time.time()
        
        try:
            # Start the instance
            await self.start_agent(instance.instance_id)
            
            # Execute the agent
            result = await agent.run(task, deps=dependencies)
            
            # Create result object
            execution_time = time.time() - start_time
            agent_result = AgentResult(
                result_id=f"result_{instance.instance_id}_{int(start_time)}",
                agent_instance_id=instance.instance_id,
                agent_type=instance.agent_type,
                success=True,
                data=result.data if hasattr(result, 'data') else {"response": str(result)},
                execution_time_seconds=execution_time,
                token_usage=result.usage.total_tokens if hasattr(result, 'usage') else 0
            )
            
            # Stop the instance
            await self.stop_agent(instance.instance_id, "completed")
            
            logger.info(f"Agent {instance.instance_id} completed successfully")
            return agent_result
            
        except Exception as e:
            # Handle execution error
            execution_time = time.time() - start_time
            
            agent_result = AgentResult(
                result_id=f"result_{instance.instance_id}_{int(start_time)}",
                agent_instance_id=instance.instance_id,
                agent_type=instance.agent_type,
                success=False,
                error=str(e),
                execution_time_seconds=execution_time
            )
            
            # Stop the instance with error
            await self.stop_agent(instance.instance_id, "failed")
            
            logger.error(f"Agent {instance.instance_id} failed: {e}")
            return agent_result
    
    @asynccontextmanager
    async def agent_context(
        self,
        agent_type: AgentType,
        project_id: str,
        workflow_stage: WorkflowStage,
        dependencies: AgentDependencies,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for agent lifecycle management.
        
        Usage:
            async with factory.agent_context(AgentType.DISCOVERY, ...) as (agent, instance):
                result = await agent.run("task")
        """
        agent, instance = await self.create_agent(
            agent_type, project_id, workflow_stage, dependencies, custom_config
        )
        
        try:
            yield agent, instance
        finally:
            # Ensure cleanup
            await self.stop_agent(instance.instance_id, "context_exit")
    
    async def cleanup_stale_agents(self) -> int:
        """Clean up stale agent instances."""
        return await self.lifecycle_manager.cleanup_stale_instances()
    
    async def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        active_instances = await self.list_active_agents()
        
        stats = {
            "active_agents": len(active_instances),
            "agents_by_type": {},
            "agents_by_project": {},
            "resource_usage": self.lifecycle_manager._resource_usage,
            "max_resources": self.lifecycle_manager._max_resources
        }
        
        # Count by type and project
        for instance in active_instances:
            agent_type = instance.agent_type
            project_id = instance.project_id
            
            stats["agents_by_type"][agent_type] = stats["agents_by_type"].get(agent_type, 0) + 1
            stats["agents_by_project"][project_id] = stats["agents_by_project"].get(project_id, 0) + 1
        
        return stats
    
    async def _register_agent_tools(self, agent: Agent, tool_names: List[str]):
        """Register tools for an agent (placeholder implementation)."""
        # This would be implemented to register actual tools based on tool names
        # For now, this is a placeholder
        for tool_name in tool_names:
            logger.debug(f"Would register tool: {tool_name}")
    
    def register_custom_agent_type(
        self,
        agent_type: AgentType,
        config: Dict[str, Any],
        agent_class: Optional[Type] = None
    ):
        """Register a custom agent type with configuration."""
        self.registry._agent_configs[agent_type] = config
        
        if agent_class:
            self.registry.register_agent_class(agent_type, agent_class)
        
        logger.info(f"Registered custom agent type: {agent_type}")