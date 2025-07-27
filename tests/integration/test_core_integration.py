"""
Integration tests for core sub-agent system functionality.
Tests the interaction between major components.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.workflow_analyzer import WorkflowAnalyzer
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.agent_factory import AgentFactory
from src.agents.discovery_agent import DiscoveryAgent
from src.models.workflow_models import AgentType, WorkflowStage
from src.models.agent_models import AgentDependencies
from src.config.settings import settings


class TestCoreIntegration:
    """Test core integration between components."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test_context.db")
    
    @pytest.fixture
    def context_manager(self, temp_db_path):
        """Create context manager with temporary storage."""
        # Override storage backend to use temp path
        cm = ContextManager("sqlite")
        cm.storage_backend.db_path = Path(temp_db_path)
        cm.storage_backend._init_db()
        return cm
    
    @pytest.fixture
    def workflow_analyzer(self):
        """Create workflow analyzer instance."""
        return WorkflowAnalyzer()
    
    @pytest.fixture
    def agent_factory(self):
        """Create agent factory instance."""
        return AgentFactory()
    
    @pytest.fixture
    def discovery_agent(self):
        """Create discovery agent instance."""
        return DiscoveryAgent()
    
    def test_workflow_analysis_basic(self, workflow_analyzer):
        """Test basic workflow analysis functionality."""
        # This is a sync test since we can't easily mock LLM calls
        # In practice, you'd mock the LLM response
        
        description = "Build a simple web application for task management"
        
        # Test template matching (which doesn't require LLM)
        result = asyncio.run(workflow_analyzer._try_template_matching(description, "Test Project"))
        
        if result:  # Template matched
            assert result.project_id
            assert result.name == "Test Project"
            assert result.description == description
            assert len(result.nodes) > 0
            assert result.status.value == "initialized"
        
        # Test workflow summary generation
        if result:
            summary = workflow_analyzer.get_workflow_summary(result)
            assert "project_id" in summary
            assert "total_nodes" in summary
            assert "estimated_duration_hours" in summary
    
    @pytest.mark.asyncio
    async def test_context_management(self, context_manager):
        """Test context management functionality."""
        
        # Test context snapshot creation
        test_data = {
            "requirements": ["req1", "req2"],
            "constraints": ["constraint1"],
            "objectives": ["objective1"]
        }
        
        snapshot = await context_manager.create_snapshot(
            project_id="test_project",
            stage=WorkflowStage.DISCOVERY,
            agent_type=AgentType.DISCOVERY,
            data=test_data
        )
        
        assert snapshot.snapshot_id
        assert snapshot.project_id == "test_project"
        assert snapshot.stage == WorkflowStage.DISCOVERY
        assert snapshot.data == test_data
        assert snapshot.validate_integrity()
        
        # Test snapshot retrieval
        retrieved = await context_manager.get_snapshot(snapshot.snapshot_id)
        assert retrieved is not None
        assert retrieved.snapshot_id == snapshot.snapshot_id
        assert retrieved.data == test_data
        
        # Test latest snapshot retrieval
        latest = await context_manager.get_latest_snapshot("test_project", WorkflowStage.DISCOVERY)
        assert latest is not None
        assert latest.snapshot_id == snapshot.snapshot_id
    
    @pytest.mark.asyncio 
    async def test_agent_factory_basic(self, agent_factory):
        """Test basic agent factory functionality."""
        
        # Test factory stats
        stats = await agent_factory.get_factory_stats()
        assert "active_agents" in stats
        assert "resource_usage" in stats
        assert "max_resources" in stats
        
        # Test cleanup (should not fail even with no agents)
        cleaned = await agent_factory.cleanup_stale_agents()
        assert cleaned >= 0
    
    def test_discovery_agent_basic(self, discovery_agent):
        """Test basic discovery agent functionality."""
        
        # Test agent initialization
        assert discovery_agent.agent_type == AgentType.DISCOVERY
        assert len(discovery_agent.capabilities) > 0
        
        # Test system prompt
        prompt = discovery_agent.get_system_prompt()
        assert "Discovery Agent" in prompt
        assert len(prompt) > 100
        
        # Test required tools
        tools = discovery_agent.get_required_tools()
        assert len(tools) > 0
        assert "web_research" in tools
        
        # Test input validation
        valid_task = "Build a social media platform for dog owners"
        invalid_task = "x"
        
        assert asyncio.run(discovery_agent.validate_input(valid_task))
        assert not asyncio.run(discovery_agent.validate_input(invalid_task))
        
        # Test execution stats
        stats = discovery_agent.get_execution_stats()
        assert "tasks_completed" in stats
        assert "success_rate" in stats
    
    @pytest.mark.asyncio
    async def test_agent_execution_flow(self, discovery_agent):
        """Test agent execution without LLM calls."""
        
        task = "Build a task management application for small teams"
        
        # Test standalone processing (fallback mode without dependencies)
        result = await discovery_agent.execute(
            task=task,
            context=None,
            dependencies=None
        )
        
        assert result.success
        assert result.agent_type == AgentType.DISCOVERY
        assert result.data is not None
        assert "market_analysis" in result.data
        assert result.execution_time_seconds > 0
        assert result.quality_score >= 0.0
        assert result.completeness_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_integration_workflow(self, workflow_analyzer, context_manager, discovery_agent):
        """Test integration between multiple components."""
        
        # Step 1: Analyze workflow
        description = "Create a simple e-commerce website"
        workflow = await workflow_analyzer._try_template_matching(description, "E-commerce Site")
        
        if workflow:
            # Step 2: Create context snapshot
            initial_context_data = {
                "project_description": description,
                "project_name": workflow.name,
                "workflow_nodes": len(workflow.nodes)
            }
            
            context_snapshot = await context_manager.create_snapshot(
                project_id=workflow.project_id,
                stage=WorkflowStage.DISCOVERY,
                agent_type=AgentType.DISCOVERY,
                data=initial_context_data
            )
            
            # Step 3: Execute discovery agent
            result = await discovery_agent.execute(
                task=description,
                context=context_snapshot,
                dependencies=None  # Would normally have real dependencies
            )
            
            assert result.success
            
            # Step 4: Create handoff package
            handoff_package = await discovery_agent.create_handoff_package(
                target_agent=AgentType.SPECIFICATION,
                context=context_snapshot,
                result=result
            )
            
            assert handoff_package.source_agent == AgentType.DISCOVERY
            assert handoff_package.target_agent == AgentType.SPECIFICATION
            assert handoff_package.project_id == workflow.project_id
            assert len(handoff_package.artifacts) > 0
            
            # Validate handoff package quality
            validation_summary = handoff_package.get_validation_summary()
            assert "quality_score" in validation_summary
            assert "ready_for_handoff" in validation_summary
            assert "context_valid" in validation_summary
    
    def test_configuration_validation(self):
        """Test that configuration is properly loaded."""
        
        # Test that settings are accessible
        assert hasattr(settings, 'app_env')
        assert hasattr(settings, 'max_parallel_agents')
        assert hasattr(settings, 'context_retention_days')
        
        # Test orchestration limits
        limits = settings.get_orchestration_limits()
        assert "max_parallel_agents" in limits
        assert "agent_timeout" in limits
        
        # Test tool configuration checking
        github_enabled = settings.is_tool_enabled("github")
        assert isinstance(github_enabled, bool)


if __name__ == "__main__":
    # Run basic smoke test
    print("Running basic smoke test...")
    
    # Test imports
    try:
        from src.config.settings import settings
        from src.orchestrator.workflow_analyzer import WorkflowAnalyzer
        from src.orchestrator.context_manager import ContextManager
        from src.agents.discovery_agent import DiscoveryAgent
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import error: {e}")
        sys.exit(1)
    
    # Test basic initialization
    try:
        analyzer = WorkflowAnalyzer()
        agent = DiscoveryAgent()
        print("✓ Component initialization successful")
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        sys.exit(1)
    
    print("✓ Basic smoke test passed!")
    print("Run 'python -m pytest tests/integration/test_core_integration.py -v' for full tests")