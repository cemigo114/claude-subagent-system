#!/usr/bin/env python3
"""
Simple validation script for the sub-agent system implementation.
Tests basic functionality without external dependencies.
"""

import sys
import asyncio
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config.settings import settings
        print("  ✓ Settings imported")
        
        from src.models.workflow_models import WorkflowStage, AgentType, ProjectWorkflow
        print("  ✓ Workflow models imported")
        
        from src.models.context_models import ContextSnapshot, HandoffPackage
        print("  ✓ Context models imported")
        
        from src.models.agent_models import AgentDependencies, AgentResult
        print("  ✓ Agent models imported")
        
        from src.orchestrator.workflow_analyzer import WorkflowAnalyzer
        print("  ✓ Workflow analyzer imported")
        
        from src.orchestrator.context_manager import ContextManager
        print("  ✓ Context manager imported")
        
        from src.orchestrator.agent_factory import AgentFactory
        print("  ✓ Agent factory imported")
        
        from src.agents.discovery_agent import DiscoveryAgent
        print("  ✓ Discovery agent imported")
        
        from src.utils.logging_config import setup_logging, get_logger
        print("  ✓ Logging utilities imported")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        from src.config.settings import settings
        
        # Test basic settings access
        assert hasattr(settings, 'app_env')
        assert hasattr(settings, 'max_parallel_agents')
        print("  ✓ Basic settings accessible")
        
        # Test orchestration limits
        limits = settings.get_orchestration_limits()
        assert isinstance(limits, dict)
        assert 'max_parallel_agents' in limits
        print("  ✓ Orchestration limits working")
        
        # Test tool configuration
        github_enabled = settings.is_tool_enabled("github")
        assert isinstance(github_enabled, bool)
        print("  ✓ Tool configuration working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False

def test_models():
    """Test data model validation."""
    print("Testing data models...")
    
    try:
        from src.models.workflow_models import WorkflowStage, AgentType, WorkflowNode, ProjectWorkflow
        from src.models.context_models import ContextSnapshot, ContextVersion
        
        # Test workflow node creation
        node = WorkflowNode(
            id="test_node",
            stage=WorkflowStage.DISCOVERY,
            agent_type=AgentType.DISCOVERY
        )
        assert node.id == "test_node"
        print("  ✓ Workflow node creation working")
        
        # Test project workflow creation
        workflow = ProjectWorkflow(
            project_id="test_project",
            name="Test Project",
            description="Test description",
            nodes=[node]
        )
        assert workflow.project_id == "test_project"
        assert len(workflow.nodes) == 1
        print("  ✓ Project workflow creation working")
        
        # Test context version
        version = ContextVersion(
            version=1,
            checksum="test_checksum",
            size_bytes=100,
            created_by="test_agent"
        )
        assert version.version == 1
        print("  ✓ Context version creation working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False

async def test_context_manager():
    """Test context manager functionality."""
    print("Testing context manager...")
    
    try:
        from src.orchestrator.context_manager import ContextManager, SQLiteContextStorage
        from src.models.workflow_models import WorkflowStage, AgentType
        
        # Create temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_context.db")
            
            # Test storage backend creation
            storage = SQLiteContextStorage(db_path)
            print("  ✓ SQLite storage backend created")
            
            # Test context manager creation
            cm = ContextManager("sqlite")
            cm.storage_backend = storage
            print("  ✓ Context manager created")
            
            # Test snapshot creation
            snapshot = await cm.create_snapshot(
                project_id="test_project",
                stage=WorkflowStage.DISCOVERY,
                agent_type=AgentType.DISCOVERY,
                data={"test": "data"}
            )
            
            assert snapshot.project_id == "test_project"
            assert snapshot.validate_integrity()
            print("  ✓ Context snapshot created and validated")
            
            # Test snapshot retrieval
            retrieved = await cm.get_snapshot(snapshot.snapshot_id)
            assert retrieved is not None
            assert retrieved.data == {"test": "data"}
            print("  ✓ Context snapshot retrieval working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Context manager test failed: {e}")
        return False

def test_workflow_analyzer():
    """Test workflow analyzer."""
    print("Testing workflow analyzer...")
    
    try:
        from src.orchestrator.workflow_analyzer import WorkflowAnalyzer
        
        # Test analyzer creation
        analyzer = WorkflowAnalyzer()
        print("  ✓ Workflow analyzer created")
        
        # Test template matching (doesn't require LLM)
        result = asyncio.run(analyzer._try_template_matching(
            "Build a simple web application", 
            "Test Project"
        ))
        
        if result:
            assert result.project_id
            assert result.name == "Test Project"
            print("  ✓ Template matching working")
            
            # Test workflow summary
            summary = analyzer.get_workflow_summary(result)
            assert "project_id" in summary
            print("  ✓ Workflow summary generation working")
        else:
            print("  ! Template matching returned no result (expected for some inputs)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Workflow analyzer test failed: {e}")
        return False

def test_discovery_agent():
    """Test discovery agent."""
    print("Testing discovery agent...")
    
    try:
        from src.agents.discovery_agent import DiscoveryAgent
        from src.models.workflow_models import AgentType
        
        # Test agent creation
        agent = DiscoveryAgent()
        assert agent.agent_type == AgentType.DISCOVERY
        print("  ✓ Discovery agent created")
        
        # Test system prompt
        prompt = agent.get_system_prompt()
        assert len(prompt) > 100
        assert "Discovery Agent" in prompt
        print("  ✓ System prompt working")
        
        # Test required tools
        tools = agent.get_required_tools()
        assert len(tools) > 0
        print("  ✓ Required tools defined")
        
        # Test input validation
        valid = asyncio.run(agent.validate_input("Build a social media platform"))
        assert valid
        print("  ✓ Input validation working")
        
        # Test capabilities
        capabilities = agent.get_capabilities()
        assert len(capabilities) > 0
        print("  ✓ Agent capabilities defined")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Discovery agent test failed: {e}")
        return False

async def test_agent_execution():
    """Test basic agent execution."""
    print("Testing agent execution...")
    
    try:
        from src.agents.discovery_agent import DiscoveryAgent
        
        agent = DiscoveryAgent()
        
        # Test execution without dependencies (fallback mode)
        result = await agent.execute(
            task="Build a task management application",
            context=None,
            dependencies=None
        )
        
        assert result.success
        assert result.data is not None
        assert result.execution_time_seconds > 0
        print("  ✓ Basic agent execution working")
        
        # Test execution stats
        stats = agent.get_execution_stats()
        assert stats["tasks_completed"] == 1
        print("  ✓ Execution stats tracking working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Agent execution test failed: {e}")
        return False

async def main():
    """Run all validation tests."""
    print("🚀 Validating Sub-Agent System Implementation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Models", test_models),
        ("Context Manager", test_context_manager),
        ("Workflow Analyzer", test_workflow_analyzer),
        ("Discovery Agent", test_discovery_agent),
        ("Agent Execution", test_agent_execution),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
                print(f"  ✓ {test_name} passed")
            else:
                failed += 1
                print(f"  ✗ {test_name} failed")
                
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All validation tests passed!")
        print("\nCore sub-agent system implementation is working correctly.")
        print("Ready for further development and testing.")
    else:
        print("⚠️  Some validation tests failed.")
        print("Please review the errors and fix issues before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)