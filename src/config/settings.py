"""
Comprehensive configuration management using pydantic-settings.
Follows patterns from use-cases/pydantic-ai/examples/main_agent_reference/settings.py
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Core Configuration
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # LLM Configuration
    llm_provider: str = Field(default="openai")
    llm_api_key: str = Field(...)
    llm_model: str = Field(default="gpt-4")
    llm_base_url: Optional[str] = Field(default="https://api.openai.com/v1")
    
    # External Tool APIs
    github_token: Optional[str] = Field(default=None)
    figma_access_token: Optional[str] = Field(default=None)
    jira_api_token: Optional[str] = Field(default=None)
    jira_base_url: Optional[str] = Field(default=None)
    
    # Orchestration Settings
    max_parallel_agents: int = Field(default=5, ge=1, le=20)
    context_retention_days: int = Field(default=30, ge=1, le=365)
    quality_gate_timeout: int = Field(default=300, ge=60, le=3600)  # seconds
    
    # Context Management
    context_storage_backend: str = Field(default="local")  # local, redis, postgresql
    context_db_url: Optional[str] = Field(default=None)
    context_max_size_mb: int = Field(default=100, ge=1, le=1000)
    
    # Agent Configuration
    agent_timeout: int = Field(default=1800, ge=60, le=7200)  # 30 minutes default
    agent_retry_attempts: int = Field(default=3, ge=1, le=10)
    agent_retry_delay: int = Field(default=5, ge=1, le=60)  # seconds
    
    # Security Settings
    enable_security_scanning: bool = Field(default=True)
    security_scan_timeout: int = Field(default=300, ge=30, le=1800)
    allowed_file_extensions: list[str] = Field(
        default_factory=lambda: [".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".md"]
    )
    
    # Performance Settings
    max_concurrent_workflows: int = Field(default=10, ge=1, le=100)
    workflow_checkpoint_interval: int = Field(default=300, ge=60, le=3600)  # seconds
    
    # External Service Timeouts
    github_api_timeout: int = Field(default=30, ge=5, le=300)
    figma_api_timeout: int = Field(default=30, ge=5, le=300)
    jira_api_timeout: int = Field(default=30, ge=5, le=300)
    
    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, v):
        """Ensure LLM API key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("LLM API key cannot be empty")
        return v
    
    @field_validator("jira_base_url")
    @classmethod
    def validate_jira_url(cls, v):
        """Validate Jira base URL format if provided."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Jira base URL must start with http:// or https://")
        return v
    
    @field_validator("context_db_url")
    @classmethod
    def validate_context_db_url(cls, v, info):
        """Validate context database URL if using external storage."""
        if info.data.get("context_storage_backend") in ["redis", "postgresql"] and not v:
            raise ValueError(f"Context DB URL required for {info.data.get('context_storage_backend')} backend")
        return v
    
    def get_external_tool_config(self, tool_name: str) -> dict:
        """Get configuration for external tools."""
        configs = {
            "github": {
                "token": self.github_token,
                "timeout": self.github_api_timeout,
                "base_url": "https://api.github.com"
            },
            "figma": {
                "token": self.figma_access_token,
                "timeout": self.figma_api_timeout,
                "base_url": "https://api.figma.com/v1"
            },
            "jira": {
                "token": self.jira_api_token,
                "timeout": self.jira_api_timeout,
                "base_url": self.jira_base_url
            }
        }
        return configs.get(tool_name, {})
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if external tool is properly configured."""
        config = self.get_external_tool_config(tool_name)
        return bool(config.get("token") and config.get("base_url"))
    
    def get_orchestration_limits(self) -> dict:
        """Get orchestration configuration limits."""
        return {
            "max_parallel_agents": self.max_parallel_agents,
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "agent_timeout": self.agent_timeout,
            "quality_gate_timeout": self.quality_gate_timeout,
            "workflow_checkpoint_interval": self.workflow_checkpoint_interval
        }


# Global settings instance
try:
    settings = Settings()
except Exception as e:
    # For testing and development, create settings with dummy values
    import os
    os.environ.setdefault("LLM_API_KEY", "test_key_for_development")
    print(f"Warning: Failed to load settings ({e}), using development defaults")
    settings = Settings()