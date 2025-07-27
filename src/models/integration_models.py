"""
External tool integration data models.
Models for GitHub, Figma, Jira, and other external service integrations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum


class IntegrationStatus(str, Enum):
    """Status of external integrations."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"


class ExternalToolConfig(BaseModel):
    """Configuration for external tools."""
    tool_name: str = Field(..., description="Name of the external tool")
    base_url: str = Field(..., description="Base URL for the tool's API")
    api_version: str = Field(default="v1", description="API version")
    timeout_seconds: int = Field(default=30, ge=5, le=300, description="Request timeout")
    
    # Authentication (sensitive fields excluded from serialization)
    auth_type: str = Field(default="bearer", description="Authentication type")
    api_key: Optional[str] = Field(None, description="API key or token")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(default=1000, ge=1, description="Rate limit per hour")
    
    # Status
    status: IntegrationStatus = Field(IntegrationStatus.INACTIVE, description="Integration status")
    last_check: Optional[datetime] = Field(None, description="Last health check")
    error_message: Optional[str] = Field(None, description="Last error message")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        # Exclude sensitive data from serialization
        fields = {
            'api_key': {'exclude': True}
        }
    
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v):
        """Ensure base URL starts with http or https."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v


class GitHubRepo(BaseModel):
    """GitHub repository representation."""
    repo_id: str = Field(..., description="Repository identifier")
    owner: str = Field(..., description="Repository owner")
    name: str = Field(..., description="Repository name")
    full_name: str = Field(..., description="Full repository name (owner/name)")
    description: Optional[str] = Field(None, description="Repository description")
    
    # Repository metadata
    url: str = Field(..., description="Repository URL")
    clone_url: str = Field(..., description="Clone URL")
    default_branch: str = Field(default="main", description="Default branch")
    language: Optional[str] = Field(None, description="Primary programming language")
    
    # Status
    private: bool = Field(False, description="Whether repository is private")
    archived: bool = Field(False, description="Whether repository is archived")
    disabled: bool = Field(False, description="Whether repository is disabled")
    
    # Statistics
    stars_count: int = Field(default=0, ge=0, description="Number of stars")
    forks_count: int = Field(default=0, ge=0, description="Number of forks")
    open_issues_count: int = Field(default=0, ge=0, description="Number of open issues")
    
    # Timestamps
    created_at: datetime = Field(..., description="Repository creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    pushed_at: Optional[datetime] = Field(None, description="Last push timestamp")
    
    # Integration metadata
    last_sync: Optional[datetime] = Field(None, description="Last synchronization")
    sync_status: str = Field(default="pending", description="Synchronization status")
    
    def get_github_url(self) -> str:
        """Get the GitHub web URL."""
        return f"https://github.com/{self.full_name}"
    
    def is_active(self) -> bool:
        """Check if repository is active (not archived or disabled)."""
        return not (self.archived or self.disabled)


class GitHubIssue(BaseModel):
    """GitHub issue representation."""
    issue_id: str = Field(..., description="Issue identifier")
    number: int = Field(..., description="Issue number")
    title: str = Field(..., description="Issue title")
    body: Optional[str] = Field(None, description="Issue body")
    
    # Status
    state: str = Field(..., description="Issue state (open/closed)")
    labels: List[str] = Field(default_factory=list, description="Issue labels")
    assignees: List[str] = Field(default_factory=list, description="Assigned users")
    
    # Metadata
    author: str = Field(..., description="Issue author")
    url: str = Field(..., description="Issue URL")
    repository: str = Field(..., description="Repository full name")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    closed_at: Optional[datetime] = Field(None, description="Closure timestamp")


class FigmaProject(BaseModel):
    """Figma project representation."""
    project_id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    
    # Access
    team_id: str = Field(..., description="Team identifier")
    url: str = Field(..., description="Project URL")
    
    # Status
    status: str = Field(default="active", description="Project status")
    
    # Files
    files: List[Dict[str, Any]] = Field(default_factory=list, description="Project files")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Integration metadata
    last_sync: Optional[datetime] = Field(None, description="Last synchronization")
    sync_status: str = Field(default="pending", description="Synchronization status")


class FigmaFile(BaseModel):
    """Figma file representation."""
    file_id: str = Field(..., description="File identifier")
    name: str = Field(..., description="File name")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    
    # Content
    document: Optional[Dict[str, Any]] = Field(None, description="Document structure")
    components: Dict[str, Any] = Field(default_factory=dict, description="File components")
    styles: Dict[str, Any] = Field(default_factory=dict, description="File styles")
    
    # Metadata
    version: str = Field(..., description="File version")
    last_modified: datetime = Field(..., description="Last modification timestamp")
    
    # Access
    project_id: Optional[str] = Field(None, description="Associated project ID")
    url: str = Field(..., description="File URL")


class JiraIssue(BaseModel):
    """Jira issue representation."""
    issue_id: str = Field(..., description="Issue identifier")
    key: str = Field(..., description="Issue key (e.g., PROJ-123)")
    summary: str = Field(..., description="Issue summary")
    description: Optional[str] = Field(None, description="Issue description")
    
    # Status and type
    status: str = Field(..., description="Issue status")
    issue_type: str = Field(..., description="Issue type")
    priority: str = Field(..., description="Issue priority")
    
    # Assignment
    assignee: Optional[str] = Field(None, description="Assigned user")
    reporter: str = Field(..., description="Issue reporter")
    
    # Project context
    project_key: str = Field(..., description="Project key")
    project_name: str = Field(..., description="Project name")
    
    # Timestamps
    created: datetime = Field(..., description="Creation timestamp")
    updated: datetime = Field(..., description="Last update timestamp")
    resolved: Optional[datetime] = Field(None, description="Resolution timestamp")
    
    # Custom fields and metadata
    labels: List[str] = Field(default_factory=list, description="Issue labels")
    components: List[str] = Field(default_factory=list, description="Issue components")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom field values")
    
    # URLs
    url: str = Field(..., description="Issue URL")
    
    def is_resolved(self) -> bool:
        """Check if issue is resolved."""
        return self.resolved is not None
    
    def get_age_days(self) -> int:
        """Get issue age in days."""
        return (datetime.now() - self.created).days


class JiraProject(BaseModel):
    """Jira project representation."""
    project_id: str = Field(..., description="Project identifier")
    key: str = Field(..., description="Project key")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    
    # Type and category
    project_type: str = Field(..., description="Project type")
    project_category: Optional[str] = Field(None, description="Project category")
    
    # Lead and permissions
    lead: str = Field(..., description="Project lead")
    url: str = Field(..., description="Project URL")
    
    # Status
    archived: bool = Field(False, description="Whether project is archived")
    
    # Statistics
    issues_count: int = Field(default=0, ge=0, description="Total number of issues")
    
    # Integration metadata
    last_sync: Optional[datetime] = Field(None, description="Last synchronization")
    sync_status: str = Field(default="pending", description="Synchronization status")


class WebSearchResult(BaseModel):
    """Web search result for research tools."""
    result_id: str = Field(..., description="Result identifier")
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet")
    
    # Relevance
    score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    rank: int = Field(..., ge=1, description="Search result rank")
    
    # Metadata
    domain: str = Field(..., description="Source domain")
    published_date: Optional[datetime] = Field(None, description="Content publication date")
    
    # Analysis
    content_type: str = Field(default="webpage", description="Type of content")
    language: str = Field(default="en", description="Content language")
    
    # Timestamps
    crawled_at: datetime = Field(default_factory=datetime.now, description="When result was crawled")


class SecurityScanResult(BaseModel):
    """Security scan result."""
    scan_id: str = Field(..., description="Scan identifier")
    file_path: str = Field(..., description="Scanned file path")
    
    # Findings
    vulnerabilities: List[Dict[str, Any]] = Field(default_factory=list, description="Found vulnerabilities")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Security warnings")
    
    # Severity
    critical_count: int = Field(default=0, ge=0, description="Critical issues count")
    high_count: int = Field(default=0, ge=0, description="High severity issues count")
    medium_count: int = Field(default=0, ge=0, description="Medium severity issues count")
    low_count: int = Field(default=0, ge=0, description="Low severity issues count")
    
    # Status
    passed: bool = Field(..., description="Whether scan passed security requirements")
    scanner_version: str = Field(..., description="Scanner version used")
    
    # Timestamps
    scanned_at: datetime = Field(default_factory=datetime.now, description="Scan timestamp")
    
    def get_total_issues(self) -> int:
        """Get total number of security issues."""
        return self.critical_count + self.high_count + self.medium_count + self.low_count
    
    def get_severity_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity."""
        return {
            "critical": self.critical_count,
            "high": self.high_count,
            "medium": self.medium_count,
            "low": self.low_count,
            "total": self.get_total_issues()
        }