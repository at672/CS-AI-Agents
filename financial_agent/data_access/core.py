from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Supported data source types"""
    FILE = "file"
    POSTGRESQL = "postgresql"
    ARCTIC = "arctic"
    SNOWFLAKE = "snowflake"

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    type: DataSourceType
    credentials: Dict[str, Any]
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DataSourceConfig':
        """Create config from JSON file"""
        with open(json_path) as f:
            data = json.load(f)
        return cls(
            type=DataSourceType(data['type']),
            credentials=data['credentials']
        )

class FinancialDataSource(ABC):
    """Abstract base class for financial data sources"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source"""
        pass
    
    @abstractmethod
    def get_financial_statement(
        self,
        company: str,
        statement_type: str,
        period: str
    ) -> Dict[str, Any]:
        """Retrieve financial statement data"""
        pass
    
    @abstractmethod
    def get_available_periods(self, company: str) -> list[str]:
        """Get available time periods for a company"""
        pass
    
    @abstractmethod
    def get_company_metadata(self, company: str) -> Dict[str, Any]:
        """Get company metadata (sector, industry, etc.)"""
        pass
