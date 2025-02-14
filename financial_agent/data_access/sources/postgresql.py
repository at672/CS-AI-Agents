from datetime import timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from ..core import FinancialDataSource, DataSourceConfig
from ..cache import TieredCache, cached_financial_data

class CachedPostgreSQLDataSource(FinancialDataSource):
    """PostgreSQL implementation with caching support"""
    
    def __init__(
        self,
        config: DataSourceConfig,
        cache_dir: Optional[Path] = None,
        memory_cache_size: int = 1000,
        default_ttl: Optional[timedelta] = timedelta(hours=1)
    ):
        self.conn_params = config.credentials
        self._connection = None
        self.cache = TieredCache(
            memory_cache_size=memory_cache_size,
            disk_cache_dir=cache_dir
        )
        self.default_ttl = default_ttl
    
    def connect(self) -> bool:
        try:
            self._connection = psycopg2.connect(
                **self.conn_params,
                cursor_factory=RealDictCursor
            )
            return True
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
    
    def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
        self.cache.clear()
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor"""
        if not self._connection:
            self.connect()
        
        cursor = self._connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    @cached_financial_data(ttl=timedelta(hours=1))
    def get_financial_statement(
        self,
        company: str,
        statement_type: str,
        period: str
    ) -> Dict[str, Any]:
        with self._get_cursor() as cursor:
            query = """
                SELECT *
                FROM financial_statements
                WHERE company_ticker = %s
                AND statement_type = %s
                AND period = %s
            """
            cursor.execute(query, (company, statement_type, period))
            result = cursor.fetchone()
            
            if not result:
                raise KeyError(
                    f"No data found for {company} {statement_type} in period {period}"
                )
            
            return dict(result)
    
    @cached_financial_data(ttl=timedelta(hours=24))
    def get_available_periods(self, company: str) -> list[str]:
        with self._get_cursor() as cursor:
            query = """
                SELECT DISTINCT period
                FROM financial_statements
                WHERE company_ticker = %s
                ORDER BY period
            """
            cursor.execute(query, (company,))
            return [row['period'] for row in cursor.fetchall()]
    
    @cached_financial_data(ttl=timedelta(hours=24))
    def get_company_metadata(self, company: str) -> Dict[str, Any]:
        with self._get_cursor() as cursor:
            query = """
                SELECT *
                FROM company_metadata
                WHERE ticker = %s
            """
            cursor.execute(query, (company,))
            result = cursor.fetchone()
            
            if not result:
                raise KeyError(f"No metadata found for company {company}")
            
            return dict(result)

def create_data_source(
    config_path: Path,
    cache_dir: Optional[Path] = None
) -> CachedPostgreSQLDataSource:
    """Create and initialize a data source"""
    config = DataSourceConfig.from_json(config_path)
    return CachedPostgreSQLDataSource(
        config=config,
        cache_dir=cache_dir
    )
