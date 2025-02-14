from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class FinancialMetric(Enum):
    """Standard metrics for financial analysis"""
    PROFIT_MARGIN = "profit_margin"
    OPERATING_MARGIN = "operating_margin"
    ROE = "return_on_equity"
    ROA = "return_on_assets"
    CURRENT_RATIO = "current_ratio"
    DEBT_TO_EQUITY = "debt_to_equity"
    ASSET_TURNOVER = "asset_turnover"

class PeerGroupCriteria(Enum):
    """Criteria for peer group formation"""
    MARKET_CAP = "market_cap"
    REVENUE = "revenue"
    SECTOR = "sector"
    INDUSTRY = "industry"
    REGION = "region"
    BUSINESS_MODEL = "business_model"

@dataclass
class AnalysisResult:
    """Structure for holding analysis results"""
    metric: str
    values: Dict[str, float]
    period: str
    statistics: Dict[str, float]
    analysis: str

@dataclass
class PeerGroup:
    """Represents a peer group and its characteristics"""
    id: str
    name: str
    criteria: Dict[PeerGroupCriteria, Any]
    companies: Set[str]
    metrics: Dict[str, float]

class FinancialAnalyzer:
    """Handles all financial analysis operations"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        self._peer_groups: Dict[str, PeerGroup] = {}
        self._company_to_groups: Dict[str, Set[str]] = defaultdict(set)
    
    def _calculate_metric(
        self,
        company: str,
        metric: FinancialMetric,
        period: str
    ) -> float:
        """Calculate a specific metric for a company"""
        try:
            income_stmt = self.data_source.get_financial_statement(
                company, "income", period
            )
            balance_stmt = self.data_source.get_financial_statement(
                company, "balance", period
            )
            
            if metric == FinancialMetric.PROFIT_MARGIN:
                return (income_stmt.get("netIncome", 0) / income_stmt.get("revenue", 1)) * 100
            
            elif metric == FinancialMetric.OPERATING_MARGIN:
                return (income_stmt.get("operatingIncome", 0) / income_stmt.get("revenue", 1)) * 100
            
            elif metric == FinancialMetric.ROE:
                return (income_stmt.get("netIncome", 0) / balance_stmt.get("totalEquity", 1)) * 100
            
            elif metric == FinancialMetric.ROA:
                return (income_stmt.get("netIncome", 0) / balance_stmt.get("totalAssets", 1)) * 100
            
            elif metric == FinancialMetric.CURRENT_RATIO:
                return balance_stmt.get("currentAssets", 0) / balance_stmt.get("currentLiabilities", 1)
            
            elif metric == FinancialMetric.DEBT_TO_EQUITY:
                return balance_stmt.get("totalLiabilities", 0) / balance_stmt.get("totalEquity", 1)
            
            elif metric == FinancialMetric.ASSET_TURNOVER:
                return income_stmt.get("revenue", 0) / balance_stmt.get("totalAssets", 1)
            
        except Exception as e:
            print(f"Error calculating {metric.value} for {company}: {str(e)}")
            return 0.0
    
    def compare_companies(
        self,
        companies: List[str],
        metrics: List[FinancialMetric],
        period: str
    ) -> Dict[str, AnalysisResult]:
        """Compare multiple companies across metrics"""
        results = {}
        
        for metric in metrics:
            values = {
                company: self._calculate_metric(company, metric, period)
                for company in companies
            }
            
            # Calculate statistics
            vals = list(values.values())
            stats = {
                'mean': np.mean(vals),
                'median': np.median(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals)
            }
            
            # Generate analysis
            sorted_companies = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=True
            )
            analysis = (
                f"For {metric.value}:\n"
                f"Leader: {sorted_companies[0][0]} ({sorted_companies[0][1]:.2f}%)\n"
                f"Average: {stats['mean']:.2f}%\n"
                f"Range: {stats['min']:.2f}% - {stats['max']:.2f}%"
            )
            
            results[metric.value] = AnalysisResult(
                metric=metric.value,
                values=values,
                period=period,
                statistics=stats,
                analysis=analysis
            )
        
        return results
    
    def create_peer_groups(
        self,
        companies: List[str],
        criteria: List[PeerGroupCriteria],
        period: str,
        num_groups: int = 3
    ) -> Dict[str, PeerGroup]:
        """Create peer groups based on specified criteria"""
        # Get company characteristics
        company_data = {}
        for company in companies:
            chars = {}
            income_stmt = self.data_source.get_financial_statement(
                company, "income", period
            )
            balance_stmt = self.data_source.get_financial_statement(
                company, "balance", period
            )
            metadata = self.data_source.get_company_metadata(company)
            
            chars[PeerGroupCriteria.MARKET_CAP] = balance_stmt.get("marketCap", 0)
            chars[PeerGroupCriteria.REVENUE] = income_stmt.get("revenue", 0)
            chars[PeerGroupCriteria.SECTOR] = metadata.get("sector", "Unknown")
            chars[PeerGroupCriteria.INDUSTRY] = metadata.get("industry", "Unknown")
            chars[PeerGroupCriteria.REGION] = metadata.get("region", "Unknown")
            
            company_data[company] = chars
        
        # Create feature matrix for clustering
        numeric_features = []
        categorical_features = defaultdict(set)
        
        for company, chars in company_data.items():
            company_features = []
            for criterion in criteria:
                value = chars.get(criterion)
                if criterion in {PeerGroupCriteria.MARKET_CAP, PeerGroupCriteria.REVENUE}:
                    company_features.append(float(value or 0))
                else:
                    categorical_features[criterion].add(value)
            
            if company_features:
                numeric_features.append(company_features)
        
        # Perform clustering if we have numeric features
        if numeric_features:
            numeric_features = np.array(numeric_features)
            normalized = (numeric_features - np.mean(numeric_features, axis=0)) / np.std(numeric_features, axis=0)
            
            kmeans = KMeans(n_clusters=min(num_groups, len(companies)))
            clusters = kmeans.fit_predict(normalized)
            
            for i in range(len(companies)):
                group_id = f"group_{clusters[i]}"
                if group_id not in self._peer_groups:
                    self._peer_groups[group_id] = PeerGroup(
                        id=group_id,
                        name=f"Peer Group {clusters[i] + 1}",
                        criteria={c: None for c in criteria},
                        companies=set(),
                        metrics={}
                    )
                
                self._peer_groups[group_id].companies.add(companies[i])
                self._company_to_groups[companies[i]].add(group_id)
        
        # Create categorical groups
        for criterion in criteria:
            if criterion not in {PeerGroupCriteria.MARKET_CAP, PeerGroupCriteria.REVENUE}:
                for company, chars in company_data.items():
                    value = chars.get(criterion)
                    group_id = f"{criterion.value}_{value}"
                    
                    if group_id not in self._peer_groups:
                        self._peer_groups[group_id] = PeerGroup(
                            id=group_id,
                            name=f"{criterion.value.title()}: {value}",
                            criteria={criterion: value},
                            companies=set(),
                            metrics={}
                        )
                    
                    self._peer_groups[group_id].companies.add(company)
                    self._company_to_groups[company].add(group_id)
        
        return self._peer_groups
    
    def analyze_peer_group(
        self,
        group_id: str,
        metrics: List[FinancialMetric],
        period: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for a peer group"""
        if group_id not in self._peer_groups:
            raise ValueError(f"Unknown peer group: {group_id}")
        
        group = self._peer_groups[group_id]
        results = {}
        
        for metric in metrics:
            values = []
            for company in group.companies:
                value = self._calculate_metric(company, metric, period)
                if value is not None:
                    values.append(value)
            
            if values:
                results[metric.value] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return results
    
    def compare_to_peers(
        self,
        company: str,
        metrics: List[FinancialMetric],
        period: str
    ) -> Dict[str, Dict[str, Any]]:
        """Compare a company to its peer groups"""
        if company not in self._company_to_groups:
            raise ValueError(f"No peer groups found for {company}")
        
        results = {}
        for group_id in self._company_to_groups[company]:
            group_metrics = self.analyze_peer_group(group_id, metrics, period)
            company_metrics = {
                metric.value: self._calculate_metric(company, metric, period)
                for metric in metrics
            }
            
            comparison = {}
            for metric, stats in group_metrics.items():
                company_value = company_metrics[metric]
                z_score = (company_value - stats['mean']) / stats['std'] if stats['std'] != 0 else 0
                values = [
                    self._calculate_metric(c, FinancialMetric(metric), period)
                    for c in self._peer_groups[group_id].companies
                ]
                percentile = sum(1 for v in values if v <= company_value) / len(values) * 100
                
                comparison[metric] = {
                    'company_value': company_value,
                    'peer_stats': stats,
                    'z_score': z_score,
                    'percentile': percentile
                }
            
            results[group_id] = comparison
        
        return results
