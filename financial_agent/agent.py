from typing import Annotated, Dict, Any, Optional, List
from typing_extensions import TypedDict
from pathlib import Path
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages

from .data_access.sources.postgresql import create_data_source
from .analysis import FinancialAnalyzer, FinancialMetric, PeerGroupCriteria

class AnalysisState(TypedDict):
    """State for financial analysis conversation."""
    messages: Annotated[list, add_messages]  # Conversation history
    sentiment_score: float  # Confidence/sentiment score (0-1)
    current_company: Optional[str]  # Currently analyzed company
    analysis_cache: Dict[str, Any]  # Cache for analysis results

class FinancialAgent:
    """Financial analysis agent with integrated analysis capabilities"""
    
    def __init__(
        self,
        config_path: Path,
        cache_dir: Optional[Path] = None,
        model_name: str = "gemini-1.5-pro-latest"
    ):
        # Initialize components
        self.data_source = create_data_source(config_path, cache_dir)
        self.analyzer = FinancialAnalyzer(self.data_source)
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def _create_tools(self):
        """Create analysis tools"""
        
        @tool
        def get_financial_data(company: str, statement_type: str, period: str) -> str:
            """
            Retrieve financial statement data.
            
            Args:
                company: Company ticker symbol
                statement_type: Type of statement ('income', 'balance', 'cash_flow')
                period: Time period (e.g., 'Q3_2023')
            """
            try:
                data = self.data_source.get_financial_statement(
                    company=company,
                    statement_type=statement_type,
                    period=period
                )
                return f"Financial data for {company} {statement_type} in {period}:\n{data}"
            except Exception as e:
                return f"Error retrieving data: {str(e)}"

        @tool
        def compare_companies(
            companies: List[str],
            metrics: List[str],
            period: str
        ) -> str:
            """
            Compare multiple companies across metrics.
            
            Args:
                companies: List of company tickers
                metrics: List of metrics to compare
                period: Time period
            """
            try:
                metric_enums = [FinancialMetric(m.lower()) for m in metrics]
                results = self.analyzer.compare_companies(companies, metric_enums, period)
                
                output = [f"Comparative Analysis for {period}:"]
                for metric, result in results.items():
                    output.append(f"\n{metric.upper()}:")
                    for company, value in result.values.items():
                        output.append(f"{company}: {value:.2f}%")
                    output.append(f"\n{result.analysis}")
                
                return "\n".join(output)
            except Exception as e:
                return f"Error comparing companies: {str(e)}"

        @tool
        def create_peer_groups(
            companies: List[str],
            criteria: List[str],
            period: str,
            num_groups: int = 3
        ) -> str:
            """
            Create and analyze peer groups.
            
            Args:
                companies: List of company tickers
                criteria: Criteria for grouping
                period: Time period
                num_groups: Number of groups for numeric criteria
            """
            try:
                criteria_enums = [PeerGroupCriteria(c.lower()) for c in criteria]
                groups = self.analyzer.create_peer_groups(
                    companies,
                    criteria_enums,
                    period,
                    num_groups
                )
                
                output = ["Peer Groups Created:"]
                for group_id, group in groups.items():
                    output.append(f"\n{group.name}")
                    output.append("Companies:")
                    for company in sorted(group.companies):
                        output.append(f"  - {company}")
                    output.append("Criteria:")
                    for criterion, value in group.criteria.items():
                        if value:
                            output.append(f"  - {criterion.value}: {value}")
                
                return "\n".join(output)
            except Exception as e:
                return f"Error creating peer groups: {str(e)}"

        @tool
        def analyze_peer_group(
            group_id: str,
            metrics: List[str],
            period: str
        ) -> str:
            """
            Analyze metrics for a peer group.
            
            Args:
                group_id: Peer group identifier
                metrics: List of metrics to analyze
                period: Time period
            """
            try:
                metric_enums = [FinancialMetric(m.lower()) for m in metrics]
                results = self.analyzer.analyze_peer_group(group_id, metric_enums, period)
                
                output = [f"Peer Group Analysis for {group_id} ({period}):"]
                for metric, stats in results.items():
                    output.append(f"\n{metric.upper()}:")
                    output.append(f"  Mean: {stats['mean']:.2f}")
                    output.append(f"  Median: {stats['median']:.2f}")
                    output.append(f"  Std Dev: {stats['std']:.2f}")
                    output.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
                
                return "\n".join(output)
            except Exception as e:
                return f"Error analyzing peer group: {str(e)}"

        @tool
        def update_sentiment(score: float, reason: str) -> str:
            """
            Update confidence/sentiment score.
            
            Args:
                score: New score (0-1)
                reason: Reason for the update
            """
            if not 0 <= score <= 1:
                return "Error: Score must be between 0 and 1"
            return f"Updated sentiment score to {score}: {reason}"

        return [
            get_financial_data,
            compare_companies,
            create_peer_groups,
            analyze_peer_group,
            update_sentiment
        ]
    
    def create_graph(self) -> StateGraph:
        """Create the agent's processing graph"""
        
        def agent_node(state: AnalysisState) -> AnalysisState:
            """Process queries and generate responses"""
            if state["messages"]:
                response = self.llm_with_tools.invoke(
                    [AGENT_SYSTEM_PROMPT] + state["messages"]
                )
            else:
                response = AIMessage(
                    content="Welcome! I'm your Financial Analysis Agent. "
                    "I can help you analyze company financial data, perform "
                    "comparisons, and provide insights. What would you like "
                    "to analyze?"
                )
            return {"messages": [response]}
        
        def tool_processing_node(state: AnalysisState) -> AnalysisState:
            """Process tool calls and update state"""
            tool_msg = state["messages"][-1]
            outbound_msgs = []
            
            for tool_call in tool_msg.tool_calls:
                if tool_call["name"] == "update_sentiment":
                    state["sentiment_score"] = float(tool_call["args"]["score"])
                
                result = self._process_tool_call(tool_call, state)
                outbound_msgs.append(result)
            
            return {
                "messages": outbound_msgs,
                "sentiment_score": state.get("sentiment_score", 0.5)
            }
        
        # Build the graph
        graph = StateGraph(AnalysisState)
        
        # Add nodes
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_processing_node)
        
        # Add edges
        graph.add_edge("agent", "tools")
        graph.add_edge("tools", "agent")
        
        return graph.compile()
    
    def _process_tool_call(
        self,
        tool_call: Dict[str, Any],
        state: AnalysisState
    ) -> AIMessage:
        """Process individual tool calls"""
        try:
            tool = next(t for t in self.tools if t.name == tool_call["name"])
            result = tool(**tool_call["args"])
            return AIMessage(content=str(result))
        except Exception as e:
            return AIMessage(
                content=f"Error processing tool call {tool_call['name']}: {str(e)}"
            )

# System prompt
AGENT_SYSTEM_PROMPT = """
You are a Financial Analysis Agent specialized in analyzing company financial data.
You can:

1. Access and analyze financial statements
2. Compare multiple companies
3. Create and analyze peer groups
4. Track sentiment/confidence scores

When analyzing:
- Be precise with calculations and data
- Explain your reasoning clearly
- Update sentiment scores based on findings
- Provide actionable insights

For each response:
1. Verify data availability first
2. Use appropriate tools for the task
3. Consider multiple metrics
4. Maintain objectivity
5. Be clear about limitations

Remember to use sentiment scores to reflect:
- Data quality and availability
- Metric performance
- Relative peer performance
- Overall confidence in analysis
"""

def create_agent(
    config_path: Path,
    cache_dir: Optional[Path] = None
) -> FinancialAgent:
    """Create and initialize a financial agent"""
    return FinancialAgent(config_path, cache_dir)

def run_analysis(
    config_path: Path,
    cache_dir: Optional[Path] = None,
    query: Optional[str] = None
):
    """Run financial analysis"""
    agent = create_agent(config_path, cache_dir)
    
    state = {
        "messages": [],
        "sentiment_score": 0.5,
        "current_company": None,
        "analysis_cache": {}
    }
    
    if query:
        state["messages"].append(query)
    
    try:
        config = {"recursion_limit": 100}
        final_state = agent.create_graph().invoke(state, config)
        
        print("\nAnalysis Results:")
        print("=" * 50)
        for msg in final_state.get("messages", []):
            print(f"{type(msg).__name__}: {msg.content}\n")
        print(f"Final Sentiment Score: {final_state.get('sentiment_score', 0.5):.2f}")
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
    finally:
        agent.data_source.disconnect()
