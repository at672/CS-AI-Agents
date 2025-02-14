from setuptools import setup, find_packages

setup(
    name="financial_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.2.45",
        "langchain-google-genai>=2.0.4",
        "numpy",
        "pandas",
        "scikit-learn",
        "psycopg2-binary",
        "diskcache",
    ],
    python_requires=">=3.8",
    author="Avi Thakore",
    author_email="your.email@example.com",
    description="A financial analysis agent using LangGraph and Google's Gemini API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)