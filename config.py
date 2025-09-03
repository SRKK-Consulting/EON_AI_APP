import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import BingSearchAPIWrapper
load_dotenv()

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  
# Lakehouse (Spark SQL) config
LAKEHOUSE_DB = os.getenv("LAKEHOUSE_DB")        # your Fabric Lakehouse name
OPEN_DEALS_TABLE = os.getenv("OPEN_DEALS_TABLE")
SHAP_TABLE = os.getenv("SHAP_TABLE")
BING_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
BING_URL = os.getenv("BING_SEARCH_URL")

def make_llm() -> AzureChatOpenAI:
    """Create Azure OpenAI LLM without temperature parameter for o3-mini model"""
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
    )

search = BingSearchAPIWrapper(
    bing_subscription_key=BING_KEY,
    bing_search_url=BING_URL
)