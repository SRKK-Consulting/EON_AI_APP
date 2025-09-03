import os
import requests

# ============================
# Setup
# ============================
BING_API_KEY = os.getenv("BING_API_KEY")  # make sure this is set
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"

def search_news(query: str, count: int = 5):
    """Search Bing News for the given query."""
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count, "mkt": "en-US", "freshness": "Day"}
    
    response = requests.get(BING_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    results = []
    for article in data.get("value", []):
        results.append({
            "name": article.get("name"),
            "url": article.get("url"),
            "snippet": article.get("description"),
            "provider": article.get("provider", [{}])[0].get("name", "Unknown"),
            "datePublished": article.get("datePublished"),
        })
    return results


# ============================
# Example usage
# ============================
if __name__ == "__main__":
    for industry in ["oil and gas", "maritime"]:
        print(f"\n🔎 Latest news on {industry}:\n")
        try:
            news = search_news(f"latest {industry} industry", count=5)
            for n in news:
                print(f"- {n['name']} ({n['provider']})")
                print(f"  {n['snippet']}")
                print(f"  {n['url']}\n")
        except Exception as e:
            print(f"Error searching news for {industry}: {e}")
