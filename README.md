# EON_AI_APP

Scenario-driven demo for a Chainlit + langgraph sales analytics assistant that queries a Fabric Lakehouse, explains ML outputs (SHAP), fetches industry news via Azure AI Projects and synthesizes manager-style notes.

## Overview

The app is organized as a small pipeline (a StateGraph) with these logical nodes:

- orchestrator: parses the user query (via an LLM) and decides which agent nodes to run. It returns filters and an array of agent IDs to execute (strings: `"1"`, `"2"`, `"3"`).
- `agent1` (Data Agent): builds a T-SQL `SELECT` (via LLM + schema) and retrieves open deals from the Fabric Lakehouse using the SQLAlchemy `engine`. Can be replaced with Data Agent in Fabric. 
- `agent2` (ML Prediction Agent): runs a sales-classification ML model (the production model used to predict deal outcomes) and returns a probability that the deal will win or lose. These probabilities are explainable using SHAP values/ratios — `agent2` fetches SHAP rows for retrieved deals and uses the LLM to prepare plain-English explanations that list top positive and negative drivers and provide one actionable suggestion.
- `agent3` (News agent): queries an Azure AI Projects agent to obtain recent industry/customer news snippets. The news agent uses customer information returned by `agent1` (for example `customer_name` and `customer_industry`) to target searches and return snippets specific to the customers and industries found in the retrieved deals.
- synthesizer: compiles the collected outputs into a structured Markdown manager note (MEDDIC-aligned) and sends it to Chainlit.

The `main.py` file wires these nodes into a `langgraph.StateGraph` and registers Chainlit event handlers for chat start and messages.

---

## Quick start

1. Install dependencies in a Python environment (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the env vars below (or supply them via your environment/CI):

Required environment variables

```text
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
LAKEHOUSE_DB
OPEN_DEALS_TABLE
SHAP_TABLE
BING_SUBSCRIPTION_KEY
BING_SEARCH_URL
SQL_ENDPOINT
```

Notes:
- `DefaultAzureCredential` is used in `db_connection.py` and for Azure SDK clients. On local dev that may open an interactive browser for sign-in unless you configure a service principal or environment-based credentials.
- `OPEN_DEALS_TABLE` and `SHAP_TABLE` should point to the fully-qualified Fabric Lakehouse table names used by the app.

3. Launch the Chainlit app (from the project root):

```powershell
chainlit run main.py
```

Open the Chainlit UI it prints in your browser and send messages as the user.

---

## Scenario-based examples

Below are three common scenarios and how the system routes them to nodes.

Scenario 1 — "Give me the latest open deal"
- Example user query: `Give me the latest open deal.`
- What runs:
	- `orchestrator` (parses query -> picks agents)
	- `agent1` (Data Agent) — retrieves deal rows from `OPEN_DEALS_TABLE`
	- `synthesizer` — summarizes retrieved deal(s) into a concise response
- Expected output: A short Markdown note that lists the latest open deal(s) and key fields (opportunity number, customer, win probability, expected value, etc.).

Notes: this is the minimal analysis flow. The orchestrator will include `agent1` for retrieval queries by default. If the user mentions a specific opportunity number, the SQL will filter by `opportunity_number`.

DAG (Scenario 1)

```text
(●) orchestrator
	|   |   |
	v   v   v
(●) (Data Agent) --- (○) (ML Prediction Agent) --- (○) (News agent)
	\__________________________/
				 |
				 v
(●) synthesizer

Triggered: orchestrator -> agent1 -> synthesizer
```

Scenario 2 — "Give me the latest open deal and their potential outcome with probability"
- Example user query: `Give me the latest open deal and their potential outcome with probability.`
- What runs:
	- `orchestrator`
	- `agent1` (Data Agent)
	- `agent2` (ML Prediction Agent) — fetch SHAP rows and create human-readable explanations (probability + SHAP drivers)
	- `synthesizer` — include model probabilities and top SHAP drivers in the final note
- Expected output: A Markdown note including deal details and an explanation of model probability with top positive/negative drivers and one actionable suggestion.

Notes: mention of probability or model outputs cues the orchestrator to include `agent2` (ML Prediction Agent). If SHAP rows are missing for a deal, the synthesizer will note that and recommend next steps.

DAG (Scenario 2)

```text
(●) orchestrator
	|   |   |
	v   v   v
(●) (Data Agent) --- (●) (ML Prediction Agent) --- (○) (News agent)
	\__________________________/
				 |
				 v
(●) synthesizer

Triggered: orchestrator -> agent1 -> agent2 -> synthesizer
```

Scenario 3 — "Give me a detailed analysis and recommendation for these open deals"
- Example user query: `Give me a detailed analysis and recommendation for these open deals.`
- What runs:
	- `orchestrator`
	- `agent1` (Data Agent)
	- `agent2` (ML Prediction Agent)
	- `agent3` (News agent)
	- `synthesizer` (comprehensive manager note)
- Expected output: A structured, MEDDIC-aligned Markdown manager note with:
	1) What we looked at (filters / counts)
	2) Deal(s) identified — Markdown table
	3) Model outputs (probabilities & SHAP explanations)
	4) Industry and customer news snippets
	5) Detailed insights and reasoning
	6) Actionable, SMART recommendations (with MEDDIC alignment)
	7) Risks and data gaps

Notes: The synthesizer prompt enforces Markdown table syntax and MEDDIC-style recommendations. To ensure the news fetch is relevant, mention industry keywords or 'news', 'industry', or 'recent updates' in the user query.

DAG (Scenario 3)

```text
(●) orchestrator
	|   |   |
	v   v   v
(●) (Data Agent) --- (●) (ML Prediction Agent) --- (●) (News agent)
	\__________________________/
				 |
				 v
(●) synthesizer

Triggered: orchestrator -> agent1 -> [agent2, agent3] -> synthesizer
```

---

## How the orchestrator chooses agents

The orchestrator calls an LLM to return strict JSON with two keys: `filters` (string) and `agents` (array of strings from `['1','2','3']`). Practical guidance:

- Use words like `probability`, `prediction`, `explain` to trigger `agent2`.
- Use `news`, `industry`, `recent` to trigger `agent3`.
- Retrieval/analysis phrases usually trigger `agent1` automatically.

If the LLM returns invalid JSON the orchestrator falls back to default agents (`['1','2','3']`) and carries a parse error in `state["errors"]`.

---

## Troubleshooting

- LLM output parsing failures: check `stdout` logs — the orchestrator prints the raw LLM output. Improve prompts or add explicit JSON wrappers in the user query to reduce parsing errors.
- SQL authentication errors: `db_connection.py` uses `DefaultAzureCredential`. If you see interactive login issues, configure a service principal or set environment vars for non-interactive login.
- LLM or Azure SDK calls may fail in CI if credentials or endpoints are missing. For local dev, `python-dotenv` is used; ensure the `.env` file is loaded.

---

## Developer tips

- To test without calling Azure APIs, patch `make_llm()` in `config.py` to return a deterministic mock LLM or monkeypatch `nodes.make_llm` at runtime.
- To avoid interactive auth on import, consider refactoring `db_connection.py` to expose a factory function that obtains tokens and builds the SQLAlchemy engine on demand instead of at import time.
- Add unit tests around `orchestrator_node` by mocking `make_llm()` to return controlled responses (good way to verify JSON parsing/fallback logic).

---

If you'd like, I can:

1. Add a small smoke test that runs `orchestrator_node` with a mocked LLM to demonstrate routing for the three scenarios.
2. Refactor `db_connection.py` to move token retrieval into a factory function (recommended for testability).

Tell me which you'd prefer and I'll make the changes and run the tests locally.
