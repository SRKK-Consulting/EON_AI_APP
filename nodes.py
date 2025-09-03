import json
from config import make_llm, LAKEHOUSE_DB, OPEN_DEALS_TABLE, SHAP_TABLE
from state import AppState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from db_connection import engine  # Use SQLAlchemy engine for Fabric Lakehouse SQL endpoint
from typing import List, Dict, Any, Optional
from langchain_community.utilities import BingSearchAPIWrapper
def orchestrator_node(state: AppState) -> AppState:
    """Parse the user query -> filters + which agents to run."""
    llm = make_llm()
    parse_prompt = ChatPromptTemplate.from_template(
        (
            "You are a planner for a sales analytics assistant.\n"
            "User query: {query}\n\n"
            "1) Extract a concise SQL WHERE clause to filter deals in table `{table}`.\n"
            "2) Decide which agents to call:\n"
            "   - '1' Retrieval: query open deals\n"
            "   - '2' Explain: explain ML probability using SHAP\n\n"
            "   - '3' News: search for related industry news\n"
            "Return STRICT JSON with keys: filters (string), agents (array of strings from ['1','2']).\n"
            "If unsure, use empty string for filters and agents=['1','2'].\n"
        )
    )
    chain = parse_prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": state["user_query"], "table": OPEN_DEALS_TABLE}).strip()

    parsed_filters, agents, errors = "", ["1","2"], []

    try:
        data = json.loads(raw)
        parsed_filters = data.get("filters", "") or ""
        agents = data.get("agents", ["1","2"]) or ['1','2','3']
        if any(a not in {"1","2","3"} for a in agents):
            errors.append(f"Invalid agent id in {agents}; defaulting to ['1','2','3'].")
            agents = ['1','2','3']
    except Exception as e:
        errors.append(f"Failed to parse planner JSON: {e}. Raw: {raw}")

    return {
        "parsed_filters": parsed_filters,
        "agents_to_call": agents,
        "errors": state.get("errors", []) + errors,
    }

def agent1_node(state: AppState) -> AppState:
    """Agent 1: NL → SQL → T-SQL retrieval from Fabric Lakehouse SQL endpoint."""
    if "1" not in state.get("agents_to_call", []):
        return {}

    llm = make_llm()

    # --- Fetch schema instead of just column names ---
    schema_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{OPEN_DEALS_TABLE.split('.')[-1]}'
    """
    schema_df = pd.read_sql(schema_query, engine)

    # Build schema description string: column_name (data_type)
    schema_description = ", ".join(
        f"{row.COLUMN_NAME} ({row.DATA_TYPE})" for _, row in schema_df.iterrows()
    )

    # Debug print schema
    print("\n=== Table Schema ===")
    print(schema_df)
    print("====================\n")

    # --- Build prompt with schema ---
    sql_prompt = ChatPromptTemplate.from_template(
       (
        "Compose only the SQL statement for T-SQL.\n"
        "Target table: `{table}`.\n"
        f"Schema: {schema_description}\n\n"

        "=== HARD CONSTRAINTS ===\n"
        "1. You MUST explicitly list the columns in the SELECT clause.\n"
        "2. You MUST always include `opportunity_number` in the SELECT list.\n"
        "3. DO NOT use `SELECT *` under any circumstance.\n"
        "4. Limit the SELECT clause to only essential columns: "
        "`opportunity_number, win_probability, predicted_outcome, expected_value, topic, opportunity_type, op_created_on,`.\n"
        "5. The table already contains only open deals — do NOT filter by op_status.\n"
        "6. SQL must be valid T-SQL. Do NOT add code fences.\n\n"

        "=== Notes ===\n"
        "- Use column types appropriately (quote strings, compare numerics correctly).\n"
        "- If filters is empty, return all open deals.\n"
        "- If filters exist, apply them to narrow down results.\n\n"

        "filters: {filters}"
       )
    )

    sql_str = (sql_prompt | llm | StrOutputParser()).invoke(
        {"table": OPEN_DEALS_TABLE, "filters": state.get("parsed_filters", "")}
    ).strip()

    # --- Minimal safeguard fallback ---
    if not sql_str.lower().startswith("select"):
        sql_str = f"SELECT *,  FROM {OPEN_DEALS_TABLE}" + (
            f" WHERE {state.get('parsed_filters')}" if state.get("parsed_filters") else ""
        )

    # Debug print SQL query
    print("\n=== Generated SQL ===")
    print(sql_str)
    print("====================\n")

    errors = state.get("errors", [])
    deals: List[Dict[str, Any]] = []
    try:
        df = pd.read_sql(sql_str, engine)
        deals = df.to_dict(orient="records")
    except Exception as e:
        errors.append(f"SQL execution failed: {e}. SQL: {sql_str}")

    return {
        "sql_query": sql_str,
        "retrieved_deals": deals,
        "errors": errors,
    }


def agent2_node(state: AppState) -> AppState:
    """Agent 2: Explain using SHAP values (from shap_values table)."""
    if "2" not in state.get("agents_to_call", []):
        return {}

    deals = state.get("retrieved_deals") or []
    if not deals:
        return {"explanations": []}

    # Collect deal IDs
    opportunity_numbers = [str(d["opportunity_number"]).strip() for d in deals if d.get("opportunity_number")]

    # --- Debug: Fetch SHAP schema ---
    shap_schema_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{SHAP_TABLE.split('.')[-1]}'
    """
    shap_schema_df = pd.read_sql(shap_schema_query, engine)
    print("\n=== SHAP Table Schema ===")
    print(shap_schema_df)
    print("=========================\n")

    # --- Fetch SHAP values for the deals ---
    shap_sql = f"""
    SELECT *
    FROM {SHAP_TABLE}
    WHERE LTRIM(RTRIM(opportunity_number)) IN ({','.join([repr(x) for x in opportunity_numbers])})
    """
    print("\n=== SHAP SQL ===")
    print(shap_sql)
    print("================\n")

    shap_df = pd.read_sql(shap_sql, engine)

    # Convert DataFrame into dict index keyed by opportunity_number
    shap_index = {
        str(row["opportunity_number"]).strip(): row.to_dict()
        for _, row in shap_df.iterrows()
    }

    llm = make_llm()
    prompt = ChatPromptTemplate.from_template(
        (
            "You explain model outputs for a sales win-probability model.\n"
            "Each row in the SHAP table has per-feature SHAP contributions for one deal.\n"
            "- `prediction` column is the model's predicted win log odds.\n"
            "- Other columns (owning_region_encoded, revenue_tier_encoded, deal_age_days, etc.) "
            "are SHAP contributions for that feature.\n\n"
            "Your task:\n"
            "- Summarize the probability in plain English.\n"
            "- List top positive and negative drivers (up to 3 each) by SHAP magnitude.\n"
            "- Provide one actionable suggestion.\n\n"
            "probability: {prob}\n"
            "shap: {shap}\n"
            "If shap is missing or empty, say so and explain based on core attributes if any."
        )
    )

    explanations = []
    for deal in deals:
        op_id = str(deal.get("opportunity_number", "")).strip()
        shap_row = shap_index.get(op_id)

        if not shap_row:
            explanations.append({
                "opportunity_number": op_id,
                "deal_id": deal.get("deal_id") or deal.get("id"),
                "explanation": "No SHAP data available for this deal."
            })
            continue

        prob = shap_row.get("prediction", None)

        # Remove keys we don't want to show as SHAP drivers
        shap_features = {
            k: v for k, v in shap_row.items()
            if k not in ("opportunity_number", "prediction")
        }

        shap_str = json.dumps(shap_features, ensure_ascii=False)

        text = (prompt | llm | StrOutputParser()).invoke(
            {"prob": prob, "shap": shap_str}
        )
        explanations.append({
            "opportunity_number": op_id,
            "deal_id": deal.get("deal_id") or deal.get("id"),
            "explanation": text,
        })

    return {"explanations": explanations}

def agent3_node(state: AppState) -> AppState:
    """Agent 3: Search Bing for recent news on account industries (oil & gas, maritime)."""
    if "3" not in state.get("agents_to_call", []):
        return {}

    deals = state.get("retrieved_deals") or []
    if not deals:
        return {"industry_news": []}

    # Collect unique industries from retrieved deals
    industries = {str(d.get("account_industry", "")).strip().lower() for d in deals}
    industries = {i for i in industries if i}  # remove blanks

    search = BingSearchAPIWrapper()

    industry_news = []
    for industry in industries:
        # Only run for Oil & Gas / Maritime
        if "oil" in industry or "gas" in industry or "maritime" in industry:
            query = f"latest news {industry} industry"
            try:
                results = search.results(query, num_results=5)
                snippets = [r["snippet"] for r in results]
                industry_news.append({
                    "industry": industry,
                    "query": query,
                    "snippets": snippets
                })
            except Exception as e:
                industry_news.append({
                    "industry": industry,
                    "query": query,
                    "error": str(e),
                    "snippets": []
                })

    return {"industry_news": industry_news}


def synthesizer_node(state: AppState) -> AppState:
    """Compile the final response for the user."""
    llm = make_llm()
    synth_prompt = ChatPromptTemplate.from_template(
        (
            "Synthesize a concise manager-ready note.\n"
            "Context:\n"
            "- SQL used: {sql}\n"
            "- #Deals: {n_deals}\n"
            "- Explanations: {exps}\n"
            "- Any errors: {errs}\n\n"
            "- Industry news: {news}\n"
            "Output structure:\n"
            "1) What we looked at (filters, counts)\n"
            "2) The deals identified\n"
            "3) Current news"
            "4) Very detailed key insights (probability patterns, SHAP values & top drivers)\n"
            "5) Risks / Data gaps\n"
        )
    )

    text = (synth_prompt | llm | StrOutputParser()).invoke({
        "sql": state.get("sql_query", ""),
        "n_deals": len(state.get("retrieved_deals") or []),
        "exps": state.get("explanations") or [],
        "errs": state.get("errors") or [],
        "news": state.get("industry_news") or [],
    })

    return {"final_response": text}