import json
from config import make_llm, LAKEHOUSE_DB, OPEN_DEALS_TABLE, SHAP_TABLE
from state import AppState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from db_connection import engine  # Use SQLAlchemy engine for Fabric Lakehouse SQL endpoint
from typing import List, Dict, Any, Optional
from langchain_community.utilities import BingSearchAPIWrapper
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
import chainlit as cl

def orchestrator_node(state: AppState) -> AppState:
    """Parse the user query -> filters + which agents to run, with history support."""
    cl.run_sync(cl.Message(content="Orchestrator is analyzing the query and selecting agents...").send())
    llm = make_llm()

    parse_prompt = ChatPromptTemplate.from_template(
        (
            "You are a planner for a sales analytics assistant for EON Chemical Indonesia.\n"
            "Conversation so far:\n{history}\n\n"
            "Current user query: {query}\n\n"
            "1) Extract a concise SQL WHERE clause to filter deals in table `{table}`.\n"
            "   - If the current query does not specify filters, reuse or infer from conversation history.\n"
            "2) Decide which agents to call (include '3' if query mentions 'news', 'industry', 'recent updates', or similar):\n"
            "   - '1' Retrieval: query open deals (always include if analyzing or retrieving deals). This contains all relevant columns and information of the opportunity \n"
            "   - '2' Explain: explain ML probability using SHAP (include if deals retrieved or probability mentioned)\n"
            "   - '3' News: search for related industry news (include for news/industry context)\n\n"
            "Return STRICT JSON with keys: filters (string), agents (array of strings from ['1','2','3']).\n"
            "If unsure about news, err on including '3'. Default to all if ambiguous.\n"
            "3) For non-analysis questions from users, you are free to select the agents from (array of strings from ['1','2','3']) based on the question. Be smart in identifying analysis or fact based questions.\n"
            "4) If the query includes terms like 'analyze', 'study', 'investigate', etc., use all agents.\n\n"
        )
    )

    # Format conversation history into readable string
    history_str = "\n".join(
        [f"{m['role']}: {m['content']}" for m in state.get("history", [])]
    )

    chain = parse_prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "query": state["user_query"],
        "table": OPEN_DEALS_TABLE,
        "history": history_str,
    }).strip()

    # LOGGING
    print(f"\n=== Orchestrator Debug ===")
    print(f"User Query: {state['user_query']}")
    print(f"History: {history_str}")
    print(f"Raw LLM Output: {raw}")
    print("=========================\n")

    parsed_filters, agents, errors = "", ["1", "2", "3"], []  # safe defaults

    try:
        data = json.loads(raw)
        parsed_filters = data.get("filters", "") or ""
        agents = data.get("agents", ["1", "2", "3"]) or ["1", "2", "3"]

        # validate agents
        if any(a not in {"1", "2", "3"} for a in agents):
            errors.append(f"Invalid agent id in {agents}; defaulting to ['1','2','3'].")
            agents = ["1", "2", "3"]

    except Exception as e:
        errors.append(f"Failed to parse planner JSON: {e}. Raw: {raw}")

    # ✅ carry-over filters if empty
    if not parsed_filters and state.get("parsed_filters"):
        parsed_filters = state["parsed_filters"]

    print(f"Selected Agents: {agents}")
    print(f"Filters: {parsed_filters}")

    # ✅ update conversation history
    new_history = state.get("history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "system", "content": f"Filters: {parsed_filters}, Agents: {agents}"}
    ]

    return {
        "parsed_filters": parsed_filters,
        "agents_to_call": agents,
        "errors": state.get("errors", []) + errors,
        "history": new_history,
    }



def agent1_node(state: AppState) -> AppState:
    """Agent 1: NL → SQL → T-SQL retrieval from Fabric Lakehouse SQL endpoint."""
    if "1" not in state.get("agents_to_call", []):
        print("⚠️ Agent 1 not scheduled to run.")
        return {}

    # LOGGING
    print("\n=== Agent 1 Triggered ===")
    cl.run_sync(cl.Message(content="Agent 1 is retrieving relevant deal...").send())

    llm = make_llm()

    # --- Fetch schema ---
    schema_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{OPEN_DEALS_TABLE.split('.')[-1]}'
    """
    schema_df = pd.read_sql(schema_query, engine)
    schema_cols = set(schema_df["COLUMN_NAME"].str.lower())

    # Always-required base columns
    base_cols = [
        "opportunity_number",
        "win_probability",
        "predicted_outcome",
        "expected_value",
        "topic",
        "opportunity_type",
        "op_created_on",
        "customer_industry",
        "customer_name",
        "process_stages",
    ]

    # --- Safely flatten history ---
    def extract_text(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
            )
        return str(content)

    history_text = " ".join(extract_text(m) for m in state.get("history", []))

    # Extract potential extra cols from query + history
    query_text = (state.get("user_query", "") + " " + history_text).lower()
    extra_cols = [col for col in schema_cols if col in query_text]

    # Merge and deduplicate
    select_cols = list(dict.fromkeys(base_cols + extra_cols))

    # Build schema description string for LLM prompt
    schema_description = ", ".join(
        f"{row.COLUMN_NAME} ({row.DATA_TYPE})" for _, row in schema_df.iterrows()
    )

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
            "1. ONLY SELECT COLUMNS AVAILABLE IN THE SCHEMA. You MUST explicitly list the columns in the SELECT clause.\n"
            "2. You MUST always include these base columns: {base_cols}.\n"
            "3.If the user query or history refers to other columns in the schema, include them in the SELECT as well.\n"
            "4. DO NOT use `SELECT *`.\n"
            "5. The table already contains only open deals — do NOT filter by op_status.\n"
            "6. SQL must be valid T-SQL. Do NOT add code fences.\n\n"
            "7. Always filter using `opportunity_number` if user mentions a specific deal. Never use opportunity_id, deal_id, or other IDs.  If the filters input ({filters}) contains these columns, replace them with `opportunity_number`.\n\n"
            "8. Apply all relevant filters:\n"
            "   - Include filters explicitly mentioned in the user query (e.g., region, industry, opportunity_type, win_probability thresholds, process_stages, discount_percent, op_eon_product, op_product_family ).\n"
            "   - Combine with filters inferred from conversation history  or `{filters}` ONLY IF RELEVANT.\n"
            "   - Apply sorting or or DESC or ASC or  TOP N if requested (e.g., 'top 3 opportunities', 'highest probability in region').\n"
            "   - Ensure no required filter is omitted; only include rows matching all specified criteria.\n"
            "9) All deals are OPEN and non-deleted. DO NOT INCLUDE THESE FILTERS like is_closed, is_deleted, is_open, op_status, is_closed, status and something similar. If the filters input ({filters}) contains these columns, REMOVE THEM`. DON'T BE STUPID, BE SMART.\n\n"
            
            ''

            "filters: {filters}"
            "User query: {query}\n"
            "Conversation history: {history}\n"
        )
    )

    sql_str = (sql_prompt | llm | StrOutputParser()).invoke({
        "table": OPEN_DEALS_TABLE,
        "filters": state.get("parsed_filters", ""),
        "base_cols": ", ".join(base_cols),   # only the base set
        "query": state.get("user_query", ""),
        "history": history_text,
    }).strip()

    # --- Minimal safeguard fallback ---
    if not sql_str.lower().startswith("select"):
        sql_str = f"SELECT {', '.join(select_cols)} FROM {OPEN_DEALS_TABLE}"
        if state.get("parsed_filters"):
            sql_str += f" WHERE {state['parsed_filters']}"

    print("\n=== Generated SQL ===")
    print(sql_str)
    print("====================\n")

    errors = state.get("errors", [])
    deals: List[Dict[str, Any]] = []
    try:
        df = pd.read_sql(sql_str, engine)
        deals = df.to_dict(orient="records")
        print(f"✅ Retrieved {len(deals)} deals.")
    except Exception as e:
        errors.append(f"SQL execution failed: {e}. SQL: {sql_str}")

    print("=== End Agent 1 ===\n")
    return {
        "sql_query": sql_str,
        "retrieved_deals": deals,
        "errors": errors,
    }



def agent2_node(state: AppState) -> AppState:
    """Agent 2: Explain using SHAP values (from shap_values table)."""
    if "2" not in state.get("agents_to_call", []):
        print("⚠️ Agent 2 not scheduled to run.")
        return {}
    cl.run_sync(cl.Message(content="Agent 2 is retrieving machine learning results...").send())

    # LOGGING
    print("\n=== Agent 2 Triggered ===")

    deals = state.get("retrieved_deals") or []
    if not deals:
        print("⚠️ No deals for SHAP explanation.")
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
    print(f"✅ Fetched {len(shap_df)} SHAP rows.")

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

    print(f"✅ Generated {len(explanations)} explanations.")
    print("=== End Agent 2 ===\n")
    return {"explanations": explanations}


def agent3_node(state: AppState) -> AppState:
    if "3" not in state.get("agents_to_call", []):
        print("⚠️ Agent 3 not scheduled to run.")
        return state
    cl.run_sync(cl.Message(content="Agent 3 is retrieving web search...").send())

    deals = state.get("retrieved_deals") or []
    if not deals:
        print("⚠️ No deals found; skipping news search.")
        state["news"] = []
        return state

    # Group unique customers by their corresponding industry
    from collections import defaultdict
    industry_customers = defaultdict(set)
    for d in deals:
        industry = str(d.get("customer_industry", "")).strip().lower()
        customer = str(d.get("customer_name", "")).strip()
        if industry and customer and industry not in ["Others", "Others 2"]:
            industry_customers[industry].add(customer)

    # Convert sets to lists for joining
    for ind in industry_customers:
        industry_customers[ind] = list(industry_customers[ind])

    industry_news = []

    try:
        project = AIProjectClient(
            credential=DefaultAzureCredential(),
            endpoint="https://aiserver1234.services.ai.azure.com/api/projects/SALES_CHATBOT_POC"
        )
        agent = project.agents.get_agent("asst_ZkzK0inGhhkrf5NFXOsHetRU")
    except Exception as e:
        state["news"] = [{"error": str(e)}]
        return state

    # Loop over each industry
    for industry, customers in industry_customers.items():
        if not customers:
            continue

        # Loop over each corresponding customer for this industry (one-by-one)
        for customer in customers:
            query = f"Give me the latest news and trends about {industry} industry in Indonesia. Also include any recent trends related to customer {customer} if available."

            try:
                # Perform the search using the Azure AI agent
                thread = project.agents.threads.create()
                project.agents.messages.create(thread_id=thread.id, role="user", content=query)

                run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

                reply = None
                if run.status != "failed":
                    messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.DESCENDING)
                    for message in messages:
                        if message.role == "assistant" and message.text_messages:
                            reply = message.text_messages[-1].text.value.strip()
                            break

                snippets = [reply] if reply else ["No recent news found."]

                industry_news.append({
                    "industry": industry,
                    "customer": customer,
                    "query": query,
                    "snippets": snippets
                })

            except Exception as e:
                error_msg = f"News search failed: {str(e)}"
                print(f"Error in Agent 3 for {industry} and {customer}: {error_msg}")
                industry_news.append({
                    "industry": industry,
                    "customer": customer,
                    "query": query,
                    "error": error_msg,
                    "snippets": []
                })

    state["news"] = industry_news
    print(f"✅ Fetched news for {len(industry_news)} industry-customer pairs (one-by-one, matched).")
    return state


def synthesizer_node(state: AppState) -> AppState:
    """Compile the final response for the user, adaptively handling analysis or general queries."""
    print("\n=== Synthesizer Triggered ===")
    cl.run_sync(cl.Message(content="Synthesizer is summarizing output...").send())
    
    news_list = state.get("news") or []
    deals = state.get("retrieved_deals") or []
    explanations = state.get("explanations") or []

    # --- Format news for LLM (join snippets into readable lines, including customer)
    formatted_news = []
    for item in news_list:
        if "snippets" in item and item["snippets"]:
            snippets_text = "\n".join([s for s in item["snippets"] if s])
            customer_str = item.get("customer", "General")
            formatted_news.append(f"{item['industry']} - {customer_str}: {snippets_text}")
        else:
            customer_str = item.get("customer", "General")
            formatted_news.append(f"{item['industry']} - {customer_str}: {item.get('error', 'No data')}")
    news_str = "\n".join(formatted_news) if formatted_news else "None"

    # --- Format deals & explanations naturally (avoid JSON dumps to keep line breaks)
    deals_str = "\n".join(str(d) for d in deals) if deals else "None"
    exps_str = "\n".join(str(e) for e in explanations) if explanations else "None"

    print(f"Raw state debug: Deals={len(deals)}, Explanations={len(explanations)}, News={len(news_list)}")

    llm = make_llm()
    synth_prompt = ChatPromptTemplate.from_template(
        (
            "You are the final synthesizer for a sales analytics assistant at EON Chemical Indonesia, a leading chemical company.\n"
            "Conversation context:\n"
            "- Original user query: {user_query}\n"
            "- SQL used: {sql}\n"
            "- #Deals: {n_deals}\n"
            "- Deals: {deals}\n"
            "- Explanations: {exps}\n"
            "- Any errors: {errs}\n"
            "- Industry and customer news: {news_str}\n\n"

            "=== Instructions ===\n"
            "1. Determine if the user query is **analysis-related** "
            "(e.g., deal performance, win probability, SHAP outputs, or industry news).\n"
            "   - If YES → follow the structured manager note format below.\n"
            "   - If NO → answer the user query directly and concisely.\n"
            "   - If no matching deals, state clearly and suggest next steps.\n\n"

            "=== Structured Output (for analysis queries) ===\n"
            "1) **What we looked at** (filters, counts)\n"
            "2) **Deal(s) identified** (table with key deal details) — Use VALID MARKDOWN TABLE SYNTAX ONLY. Do NOT wrap in code blocks (no ```), do NOT use HTML <table>, and do NOT use ASCII art or preformatted text. Example:\n"
                "| Field | Value |\n"
                "|-------|-------|\n"
                "| Opp   | Val   |\n"
                "Ensure all rows have the exact same number of pipes (|), no extra spaces, and align columns properly.\n"
            "3) **Model output** (probability & SHAP driver explanations)\n"
            "4) **Industry and Customer news (Indonesia)**\n"
            "5) **Detailed insights** (probability patterns, top drivers, reasoning)\n"
            "6) **Action steps / Recommendations**\n"
            "7) **Risks / Data gaps**\n\n"

            "=== Enhanced Guidelines for Recommendations (Section 6) ===\n"
            "- Follow the **MEDDIC methodology** (Metrics, Economic Buyer, Decision Criteria, Decision Process, Identify Pain, Champion) to structure recommendations for EON Chemical Indonesia's B2B sales team.\n"
            "- Align with **chemical industry benchmarks** (e.g., APQC: 20-30% stage progression rate, 45-60 day trial-to-close; Solomon: 25-35% close rate for high-value deals) to quantify gaps and set targets.\n"
            "- Base recommendations on: customer/industry news, SHAP value drivers, deal details (e.g., process_stages, opportunity_type), and ML outputs (e.g., win_probability). Cross-reference with benchmarks (e.g., 'Win prob of 73% is below 75% industry avg; address SHAP driver deal_age_days to close gap').\n"
            "- Provide **deep, non-shallow, professional strategies** tailored for B2B chemical sales, emphasizing:\n"
            "  - **Regulatory compliance**: Address Indonesia’s chemical/mining regulations (e.g., sustainability, emissions).\n"
            "  - **Technical validation**: Detailed plans for sampling, pilots, or proof-of-concepts with measurable criteria.\n"
            "  - **Relationship building**: Multi-stakeholder engagement (e.g., technical, procurement, C-suite) with long-term trust focus.\n"
            "  - **Data-driven insights**: Use analytics (e.g., SHAP, CRM data) for personalized proposals and forecasting.\n"
            "- Include strategies to advance from the current process_stage to the next, benchmarked against industry standards:\n"
            "  - **Qualify (Early stage)**: Assess lead fit (e.g., align EON’s chemical solutions with customer needs; benchmark: 40-50% qualification-to-propose rate per APQC). Identify Economic Buyer and Decision Criteria via stakeholder mapping.\n"
            "  - **Propose (Mid stage)**: Deliver tailored proposals (e.g., use SHAP drivers like revenue tier for ROI projections; benchmark: 60% proposal acceptance). Establish Champion to advocate internally.\n"
            "  - **Trial (Late stage)**: Facilitate rigorous testing (e.g., pilots with defined KPIs like yield improvement; benchmark: 45-day trial-to-close vs. 60-day avg). Confirm Decision Process and address Pain points.\n"
            "  - **Closing**: Negotiate strategically (e.g., leverage SHAP factors to counter objections; benchmark: 25-35% close rate per Solomon). Secure Economic Buyer sign-off and track Metrics (e.g., revenue impact).\n"
            "- Make suggestions **SMART (Specific, Measurable, Achievable, Relevant, Time-bound)**, integrating news (e.g., 'Given recent mining regulations affecting {{customer}}, propose EON’s eco-friendly formulations in a 2-week pilot to move from Propose to Trial, targeting 15% prob uplift. Owner: Sales Lead; Metric: Stage progression within 14 days.').\n"
            "- Use bullet points with 3-5 detailed strategies per deal, each including: benchmark reference, MEDDIC alignment, rationale, steps, owner, timeline, and success metric.\n\n"

            "=== Style ===\n"
            "- Be concise yet insightful and professional—advise as an expert at EON Chemical Indonesia.\n"
            "- Use **bolding**, tables that can be displayed in Chainlit App, and bullet points.\n"
            "- Use MEDDIC approach (Metric, Economic buyer, Decision criteria, Decision process, Identify plan, Champion)\n"
            "- Include emojis for readability where appropriate (e.g., ✅ for positives).\n"
            "- Output ONLY plain Markdown text — no JSON, no code fences."
        )
    )

    text = (synth_prompt | llm | StrOutputParser()).invoke({
        "user_query": state.get("user_query", ""),
        "sql": state.get("sql_query", ""),
        "n_deals": len(deals),
        "deals": deals_str,
        "exps": exps_str,
        "errs": state.get("errors") or [],
        "news_str": news_str,
    })

    print("✅ Synthesis complete.")
    print("=======================\n")

    # Send nicely formatted output to Chainlit
    cl.run_sync(cl.Message(content=text).send())

    return {"final_response": text}