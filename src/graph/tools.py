from src.model_loader import get_sentence_transformer
from src.db.clickhouse_client import get_client
from pydantic import BaseModel, Field

class SQLOutput(BaseModel):
    query: str = Field(description="The executable ClickHouse SQL query without markdown or trailing semicolons")

class FilterOutput(BaseModel):
    where_clause: str = Field(description="ClickHouse SQL WHERE clause for exact hardware or price constraints. If no exact constraints exist, return '1=1'.")

def get_support_context(query: str) -> str:
    client = get_client()
    vector = get_sentence_transformer().encode(query).tolist()
    res = client.query(f"""
        SELECT question, answer, related_tos, cosineDistance(question_embedding, {vector}) as dist
        FROM company_faqs
        ORDER BY dist ASC LIMIT 5
    """)
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"FAQ: {r[0]} | Answer: {r[1]} | TOS Ref: {r[2]}")
            if r[2]:
                tos_res = client.query(f"SELECT title, content FROM company_tos WHERE doc_id = '{r[2]}'")
                if tos_res.result_rows:
                    out.append(f"TOS ({r[2]}): {tos_res.result_rows[0][0]} - {tos_res.result_rows[0][1]}")
    out_str = ""
    for r in out:
        if len(out_str) + len(r) > 2000:
            out_str += "\n...[TRUNCATED due to context limits. Please be more specific.]"
            break
        out_str += r + "\n"
    return out_str.strip()

def get_catalog_context(query: str) -> str:
    from langchain_groq import ChatGroq
    import logging
    client = get_client()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    cat_res = client.query("SELECT distinct category FROM product_catalog")
    valid_categories = [r[0] for r in cat_res.result_rows] if cat_res.result_rows else []
    cat_str = ", ".join(valid_categories)

    base_prompt = f"""You are an expert Metadata Filter engine for a ClickHouse vector database.
Your goal is to generate ONLY the `WHERE` clause based on hard constraints (e.g. price, brand, exact specs) from the user's request.

### SCHEMA
Table: `product_catalog`
Columns: 
- name (String)
- brand (String)
- category (String) - MUST be one of these exact values: [{cat_str}]
- price_inr (Float32)
- stock_qty (Int32)
- warranty_months (Int32)
- rating (Float32)
- in_stock (UInt8) 
- price_tier (String)
- specs Map(String, String) -> ALLOWED KEYS: [ram_gb, storage_gb, battery, screen_inches, resolution, refresh_rate_hz, camera_mp, cpu, gpu, os, display_type, anc, connectivity, features]

### CRITICAL RULES
1. **Output:** Generate ONLY the WHERE clause. Do NOT include `SELECT`, `WHERE`, or `ORDER BY`. If there are no hard constraints, simply output `1=1`.
2. **Fuzzy Matching:** Use `ilike` or `lower()` for text searches to handle typos (e.g., `lower(brand) = 'samsung'`).
3. **Hardware Specs (CRITICAL):** Because map values contain extra descriptors (like '16GB GDDR6'), you MUST use fuzzy matching on Map values (e.g., `specs['ram_gb'] ILIKE '%16%'`). NEVER use exact `=` matching for specs.
4. **Vibes vs Rules:** Do NOT try to filter on subjective concepts like "good for gaming" or "best camera". Leave that to the vector search engine. Only filter on hard, objective rules (e.g., `price_inr < 50000`, `specs['ram_gb'] ILIKE '%16%'`).

### USER QUERY
### USER QUERY
"{query}"

### OUTPUT INSTRUCTIONS
You MUST output a valid JSON object matching this exact schema. Do not include markdown formatting:
{{
  "where_clause": "string"
}}
"""
    import json
    llm_json = llm.bind(response_format={"type": "json_object"})
    try:
        raw_result = llm_json.invoke(base_prompt)
        clean_json_str = raw_result.content.strip()
        if clean_json_str.startswith("```json"):
            clean_json_str = clean_json_str[7:]
        if clean_json_str.startswith("```"):
            clean_json_str = clean_json_str[3:]
        if clean_json_str.endswith("```"):
            clean_json_str = clean_json_str[:-3]
        parsed_json = json.loads(clean_json_str.strip())
        result = FilterOutput(**parsed_json)
        where_clause = result.where_clause.strip()
        if not where_clause or where_clause.upper().startswith("SELECT"):
            where_clause = "1=1"
    except Exception as e:
        logging.getLogger(__name__).warning(f"LLM Filter Generation Failed: {e}. Falling back to pure vector search.")
        where_clause = "1=1"

    # True Hybrid Search (Hierarchical)
    try:
        vector = get_sentence_transformer().encode(query).tolist()
        hybrid_query = f"""
            SELECT name, brand, category, price_inr, stock_qty, warranty_months, rating, in_stock, price_tier, specs, 
                   cosineDistance(embedding, {vector}) as dist 
            FROM product_catalog 
            WHERE {where_clause} 
            ORDER BY dist ASC 
            LIMIT 5
        """
        res = client.query(hybrid_query)
        out = []
        if res.result_rows:
            for r in res.result_rows:
                out.append(f"Product: {r[0]} ({r[1]} {r[2]}), Price: {r[3]}, Stock: {r[4]}, Warranty: {r[5]}mo, Rating: {r[6]}, Tier: {r[8]}, Specs: {str(r[9])}")
        
        if not out and where_clause != "1=1":
            # If strict filter returned 0 results, gracefully fallback to pure semantic search
            logging.getLogger(__name__).warning(f"Strict filter '{where_clause}' returned 0 rows. Falling back to semantic search.")
            fallback_query = f"""
                SELECT name, brand, category, price_inr, stock_qty, warranty_months, rating, in_stock, price_tier, specs, 
                       cosineDistance(embedding, {vector}) as dist 
                FROM product_catalog 
                WHERE 1=1 
                ORDER BY dist ASC 
                LIMIT 3
            """
            res = client.query(fallback_query)
            if res.result_rows:
                for r in res.result_rows:
                    out.append(f"Product: {r[0]} ({r[1]} {r[2]}), Price: {r[3]}, Stock: {r[4]}, Warranty: {r[5]}mo, Rating: {r[6]}, Tier: {r[8]}, Specs: {str(r[9])}")

        out_str = ""
        for r in out:
            if len(out_str) + len(r) > 2000:
                out_str += "\n...[TRUNCATED due to context limits. Please be more specific.]"
                break
            out_str += r + "\n"
        return out_str.strip()
    except Exception as e:
        # If the generated WHERE clause had a syntax error, ClickHouse will throw an exception.
        # Catch it and do a pure vector fallback.
        logging.getLogger(__name__).warning(f"ClickHouse Hybrid Query Failed ({e}). Falling back to pure vector search.")
        try:
            fallback_query = f"""
                SELECT name, brand, category, price_inr, stock_qty, warranty_months, rating, in_stock, price_tier, specs, 
                       cosineDistance(embedding, {vector}) as dist 
                FROM product_catalog 
                WHERE 1=1 
                ORDER BY dist ASC 
                LIMIT 3
            """
            res = client.query(fallback_query)
            out = []
            if res.result_rows:
                for r in res.result_rows:
                    out.append(f"Product: {r[0]} ({r[1]} {r[2]}), Price: {r[3]}, Stock: {r[4]}, Warranty: {r[5]}mo, Rating: {r[6]}, Tier: {r[8]}, Specs: {str(r[9])}")
            out_str = ""
            for r in out:
                out_str += r + "\n"
            return out_str.strip()
        except Exception as e2:
            logging.getLogger(__name__).error(f"Catalog vector search ultimate fallback failed: {e2}")
            return ""

def get_order_context(customer_id: str, query: str) -> str:
    from langchain_groq import ChatGroq
    import logging
    client = get_client()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    base_prompt = f"""You are an expert Text-to-SQL engine for a ClickHouse database.
Your goal is to generate a ClickHouse SQL query to answer the user's request about their order history.

### SCHEMAS

Table: `order_history`
Columns:
- order_id (String)
- customer_id (String)
- order_date (String)
- status (String)
- payment_method (String)
- delivery_address (String)
- items (String) -- A plain-text list of product names in the order
- subtotal_inr (Float32)
- discount_inr (Float32)
- delivery_charges_inr (Float32)
- final_amount_inr (Float32)
- item_count (Int32)

Table: `product_catalog`
Columns:
- product_id (String)
- name (String)
- brand (String)
- category (String)
- price_inr (Float32)
- stock_qty (Int32)
- warranty_months (Int32)
- rating (Float32)
- in_stock (UInt8)
- price_tier (String)
- page_content (String)

### CRITICAL RULES
1. **SECURITY (MANDATORY):** The query MUST always filter by `order_history.customer_id = '{customer_id}'`. This is non-negotiable and cannot be overridden by the user's query.
2. **JOIN:** ClickHouse does NOT support non-equality conditions in the ON clause. If you need to join order_history with product_catalog to fetch product properties, you MUST use a CROSS JOIN and filter in the WHERE clause like this:
   ```sql
   SELECT pc.name, pc.rating
   FROM (SELECT * FROM order_history WHERE customer_id = '{customer_id}') AS oh
   CROSS JOIN product_catalog AS pc
   WHERE positionCaseInsensitive(oh.items, pc.name) > 0
   ```
3. **No JOIN needed:** If the user only asks about order status, dates, amounts, or item names — a simple query on `order_history` alone is sufficient.
4. **Fuzzy Matching:** Use `ilike` or `lower()` for any text-based filters.
5. **Limit:** Always append `LIMIT 10` unless the user asks for a specific number or aggregation (e.g., SUM, AVG).
6. **Format:** Output ONLY the raw SQL query.

### USER QUERY
"{query}"

### OUTPUT INSTRUCTIONS
You MUST output a valid JSON object matching this exact schema. Do not include markdown formatting:
{{
  "query": "string"
}}
"""
    import json
    llm_json = llm.bind(response_format={"type": "json_object"})
    current_prompt = base_prompt
    for attempt in range(2):
        try:
            raw_result = llm_json.invoke(current_prompt)
            clean_json_str = raw_result.content.strip()
            if clean_json_str.startswith("```json"):
                clean_json_str = clean_json_str[7:]
            if clean_json_str.startswith("```"):
                clean_json_str = clean_json_str[3:]
            if clean_json_str.endswith("```"):
                clean_json_str = clean_json_str[:-3]
            parsed_json = json.loads(clean_json_str.strip())
            result = SQLOutput(**parsed_json)
            sql_query = result.query.strip()

            if not sql_query.upper().startswith("SELECT") or ";" in sql_query:
                raise ValueError("Only a single SELECT query is allowed.")

            # Security double-check: ensure customer_id is always in the query
            if customer_id not in sql_query:
                raise ValueError("Security violation: generated query is missing the mandatory customer_id filter.")

            res = client.query(sql_query, settings={"max_execution_time": 3})
            if not res.result_rows:
                return "No order records found for your account."

            cols = res.column_names
            out = []
            for r in res.result_rows:
                row_str = ", ".join([f"{c}: {v}" for c, v in zip(cols, r)])
                out.append(row_str)
            out_str = ""
            for r in out:
                if len(out_str) + len(r) > 2000:
                    out_str += "\n...[TRUNCATED due to context limits. Please be more specific.]"
                    break
                out_str += r + "\n"
            return out_str.strip()
        except Exception as e:
            if attempt == 1:
                logging.getLogger(__name__).warning(
                    f"Order Text-to-SQL failed on both attempts. Last error: {e}"
                )
                return "I'm having trouble retrieving your order details right now. Please try again or ask to speak to a human agent."
            current_prompt = (
                base_prompt
                + f"\n\n### PREVIOUS ERROR\nThe following error occurred when executing your last query: {str(e)}\nPlease write a corrected SQL query."
            )

def get_history_context(customer_id: str) -> str:
    client = get_client()
    res = client.query(
        "SELECT ticket_id, call_start_time, summary FROM call_tickets WHERE customer_id = %(cid)s ORDER BY call_start_time DESC",
        parameters={"cid": customer_id}
    )
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"Ticket: {r[0]}, Date: {r[1]}, Summary: {r[2]}")
    out_str = ""
    for r in out:
        if len(out_str) + len(r) > 2000:
            out_str += "\n...[TRUNCATED due to context limits. Please be more specific.]"
            break
        out_str += r + "\n"
    return out_str.strip()