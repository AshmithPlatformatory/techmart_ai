from src.model_loader import get_sentence_transformer
from src.db.clickhouse_client import get_client

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
    out_str = "\n".join(out)
    if len(out_str) > 15000:
        return out_str[:15000] + "\n...[TRUNCATED due to context limits. Please be more specific.]"
    return out_str

def get_catalog_context(query: str) -> str:
    from langchain_groq import ChatGroq
    client = get_client()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = f"""You are a Text-to-SQL generator for a ClickHouse database.
Table schema: product_catalog (name String, brand String, category String, price_inr Float32, stock_qty Int32, warranty_months Int32, rating Float32, in_stock UInt8, price_tier String, page_content String)
Generate a ClickHouse SQL query to answer this user query: "{query}"
Only output the raw SQL query. Do not add markdown blocks like ```sql. Do not add any explanation."""
    
    try:
        sql_query = llm.invoke(prompt).content.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        if not sql_query.upper().startswith("SELECT") or ";" in sql_query:
            raise ValueError("Only a single SELECT query is allowed.")
        
        res = client.query(sql_query, settings={"max_execution_time": 2})
        if not res.result_rows:
            raise ValueError("Empty SQL result. Fallback to Vector Search.")
        
        cols = res.column_names
        out = []
        for r in res.result_rows:
            row_str = ", ".join([f"{c}: {v}" for c, v in zip(cols, r)])
            out.append(row_str)
        out_str = "\n".join(out)
        if len(out_str) > 15000:
            return out_str[:15000] + "\n...[TRUNCATED due to context limits. Please be more specific.]"
        return out_str
    except Exception as e:
        vector = get_sentence_transformer().encode(query).tolist()
        res = client.query(f"SELECT name, brand, category, price_inr, stock_qty, warranty_months, rating, in_stock, price_tier, page_content, cosineDistance(embedding, {vector}) as dist FROM product_catalog ORDER BY dist ASC LIMIT 2")
        out = []
        if res.result_rows:
            for r in res.result_rows:
                out.append(f"Product: {r[0]} ({r[1]} {r[2]}), Price: {r[3]}, Stock: {r[4]}, Warranty: {r[5]}mo, Rating: {r[6]}, Tier: {r[8]}, Details: {r[9]}")
        out_str = "\n".join(out)
        if len(out_str) > 15000:
            return out_str[:15000] + "\n...[TRUNCATED due to context limits. Please be more specific.]"
        return out_str

def get_order_context(customer_id: str) -> str:
    client = get_client()
    res = client.query(
        "SELECT order_id, order_date, status, items, final_amount_inr FROM order_history WHERE customer_id = %(cid)s ORDER BY order_date DESC",
        parameters={"cid": customer_id}
    )
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"Order: {r[0]}, Date: {r[1]}, Status: {r[2]}, Total: {r[4]}, Items: {r[3]}")
    out_str = "\n".join(out)
    if len(out_str) > 15000:
        return out_str[:15000] + "\n...[TRUNCATED due to context limits. Please be more specific.]"
    return out_str

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
    out_str = "\n".join(out)
    if len(out_str) > 15000:
        return out_str[:15000] + "\n...[TRUNCATED due to context limits. Please be more specific.]"
    return out_str