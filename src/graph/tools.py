# Minimal professional comment: Tool layer implementing deterministic hybrid search and semantic querying.
import logging
from langchain_core.tools import tool
from src.model_loader import get_sentence_transformer
from src.db.clickhouse_client import get_client
from langgraph.prebuilt import InjectedState
from typing import Annotated, List, Optional, Any
from pydantic import BaseModel, Field

class CatalogFilter(BaseModel):
    field: str = Field(description="Strictly select from: 'brand', 'price_inr', 'rating', 'stock_qty', 'warranty_months', 'price_tier', 'in_stock', 'ram_gb', 'os', 'refresh_rate_hz', 'camera_resolution_mp', 'connectivity', 'cpu', 'screen_inches', 'screen_resolution', 'features', 'storage_gb', 'anc', 'battery', 'gpu', 'display_type'")
    operator: str = Field(description="Must be 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains', 'not_contains', 'in', or 'not_in'")
    value: Any = Field(description="The value to compare against. Use a list of values for 'in' and 'not_in' operators.")

class FAQFilter(BaseModel):
    field: str = Field(description="Strictly select from: 'category', 'tags'")
    operator: str = Field(description="Must be 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains', 'not_contains', 'in', or 'not_in'")
    value: Any = Field(description="The value to compare against.")

class CatalogSearchSchema(BaseModel):
    semantic_query: str = Field(description="Mandatory summary of the item being searched (e.g., 'OLED smartwatch'). MUST NOT BE EMPTY.")
    category: Optional[str] = Field(default=None, description="Strictly select from: 'Phone', 'Television', 'CPU', 'Headphone', 'Monitor', 'GPU', 'Tablet', 'Laptop', 'Watch', 'Speaker', 'Gaming Console', 'Camera'")
    filters: Optional[List[CatalogFilter]] = Field(default=None, description="Generic filters for catalog columns and specs. DO NOT put category here.")

class FAQSearchSchema(BaseModel):
    query: str = Field(description="The question or topic to search for")
    filters: Optional[List[FAQFilter]] = Field(default=None, description="Filters for 'category' or 'tags'. Use 'contains' or 'in' operator for tags.")

class TOSSearchSchema(BaseModel):
    query: str = Field(description="The topic to search for in terms of service")
    filters: Optional[List[FAQFilter]] = Field(default=None, description="Filters for 'category' or 'tags'. Use 'contains' or 'in' operator for tags.")

@tool(args_schema=FAQSearchSchema)
def get_support_faq(query: str, filters: Optional[List[Any]] = None) -> str:
    """Search this FIRST for general questions about policies, how things work, and common issues."""
    client = get_client()
    try:
        vector = get_sentence_transformer().encode(query).tolist()
        
        where_clauses = ["1=1"]
        params = {}
        
        op_map = {"eq": "=", "ne": "!=", "gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
        
        if filters:
            for i, spec in enumerate(filters):
                f_field = spec.field if hasattr(spec, 'field') else spec['field']
                f_operator = spec.operator if hasattr(spec, 'operator') else spec['operator']
                f_value = spec.value if hasattr(spec, 'value') else spec['value']
                
                op = op_map.get(f_operator)
                
                # Array Column (tags)
                if f_field == "tags":
                    if f_operator == "contains":
                        where_clauses.append(f"has(tags, %(v{i})s)")
                        params[f"v{i}"] = f_value
                    elif f_operator == "in" and isinstance(f_value, list):
                        where_clauses.append(f"hasAny(tags, %(v{i})s)")
                        params[f"v{i}"] = f_value
                # String Columns (category, etc.)
                else:
                    if f_operator == "contains":
                        where_clauses.append(f"lower({f_field}) LIKE lower(%(v{i})s)")
                        params[f"v{i}"] = f"%{str(f_value)}%"
                    elif f_operator == "in" and isinstance(f_value, list):
                        where_clauses.append(f"{f_field} IN %(v{i})s")
                        params[f"v{i}"] = tuple(f_value)
                    elif op:
                        where_clauses.append(f"{f_field} {op} %(v{i})s")
                        params[f"v{i}"] = f_value
            
        where_str = " AND ".join(where_clauses)
        
        # Lexical (Fuzzy) Search
        lexical_query = f"""
            SELECT faq_id, question, answer, related_tos, ngramDistance(lower(question || ' ' || answer), lower(%(pq)s)) as dist
            FROM company_faqs
            WHERE {where_str}
            ORDER BY dist ASC LIMIT 10
        """
        params["pq"] = query
        lex_res = client.query(lexical_query, parameters=params)
        lex_ranks = {r[0]: (idx, r) for idx, r in enumerate(lex_res.result_rows)} if lex_res.result_rows else {}
        
        # Vector Search
        vector_query = f"""
            SELECT faq_id, question, answer, related_tos, cosineDistance(question_embedding, {vector}) as dist
            FROM company_faqs
            WHERE {where_str}
            ORDER BY dist ASC LIMIT 10
        """
        vec_res = client.query(vector_query, parameters=params)
        vec_ranks = {r[0]: (idx, r) for idx, r in enumerate(vec_res.result_rows)} if vec_res.result_rows else {}
        
        rrf_scores = {}
        items_data = {}
        
        for pid, (rank, row) in lex_ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank))
            items_data[pid] = row
            
        for pid, (rank, row) in vec_ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank))
            if pid not in items_data:
                items_data[pid] = row
                
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:3]
        
        out = []
        for pid, score in top_items:
            r = items_data[pid]
            out.append(f"FAQ: {r[1]} | Answer: {r[2]}")
            if r[3]:
                tos_res = client.query(f"SELECT title, content FROM company_tos WHERE doc_id = '{r[3]}'")
                if tos_res.result_rows:
                    out.append(f"TOS Excerpt ({r[3]}): {tos_res.result_rows[0][0]} - {tos_res.result_rows[0][1]}")
                    
        return "\n".join(out).strip() if out else "No relevant FAQs found."
    except Exception as e:
        logging.getLogger(__name__).error(f"Support tool error: {e}")
        return "System error retrieving support docs."

@tool(args_schema=CatalogSearchSchema)
def search_catalog(semantic_query: str, category: Optional[str] = None, filters: Optional[List[Any]] = None) -> str:
    """Search the product catalog for availability, pricing, or tech recommendations."""
    client = get_client()
    try:
        vector = get_sentence_transformer().encode(semantic_query).tolist()
        
        where_clauses = ["1=1"]
        params = {}
        
        if category:
            where_clauses.append("category = %(cat)s")
            params["cat"] = category
            
        top_level_cols = {"brand", "category", "price_inr", "stock_qty", "rating", "warranty_months", "price_tier", "in_stock"}
        op_map = {"eq": "=", "ne": "!=", "gt": ">", "lt": "<", "gte": ">=", "lte": "<="}

        if filters:
            for i, spec in enumerate(filters):
                f_field = spec.field if hasattr(spec, 'field') else spec['field']
                
                # Map unambiguous LLM schema fields back to actual DB JSON keys
                if f_field == "camera_resolution_mp":
                    f_field = "camera_mp"
                elif f_field == "screen_resolution":
                    f_field = "resolution"
                    
                f_operator = spec.operator if hasattr(spec, 'operator') else spec['operator']
                f_value = spec.value if hasattr(spec, 'value') else spec['value']
                
                op = op_map.get(f_operator)
                
                if f_field in top_level_cols:
                    if f_operator == "contains":
                        where_clauses.append(f"lower({f_field}) LIKE lower(%(v{i})s)")
                        params[f"v{i}"] = f"%{str(f_value)}%"
                    elif f_operator == "not_contains":
                        where_clauses.append(f"lower({f_field}) NOT LIKE lower(%(v{i})s)")
                        params[f"v{i}"] = f"%{str(f_value)}%"
                    elif f_operator == "in" and isinstance(f_value, list):
                        where_clauses.append(f"lower({f_field}) IN %(v{i})s")
                        params[f"v{i}"] = tuple([str(x).lower() for x in f_value])
                    elif f_operator == "not_in" and isinstance(f_value, list):
                        where_clauses.append(f"lower({f_field}) NOT IN %(v{i})s")
                        params[f"v{i}"] = tuple([str(x).lower() for x in f_value])
                    elif op:
                        if f_field in ["brand", "category"]:
                            where_clauses.append(f"lower({f_field}) {op} lower(%(v{i})s)")
                            params[f"v{i}"] = str(f_value)
                        else:
                            where_clauses.append(f"{f_field} {op} %(v{i})s")
                            params[f"v{i}"] = f_value
                else:
                    if f_operator in ["eq", "ne", "contains", "not_contains"]:
                        like_op = "LIKE" if f_operator in ["eq", "contains"] else "NOT LIKE"
                        where_clauses.append(f"mapContains(specs, %(k{i})s) AND lower(specs[%(k{i})s]) {like_op} lower(%(v{i})s)")
                        params[f"k{i}"] = f_field
                        params[f"v{i}"] = f"%{str(f_value)}%"
                    elif f_operator in ["in", "not_in"] and isinstance(f_value, list):
                        in_op = "IN" if f_operator == "in" else "NOT IN"
                        where_clauses.append(f"mapContains(specs, %(k{i})s) AND lower(specs[%(k{i})s]) {in_op} %(v{i})s")
                        params[f"k{i}"] = f_field
                        params[f"v{i}"] = tuple([str(x).lower() for x in f_value])
                    elif op:
                        # Extract raw numbers from dirty strings ("24.2 MP" -> 24.2) before casting!
                        where_clauses.append(f"mapContains(specs, %(k{i})s) AND toFloat32OrZero(extract(specs[%(k{i})s], '([0-9.]+)')) {op} %(v{i})s")
                        params[f"k{i}"] = f_field
                        params[f"v{i}"] = float(f_value)
            
        where_str = " AND ".join(where_clauses)
        
        # Lexical (Fuzzy) Search using ngramDistance
        lexical_query = f"""
            SELECT product_id, name, brand, category, price_inr, stock_qty, specs, ngramDistance(lower(name), lower(%(pq)s)) as dist
            FROM product_catalog
            WHERE {where_str}
            ORDER BY dist ASC LIMIT 20
        """
        params["pq"] = semantic_query
        lex_res = client.query(lexical_query, parameters=params)
        lex_ranks = {r[0]: (idx, r) for idx, r in enumerate(lex_res.result_rows)} if lex_res.result_rows else {}
        
        # Vector Search
        vector_query = f"""
            SELECT product_id, name, brand, category, price_inr, stock_qty, specs, cosineDistance(embedding, {vector}) as dist
            FROM product_catalog
            WHERE {where_str}
            ORDER BY dist ASC LIMIT 20
        """
        vec_res = client.query(vector_query, parameters=params)
        vec_ranks = {r[0]: (idx, r) for idx, r in enumerate(vec_res.result_rows)} if vec_res.result_rows else {}
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        items_data = {}
        
        for pid, (rank, row) in lex_ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank))
            items_data[pid] = row
            
        for pid, (rank, row) in vec_ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank))
            if pid not in items_data:
                items_data[pid] = row
                
        # Sort by RRF score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:3] # Keep top 3 for LLM
        
        out = []
        for pid, score in top_items:
            r = items_data[pid]
            # r: [0=id, 1=name, 2=brand, 3=category, 4=price, 5=stock, 6=specs, 7=dist]
            out.append(f"Product: {r[1]} ({r[2]} {r[3]}) | Price: {r[4]} INR | Stock: {r[5]} | Specs: {r[6]}")
            
        return "\n".join(out).strip() if out else "No products matched the search criteria."
    except Exception as e:
        logging.getLogger(__name__).error(f"Catalog search error: {e}")
        return "System error retrieving catalog data."

@tool
def get_order_status(customer_id: str, order_id: Optional[str] = None) -> str:
    """Retrieve details about a customer's order history and delivery status. The customer_id is strictly required."""
    if not customer_id:
        return "Error: Customer ID required for order lookup."
    client = get_client()
    try:
        where_clause = "customer_id = %(cid)s"
        params = {"cid": customer_id}
        if order_id:
            where_clause += " AND order_id = %(oid)s"
            params["oid"] = order_id
            
        query = f"""
            SELECT order_id, order_date, status, items, final_amount_inr, delivery_address
            FROM order_history
            WHERE {where_clause}
            ORDER BY order_date DESC LIMIT 5
        """
        res = client.query(query, parameters=params)
        out = []
        if res.result_rows:
            for r in res.result_rows:
                out.append(f"Order: {r[0]} | Date: {r[1]} | Status: {r[2]} | Items: {r[3]} | Total: {r[4]} INR | Address: {r[5]}")
        return "\n".join(out).strip() if out else "No order records found."
    except Exception as e:
        logging.getLogger(__name__).error(f"Order status error: {e}")
        return "System error retrieving order details."

@tool
def get_customer_history(customer_id: str, search_query: Optional[str] = None) -> str:
    """Retrieve previous interactions, support tickets, and call summaries for the customer. If the user asks about a specific past topic, provide that as the search_query."""
    if not customer_id:
        return "Error: Customer ID required for history lookup."
    client = get_client()
    try:
        if search_query:
            vector = get_sentence_transformer().encode(search_query).tolist()
            query = f"""
                SELECT ticket_id, call_start_time, summary, cosineDistance(summary_embedding, {vector}) as dist
                FROM call_tickets
                WHERE customer_id = %(cid)s
                ORDER BY dist ASC LIMIT 3
            """
        else:
            query = f"""
                SELECT ticket_id, call_start_time, summary
                FROM call_tickets
                WHERE customer_id = %(cid)s
                ORDER BY call_start_time DESC LIMIT 3
            """
            
        res = client.query(query, parameters={"cid": customer_id})
        out = []
        if res.result_rows:
            for r in res.result_rows:
                out.append(f"Ticket: {r[0]} | Date: {r[1]} | Summary: {r[2]}")
        return "\n".join(out).strip() if out else "No previous support history found."
    except Exception as e:
        logging.getLogger(__name__).error(f"Customer history error: {e}")
        return "System error retrieving customer history."

@tool
def escalate_to_human(reason: str) -> str:
    """Use this tool to trigger a handoff to a human agent when the user explicitly requests to speak to a human, or requests an action you cannot perform."""
    return f"HANDOFF_TRIGGERED: {reason}"

@tool(args_schema=TOSSearchSchema)
def search_legal_tos(query: str, filters: Optional[List[Any]] = None) -> str:
    """Search this ONLY when the user explicitly asks for detailed legal terms or strict liabilities."""
    client = get_client()
    try:
        vector = get_sentence_transformer().encode(query).tolist()
        
        where_clauses = ["1=1"]
        params = {}
        
        op_map = {"eq": "=", "ne": "!=", "gt": ">", "lt": "<", "gte": ">=", "lte": "<="}
        
        if filters:
            for i, spec in enumerate(filters):
                f_field = spec.field if hasattr(spec, 'field') else spec['field']
                f_operator = spec.operator if hasattr(spec, 'operator') else spec['operator']
                f_value = spec.value if hasattr(spec, 'value') else spec['value']
                
                op = op_map.get(f_operator)
                
                # Array Column (tags)
                if f_field == "tags":
                    if f_operator == "contains":
                        where_clauses.append(f"has(tags, %(v{i})s)")
                        params[f"v{i}"] = f_value
                    elif f_operator == "in" and isinstance(f_value, list):
                        where_clauses.append(f"hasAny(tags, %(v{i})s)")
                        params[f"v{i}"] = f_value
                # String Columns (category, etc.)
                else:
                    if f_operator == "contains":
                        where_clauses.append(f"lower({f_field}) LIKE lower(%(v{i})s)")
                        params[f"v{i}"] = f"%{str(f_value)}%"
                    elif f_operator == "in" and isinstance(f_value, list):
                        where_clauses.append(f"{f_field} IN %(v{i})s")
                        params[f"v{i}"] = tuple(f_value)
                    elif op:
                        where_clauses.append(f"{f_field} {op} %(v{i})s")
                        params[f"v{i}"] = f_value
            
        where_str = " AND ".join(where_clauses)
        
        # Lexical (Fuzzy) Search
        lexical_query = f"""
            SELECT doc_id, title, content, ngramDistance(lower(content), lower(%(pq)s)) as dist
            FROM company_tos
            WHERE {where_str}
            ORDER BY dist ASC LIMIT 10
        """
        params["pq"] = query
        lex_res = client.query(lexical_query, parameters=params)
        lex_ranks = {r[0]: (idx, r) for idx, r in enumerate(lex_res.result_rows)} if lex_res.result_rows else {}
        
        # Vector Search
        vector_query = f"""
            SELECT doc_id, title, content, cosineDistance(content_embedding, {vector}) as dist
            FROM company_tos
            WHERE {where_str}
            ORDER BY dist ASC LIMIT 10
        """
        vec_res = client.query(vector_query, parameters=params)
        vec_ranks = {r[0]: (idx, r) for idx, r in enumerate(vec_res.result_rows)} if vec_res.result_rows else {}
        
        rrf_scores = {}
        items_data = {}
        
        for pid, (rank, row) in lex_ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank))
            items_data[pid] = row
            
        for pid, (rank, row) in vec_ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank))
            if pid not in items_data:
                items_data[pid] = row
                
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:2]
        
        out = []
        for pid, score in top_items:
            r = items_data[pid]
            out.append(f"TOS Document [{r[0]}]: {r[1]}\nLegal Text: {r[2]}")
            
        return "\n\n".join(out).strip() if out else "No matching legal terms found."
    except Exception as e:
        logging.getLogger(__name__).error(f"TOS tool error: {e}")
        return "System error retrieving TOS documents."

@tool
def check_complaint_eligibility(customer_id: str) -> str:
    """Use this tool FIRST when a user wants to file a complaint or raise a ticket. It checks if they are eligible."""
    client = get_client()
    try:
        res = client.query("SELECT ticket_id FROM complaint_tickets WHERE customer_id = %(cid)s AND status = 'raised'", parameters={"cid": customer_id})
        if res.result_rows:
            return "ELIGIBILITY_FAILED: User already has an active 'raised' ticket. Tell them they cannot raise another ticket until their current one is solved."
        else:
            return "ELIGIBILITY_PASSED: User is eligible. Briefly summarize their issue and ask for their EXPLICIT VERBAL CONFIRMATION (e.g. 'Should I go ahead and file this ticket?'). DO NOT file it yet."
    except Exception as e:
        logging.getLogger(__name__).error(f"check_complaint_eligibility error: {e}")
        return "System error checking ticket eligibility."

@tool
def raise_complaint_ticket(customer_id: str, title: str, issue: str, state: Annotated[dict, InjectedState]) -> str:
    """Use this tool ONLY AFTER the user has explicitly confirmed they want to raise the ticket."""
    client = get_client()
    try:
        session_id = state.get("session_id", "UNKNOWN")
        messages = state.get("messages", [])
        
        # Format call logs
        call_logs = ""
        for m in messages:
            if hasattr(m, 'type'):
                if m.type == "human":
                    call_logs += f"User: {m.content}\n"
                elif m.type == "ai":
                    if hasattr(m, 'content') and m.content:
                        call_logs += f"AI: {m.content}\n"
        
        customer_profile = state.get("customer_profile", {})
        c_name = customer_profile.get("name", "Unknown")
        c_phone = customer_profile.get("phone", "Unknown")
        
        insert_query = """
        INSERT INTO complaint_tickets (session_id, customer_id, customer_name, customer_phone, title, issue, call_logs, status)
        VALUES (%(sid)s, %(cid)s, %(cname)s, %(cphone)s, %(title)s, %(issue)s, %(logs)s, 'raised')
        """
        
        client.command(insert_query, parameters={
            "sid": session_id,
            "cid": customer_id,
            "cname": c_name,
            "cphone": c_phone,
            "title": title,
            "issue": issue,
            "logs": call_logs
        })
        
        return "TICKET_RAISED: The ticket has been successfully created. Tell the user it has been filed and an agent will review it."
    except Exception as e:
        logging.getLogger(__name__).error(f"raise_complaint_ticket error: {e}")
        return "System error raising the ticket."