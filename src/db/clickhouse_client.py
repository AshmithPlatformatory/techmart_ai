import threading
import clickhouse_connect
from src.core.config import CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD

_thread_local = threading.local()

def get_client():
    """Return a thread-local ClickHouse client.

    Clickhouse-connect clients cannot execute concurrent queries 
    on the exact same session instance. Using threading.local() 
    ensures parallel LangGraph workers get their own safe connections.
    """
    if not hasattr(_thread_local, "client"):
        import urllib3
        # Pool manager is strictly bound to this thread's client
        pool_mgr = urllib3.PoolManager(maxsize=10)
        _thread_local.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            secure=True
        )
    return _thread_local.client