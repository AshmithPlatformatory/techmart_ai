import clickhouse_connect
from src.core.config import CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD

_client = None

def get_client():
    """Return a module-level singleton ClickHouse client.

    Opening a new TLS connection per query adds ~300ms latency and can
    block the asyncio event loop. Re-using a single client avoids both.
    clickhouse-connect clients are thread-safe.
    """
    global _client
    if _client is None:
        _client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            secure=True
        )
    return _client