#!/bin/bash
# cache_diagnostic.sh - Quick cache health check

echo "=== CACHE DIAGNOSTIC ==="
echo ""

echo "1. Cache file status:"
ls -lh data/prepared/llm_cache.db* 2>/dev/null || echo "   ❌ Cache not found!"
echo ""

echo "2. Cache contents:"
python3 << 'PYEOF'
import sqlite3
import os

db_path = "data/prepared/llm_cache.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
    queries = conn.execute("SELECT COUNT(DISTINCT query_hash) FROM judgments").fetchone()[0]

    if total == 0:
        print(f"   ❌ Cache is EMPTY (0 judgments)")
    elif total < 10000:
        print(f"   ⚠️  Cache is SMALL ({total:,} judgments)")
    else:
        print(f"   ✅ Cache is HEALTHY ({total:,} judgments, {queries:,} unique queries)")

    conn.close()
else:
    print("   ❌ Cache database does not exist!")
PYEOF
echo ""

echo "3. Recent cache activity:"
stat data/prepared/llm_cache.db 2>/dev/null | grep "Modify:" || echo "   No cache file"
echo ""

echo "=== RECOMMENDATION ==="
python3 << 'PYEOF'
import sqlite3
import os

db_path = "data/prepared/llm_cache.db"
if not os.path.exists(db_path):
    print("❌ CREATE NEW CACHE: Lower max_concurrent to 32 and run")
elif os.path.getsize(db_path) < 10 * 1024 * 1024:
    print("⚠️  REBUILD CACHE: Set max_concurrent=32, let it warm up")
else:
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
    conn.close()
    if total < 10000:
        print("⚠️  CONTINUE BUILDING: Cache needs more entries")
    else:
        print("✅ CACHE IS GOOD: You can use max_concurrent=64")
PYEOF
