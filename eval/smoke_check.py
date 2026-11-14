"""Smoke test: asks ~40 questions and runs the leave-request flows against a RUNNING server.

    .\run_api.ps1                         # terminal 1: start the server (port 8000)
    python -m eval.smoke_check 8000         # terminal 2: run the checks

Uses the real model if the server has a key (this spends a little of your Groq quota).
A few answers can vary run to run with a real model; read the failures before worrying.
"""
import sys, json, httpx
sys.path.insert(0, ".")
from eval.smoke_cases import Q, LEAVE, WEAK
norm = lambda s: s.replace("\u202f"," ").replace("\u00a0"," ").replace("\u2011","-").replace("\u2010","-").replace("\u2019","'").lower()
has = lambda a, m: any(norm(alt) in a for alt in m.split("|"))
c = httpx.Client(base_url="http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8000"), timeout=120)
print("model:", c.get("/health").json()["llm"])
passed = failed = 0; fails = []
def rec(ok, label):
    global passed, failed
    passed += ok; failed += (not ok)
    if not ok: fails.append(label)
for role, q, must, mustnot in Q:
    r = c.post("/ask", json={"question": q}, headers={"X-User-Roles": role} if role else {})
    a = norm(r.json()["answer"]) if r.status_code == 200 else "HTTP %d" % r.status_code
    ok = r.status_code == 200 and all(has(a, m) for m in must) and not any(has(a, m) for m in mustnot)
    rec(ok, f"[{role or 'employee'}] {q} -> {a[:90]}")
for i, (msg, status, texts) in enumerate(LEAVE):
    r = c.post("/agent", json={"message": msg, "user": "checker"}).json()
    blob = norm(json.dumps(r))
    ok = r.get("status") == status and all(norm(t) in blob for t in texts)
    if status == "pending_approval" and i == 0:
        d = c.post("/agent/approve", json={"thread_id": r["thread_id"], "approve": True}).json()
        ok = ok and d["status"] == "submitted"
    rec(ok, f"leave: {msg} -> {r.get('status')}")
r = c.post("/agent", json={"message": "I want to take 2 days of leave", "user": "checker2"}).json()
before = len(c.get("/leave-requests").json())
d = c.post("/agent/approve", json={"thread_id": r["thread_id"], "approve": False}).json()
rec(d["status"] == "rejected" and len(c.get("/leave-requests").json()) == before, "reject saves nothing")
rec(c.post("/agent/approve", json={"thread_id": "nope", "approve": True}).status_code == 404, "unknown thread 404")
rec(c.post("/ask", json={"question": "x"}).status_code == 422, "too short 422")
rec(c.post("/ask", json={"question": "How many leave days?", "mode": "magic"}).status_code == 422, "bad mode 422")
for m in ("vector", "bm25", "hybrid", "hybrid_rerank"):
    r = c.post("/ask", json={"question": "How many days of annual leave do I get each year?", "mode": m}).json()
    rec("24" in r["answer"], "search mode " + m)
st = c.get("/stream", params={"q": "How many sick days do I get?"}); rec(st.status_code == 200 and "10" in st.text, "streaming")
rec(c.get("/").status_code == 200, "website page served")
total = len(Q) + len(LEAVE) + 8
print(f"\nTERMINAL RESULT: {passed} passed, {failed} failed  (of {passed + failed} checks, {len(Q)} questions)")
for f in fails: print("  FAILED:", f)
print("\nknown weak spots (informational):")
for role, q, must in WEAK:
    a = norm(c.post("/ask", json={"question": q}).json()["answer"])
    print(("  answered " if all(has(a, m) for m in must) else "  still misses ") + f"| {q} -> {a[:70]}")
