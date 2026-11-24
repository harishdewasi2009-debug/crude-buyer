"""Multi-provider LLM client (OpenAI, Anthropic, Gemini, DeepSeek) over plain REST.

API keys are read from environment variables only; they are never stored in the DB,
never returned by any API route and are redacted from error messages.
"""
import json
import os
import re
import time
from dataclasses import dataclass

import requests

PROVIDERS = {
    "openai": dict(env=["OPENAI_API_KEY"], url="https://api.openai.com/v1/chat/completions", model="gpt-4o-mini"),
    "anthropic": dict(env=["ANTHROPIC_API_KEY"], url="https://api.anthropic.com/v1/messages", model="claude-sonnet-5-5"),
    "gemini": dict(env=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
                   url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                   model="gemini-2.0-flash"),
    "deepseek": dict(env=["DEEPSEEK_API_KEY"], url="https://api.deepseek.com/chat/completions", model="deepseek-chat"),
}


class LLMError(Exception):
    pass


@dataclass
class LLMResult:
    text: str
    data: object
    provider: str
    model: str


def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    depth, in_s, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_s:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_s = False
            continue
        if c == '"':
            in_s = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None


class LLM:
    def __init__(self, db):
        self.db = db

    # ---- configuration
    def key(self, p):
        for e in PROVIDERS[p]["env"]:
            if os.getenv(e):
                return os.getenv(e)
        return None

    def available(self):
        return [p for p in PROVIDERS if self.key(p)]

    def model(self, p):
        return (self.db.get_settings().get("models") or {}).get(p) or os.getenv(f"{p.upper()}_MODEL") or PROVIDERS[p]["model"]

    def order(self, prefer=None, exclude=None):
        st = self.db.get_settings()
        seq = [prefer] if prefer else []
        seq += [st.get("primary_provider")] + list(st.get("fallback_order") or []) + list(PROVIDERS)
        out = []
        for p in seq:
            if p in PROVIDERS and p not in out and p not in (exclude or []) and self.key(p):
                out.append(p)
        return out

    def status(self):
        st = self.db.get_settings()
        return {"providers": [{"provider": p, "configured": bool(self.key(p)), "model": self.model(p),
                               "env_var": PROVIDERS[p]["env"][0]} for p in PROVIDERS],
                "primary": st.get("primary_provider"), "fallback_order": st.get("fallback_order"),
                "cross_verify": st.get("cross_verify")}

    # ---- calls
    def complete(self, system, user, purpose, prefer=None, exclude=None, max_tokens=2500, json_out=True):
        providers = self.order(prefer, exclude)
        if not providers:
            raise LLMError("no AI provider configured (set an API key environment variable)")
        errors = []
        for p in providers:
            for attempt in (1, 2):
                try:
                    text, tin, tout = self._call(p, system, user, max_tokens)
                    self._usage(p, purpose, tin, tout, True, None)
                    return LLMResult(text, parse_json(text) if json_out else None, p, self.model(p))
                except Exception as e:
                    msg = self._redact(str(e))
                    self._usage(p, purpose, 0, 0, False, msg)
                    errors.append(f"{p}: {msg}")
                    if attempt == 1 and re.search(r"\b(429|500|502|503|504|timeout)\b", msg, re.I):
                        time.sleep(2)
                        continue
                    break
        raise LLMError("all providers failed: " + " | ".join(errors)[:600])

    def _redact(self, s):
        for p in PROVIDERS:
            k = self.key(p)
            if k:
                s = s.replace(k, "***")
        return s[:300]

    def _http(self, url, headers, body):
        r = requests.post(url, headers=headers, json=body, timeout=120)
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code} {r.text[:150]}")
        return r.json()

    def _call(self, p, system, user, max_tokens):
        key, model, url = self.key(p), self.model(p), PROVIDERS[p]["url"]
        if p in ("openai", "deepseek"):
            d = self._http(url, {"Authorization": f"Bearer {key}"},
                           {"model": model, "temperature": 0, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            u = d.get("usage", {})
            return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        if p == "anthropic":
            d = self._http(url, {"x-api-key": key, "anthropic-version": "2023-06-01"},
                           {"model": model, "max_tokens": max_tokens, "temperature": 0, "system": system,
                            "messages": [{"role": "user", "content": user}]})
            u = d.get("usage", {})
            return "".join(b.get("text", "") for b in d["content"]), u.get("input_tokens", 0), u.get("output_tokens", 0)
        d = self._http(url.format(model=model), {"x-goog-api-key": key},
                       {"systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": user}]}],
                        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}})
        u = d.get("usageMetadata", {})
        return ("".join(x.get("text", "") for x in d["candidates"][0]["content"]["parts"]),
                u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))

    def _usage(self, p, purpose, tin, tout, ok, err):
        price = (self.db.get_settings().get("prices") or {}).get(p) or {}
        cost = (tin * float(price.get("in", 0)) + tout * float(price.get("out", 0))) / 1e6
        self.db.ex("INSERT INTO llm_usage(ts,provider,model,purpose,in_tokens,out_tokens,cost,ok,error) "
                   "VALUES(datetime('now'),?,?,?,?,?,?,?,?)", (p, self.model(p), purpose, tin, tout, cost, int(ok), err))

    def test(self, p):
        try:
            r = self.complete("Reply with JSON only.", 'Return {"ok": true}', "connection-test", prefer=p, exclude=[
                x for x in PROVIDERS if x != p], max_tokens=50)
            return {"provider": p, "ok": bool(r.data), "model": r.model}
        except Exception as e:
            return {"provider": p, "ok": False, "error": self._redact(str(e))}
