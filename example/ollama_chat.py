#!/usr/bin/env python3
"""
Minimal-but-robust Ollama chat client.
- Uses native /api/chat when available (you have it).
- Falls back to /api/generate if /api/chat 404s (older builds).
- Supports non-stream and streaming.
- Optional REPL: keeps multi-turn conversation in memory.
"""

import os, sys, json, argparse, time
import requests
from typing import List, Dict, Any

DEFAULT_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

def http(method: str, url: str, **kw):
    r = requests.request(method, url, timeout=kw.pop("timeout", 600), **kw)
    # Helpful debug on error
    try:
        r.raise_for_status()
    except requests.HTTPError:
        sys.stderr.write(f"\nHTTP {r.status_code} for {url}\nBody: {r.text}\n")
        raise
    return r

def supports_chat(base: str) -> bool:
    # GET should return 405 if route exists; 404 means missing
    try:
        r = requests.get(f"{base}/api/chat", timeout=10)
        return r.status_code in (400, 405)  # 405 typical
    except requests.RequestException:
        return False

def list_models(base: str) -> List[str]:
    data = http("GET", f"{base}/api/tags", timeout=30).json()
    return [m["name"] for m in data.get("models", [])]

def send_chat(base: str, model: str, messages: List[Dict[str, str]],
              stream: bool = False, options: Dict[str, Any] = None):
    payload = {"model": model, "messages": messages, "stream": stream}
    if options: payload["options"] = options
    if not stream:
        data = http("POST", f"{base}/api/chat", json=payload).json()
        return data["message"]["content"]
    else:
        with requests.post(f"{base}/api/chat", json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line: 
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return

def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    # Simple flatten: User/Assistant/System into a plain prompt
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"{role.upper()}: {content}")
    out.append("ASSISTANT:")
    return "\n".join(out)

def send_generate(base: str, model: str, messages: List[Dict[str, str]],
                  stream: bool = False, options: Dict[str, Any] = None):
    prompt = messages_to_prompt(messages)
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if options: payload["options"] = options
    if not stream:
        data = http("POST", f"{base}/api/generate", json=payload).json()
        return data.get("response", "")
    else:
        with requests.post(f"{base}/api/generate", json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return

def run_once(args):
    base = args.base
    msgs = []
    if args.system:
        msgs.append({"role":"system","content":args.system})
    msgs.append({"role":"user","content":args.prompt})

    use_chat = supports_chat(base)
    sender = send_chat if use_chat else send_generate

    options = {}
    if args.temperature is not None: options["temperature"] = args.temperature
    if args.max_tokens is not None: options["num_predict"] = args.max_tokens
    if args.seed is not None: options["seed"] = args.seed

    if args.stream:
        for piece in sender(base, args.model, msgs, stream=True, options=options or None):
            print(piece, end="", flush=True)
        print()
    else:
        out = sender(base, args.model, msgs, stream=False, options=options or None)
        print(out)

def run_repl(args):
    base = args.base
    use_chat = supports_chat(base)
    sender = send_chat if use_chat else send_generate

    messages: List[Dict[str, str]] = []
    if args.system:
        messages.append({"role":"system","content":args.system})

    options = {}
    if args.temperature is not None: options["temperature"] = args.temperature
    if args.max_tokens is not None: options["num_predict"] = args.max_tokens
    if args.seed is not None: options["seed"] = args.seed

    print(f"REPL with model={args.model} (stream={args.stream})")
    print("Type 'exit' or 'quit' to leave.")
    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if user.lower() in ("exit","quit"):
            print("Bye.")
            return
        if not user: 
            continue

        messages.append({"role":"user","content":user})

        if args.stream:
            print("Model>", end=" ", flush=True)
            acc = []
            for piece in sender(base, args.model, messages, stream=True, options=options or None):
                acc.append(piece); print(piece, end="", flush=True)
            print()
            messages.append({"role":"assistant","content":"".join(acc)})
        else:
            out = sender(base, args.model, messages, stream=False, options=options or None)
            print(f"Model> {out}")
            messages.append({"role":"assistant","content":out})

def main():
    p = argparse.ArgumentParser(description="Ollama chat client")
    p.add_argument("--base", default=DEFAULT_BASE, help="Ollama base URL (default: %(default)s)")
    p.add_argument("--model", required=True, help="Model name (e.g., llama3-groq-tool-use:8b)")
    p.add_argument("--prompt", help="Single-turn prompt (omit when using --repl)")
    p.add_argument("--system", help="Optional system message")
    p.add_argument("--stream", action="store_true", help="Stream tokens")
    p.add_argument("--repl", action="store_true", help="Interactive multi-turn chat")
    p.add_argument("--temperature", type=float, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, help="Max new tokens (num_predict)")
    p.add_argument("--seed", type=int, help="Sampling seed for reproducibility")
    p.add_argument("--list-models", action="store_true", help="List local models and exit")
    args = p.parse_args()

    if args.list-models:
        names = list_models(args.base)
        print(json.dumps(names, indent=2))
        return

    if args.repl:
        run_repl(args)
    else:
        if not args.prompt:
            print("Error: --prompt is required when not using --repl", file=sys.stderr)
            sys.exit(2)
        run_once(args)

if __name__ == "__main__":
    main()
