"""
n8n용 로컬 HTTP API 서버 (멀티스레드)
n8n HTTP Request 노드에서 Python 스크립트를 호출하기 위한 서버
실행: python src/korean/api_server.py
기본 포트: 8765
"""
import subprocess
import sys
import os
import json
import socketserver
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PYTHON   = sys.executable

SCRIPTS = {
    "/fetch-reports": os.path.join(BASE_DIR, "data", "fetch_reports.py"),
    "/fetch-news":    os.path.join(BASE_DIR, "data", "fetch_news.py"),
    "/build-index":   os.path.join(BASE_DIR, "rag", "indexer.py"),
}

KST = timezone(timedelta(hours=9))


def _fmt_signal(v: int) -> str:
    return "📈매수" if v == 1 else ("📉매도" if v == -1 else "➖중립")


def _build_agent_message(data: dict) -> dict:
    """run_daily.py JSON → Telegram 메시지 + buy_count 반환 (n8n Code 노드 대체)"""
    buy_count   = data.get("buy_count", 0)
    total       = data.get("total", 0)
    buy_signals = data.get("buy_signals", [])
    now         = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    message = ""
    if buy_count > 0:
        lines = []
        for s in buy_signals[:10]:
            lines.append(
                f"💡 {s.get('stock_name','?')}({s.get('ticker','?')}) "
                f"기술{_fmt_signal(s.get('tech_signal',0))} "
                f"펀더{_fmt_signal(s.get('fund_signal',0))} "
                f"감성{_fmt_signal(s.get('sent_signal',0))}"
            )
        tail = f"\n... 외 {buy_count - 10}종목" if buy_count > 10 else ""
        message = (
            f"🚀 AlphaFin 매수 신호 감지!\n"
            f"📅 {now}\n"
            f"총 {total}종목 중 {buy_count}종목 매수\n\n"
            + "\n".join(lines)
            + tail
            + "\n\n⚠️ 학술 목적 전용 · 실제 투자 조언 아님"
        )

    return {**data, "message": message, "buy_count": buy_count}


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """요청마다 별도 스레드 — 무거운 스크립트 실행 중에도 /health 응답 가능"""
    daemon_threads    = True   # 메인 프로세스 종료 시 스레드 자동 정리
    allow_reuse_address = True # 재시작 시 "Address already in use" 방지


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[API] {self.path} — {args[0]}", flush=True)

    def log_error(self, format, *args):
        print(f"[API ERROR] {format % args}", flush=True)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self._json(200, {"status": "ok"})
            return

        # 에이전트 분석 — JSON 파싱 + 메시지 포맷까지 처리 (n8n Code 노드 불필요)
        if path == "/run-agents":
            self._run_agents()
            return

        if path not in SCRIPTS:
            self._json(404, {"error": f"unknown path: {path}"})
            return

        self._run_script(SCRIPTS[path])

    def _run_script(self, script: str, extra: list = None):
        name = os.path.basename(script)
        print(f"[API] 실행 시작: {name}", flush=True)
        try:
            result = subprocess.run(
                [PYTHON, script] + (extra or []),
                capture_output=True, text=True, timeout=3600,
                cwd=BASE_DIR,
            )
            ok = result.returncode == 0
            print(f"[API] 완료: {name} (exit={result.returncode})", flush=True)
            self._json(200, {
                "ok":     ok,
                "stdout": result.stdout[-5000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "code":   result.returncode,
            })
        except subprocess.TimeoutExpired:
            print(f"[API] 타임아웃: {name}", flush=True)
            self._json(200, {"ok": False, "error": "timeout (3600s)"})
        except Exception as e:
            print(f"[API] 예외: {e}", flush=True)
            self._json(500, {"ok": False, "error": str(e)})

    def _run_agents(self):
        script = os.path.join(BASE_DIR, "agent", "run_daily.py")
        print("[API] 실행 시작: run_daily.py", flush=True)
        try:
            result = subprocess.run(
                [PYTHON, script, "--output", "json"],
                capture_output=True, text=True, timeout=3600,
                cwd=BASE_DIR,
            )
            print(f"[API] 완료: run_daily.py (exit={result.returncode})", flush=True)

            # stdout JSON 파싱 — 경고 메시지가 앞에 붙어도 JSON 추출
            stdout = result.stdout or ""
            data = None
            json_start = stdout.find("{")
            if json_start >= 0:
                try:
                    data = json.loads(stdout[json_start:])
                except json.JSONDecodeError:
                    pass
            if data is None:
                data = {"buy_count": 0, "buy_signals": [], "total": 0,
                        "parse_error": stdout[:500]}

            formatted = _build_agent_message(data)
            formatted["stderr"] = result.stderr[-1000:] if result.stderr else ""
            formatted["ok"]     = result.returncode == 0
            self._json(200, formatted)

        except subprocess.TimeoutExpired:
            print("[API] 타임아웃: run_daily.py", flush=True)
            self._json(200, {"ok": False, "buy_count": 0, "message": "", "error": "timeout"})
        except Exception as e:
            print(f"[API] 예외: {e}", flush=True)
            self._json(500, {"ok": False, "buy_count": 0, "message": "", "error": str(e)})

    def _json(self, status: int, data: dict):
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except BrokenPipeError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("ALPHAFIN_API_PORT", 8765))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"AlphaFin API 서버 시작 (멀티스레드): http://127.0.0.1:{port}", flush=True)
    print("엔드포인트:")
    for p in list(SCRIPTS) + ["/run-agents", "/health"]:
        print(f"  GET http://127.0.0.1:{port}{p}")
    print("Ctrl+C로 종료")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")
        server.shutdown()
