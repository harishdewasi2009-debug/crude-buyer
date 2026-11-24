"""Agent 1 · Research Director: plans tasks, runs the other agents, controls start/pause/resume/stop."""
import json
import threading
import time
from datetime import datetime, timezone, timedelta

from .agents import (Control, Ctx, Stopped, RefineryProcurementAgent, PrivateBuyerDiscoveryAgent,
                     CompanyVerificationAgent, ContactFinderAgent, DueDiligenceAgent, QualityControlAgent,
                     DataManagerAgent)
from .fetcher import Fetcher
from .util import now_iso


class Director:
    name = "Agent 1 · Research Director"

    def __init__(self, db, llm):
        self.db, self.llm = db, llm
        self.ctl = None
        self.run_id = None
        self.thread = None
        self._lock = threading.Lock()
        self._sched_stop = threading.Event()
        # a run interrupted by a restart can never finish: mark it honestly
        self.db.ex("UPDATE runs SET status='interrupted', finished=? WHERE status IN ('running','paused')", (now_iso(),))
        self.db.ex("UPDATE tasks SET status='interrupted' WHERE status='running'")

    # ------------------------------------------------------------ control
    def active(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self, cfg=None):
        with self._lock:
            if self.active():
                raise RuntimeError("a research run is already active")
            cfg = cfg or {}
            self.ctl = Control()
            self.run_id = self.db.ex("INSERT INTO runs(status,config,started) VALUES('running',?,?)",
                                     (json.dumps(cfg), now_iso()))
            self.thread = threading.Thread(target=self._run, args=(self.run_id, cfg, self.ctl), daemon=True)
            self.thread.start()
            return self.run_id

    def pause(self):
        if self.active():
            self.ctl.pause()
            self.db.ex("UPDATE runs SET status='paused' WHERE id=?", (self.run_id,))
            self.db.log(self.run_id, "info", self.name, "paused")

    def resume(self):
        if self.active():
            self.ctl.resume()
            self.db.ex("UPDATE runs SET status='running' WHERE id=?", (self.run_id,))
            self.db.log(self.run_id, "info", self.name, "resumed")

    def stop(self):
        if self.active():
            self.ctl.stop()
            self.db.log(self.run_id, "info", self.name, "stop requested")

    def join(self, timeout=60):
        if self.thread:
            self.thread.join(timeout)

    # ------------------------------------------------------------ run
    def _task(self, ctx, agent, ttype, target, fn):
        tid = self.db.ex("INSERT INTO tasks(run_id,agent,type,target,status,started) VALUES(?,?,?,?, 'running',?)",
                         (ctx.run_id, agent.name, ttype, str(target)[:200], now_iso()))
        try:
            res = fn() or {}
            if res.get("skipped"):
                status = "skipped"
            elif res.get("findings", 1) == 0:
                status = "no_findings"
            else:
                status = "completed"
            self.db.ex("UPDATE tasks SET status=?, finished=?, result=? WHERE id=?",
                       (status, now_iso(), json.dumps(res)[:1000], tid))
            return True
        except Stopped:
            self.db.ex("UPDATE tasks SET status='stopped', finished=? WHERE id=?", (now_iso(), tid))
            raise
        except Exception as e:
            self.db.ex("UPDATE tasks SET status='failed', finished=?, error=? WHERE id=?",
                       (now_iso(), f"{type(e).__name__}: {str(e)[:300]}", tid))
            self.db.log(ctx.run_id, "error", agent.name, f"{ttype} '{target}' failed: {type(e).__name__}: {str(e)[:200]}")
            return False

    def _run(self, run_id, cfg, ctl):
        db = self.db
        ctx = Ctx(db, self.llm, Fetcher(db, ctl, run_id), ctl, run_id)
        a2, a3, a4 = RefineryProcurementAgent(ctx), PrivateBuyerDiscoveryAgent(ctx), CompanyVerificationAgent(ctx)
        a5, a6, a8, a7 = ContactFinderAgent(ctx), DueDiligenceAgent(ctx), QualityControlAgent(ctx), DataManagerAgent(ctx)
        failures = 0
        final = "completed"
        try:
            db.log(run_id, "info", self.name, f"run started; AI providers available: {self.llm.available() or 'none (rule-based mode)'}")
            targets = []
            if cfg.get("companies"):
                for c in cfg["companies"]:
                    cid, _ = db.upsert_company(c["name"], website=c.get("website") or None, country=c.get("country") or None,
                                               category="Manually added")
                    targets.append(cid)
            else:
                ids = cfg.get("sources")
                db.ensure_source_companies(ids)
                q = "SELECT * FROM sources WHERE enabled=1" if not ids else \
                    f"SELECT * FROM sources WHERE id IN ({','.join('?' * len(ids))})"
                for s in db.q(q, tuple(ids or ())):
                    if not self._task(ctx, a2, "refinery_crawl", s["name"], lambda s=s: a2.run(s)):
                        failures += 1
                if cfg.get("discovery"):
                    for qy in db.get_settings()["search_queries"]:
                        if not self._task(ctx, a3, "discover_buyers", qy, lambda qy=qy: a3.run_query(qy)):
                            failures += 1
                targets = sorted(ctx.touched)
                if not ids:   # full run: also pick up companies never researched; a single-source run stays scoped
                    stale = db.q("SELECT id FROM companies WHERE classification='UNCLASSIFIED' OR last_research_date IS NULL")
                    targets = list(dict.fromkeys(targets + [r["id"] for r in stale]))
            for cid in targets:
                ctl.checkpoint()
                nm = db.company(cid)["name"]
                for agent, stage, fn in ((a4, "verify_company", a4.run), (a5, "find_contacts", a5.run),
                                         (a6, "due_diligence", a6.run), (a8, "quality_control", a8.run)):
                    if not self._task(ctx, agent, stage, nm, lambda fn=fn, cid=cid: fn(cid)):
                        failures += 1
            self._task(ctx, a7, "dedupe_and_store", "database", lambda: {"merged": a7.dedupe(), "findings": 1})
            final = "completed_with_errors" if failures else "completed"
        except Stopped:
            final = "stopped"
        except Exception as e:
            final = "failed"
            db.log(run_id, "error", self.name, f"run failed: {type(e).__name__}: {str(e)[:200]}")
        db.ex("UPDATE runs SET status=?, finished=?, summary=? WHERE id=?",
              (final, now_iso(), json.dumps({"failures": failures}), run_id))
        db.log(run_id, "info", self.name, f"run {final}")

    # ------------------------------------------------------------ scheduler (periodic rechecks)
    def start_scheduler(self):
        threading.Thread(target=self._sched_loop, daemon=True).start()

    def _sched_loop(self):
        while not self._sched_stop.wait(60):
            try:
                hrs = float(self.db.get_settings().get("recheck_hours") or 0)
                if hrs <= 0 or self.active():
                    continue
                last = self.db.one("SELECT started FROM runs ORDER BY id DESC LIMIT 1")
                due = True
                if last:
                    t = datetime.fromisoformat(last["started"])
                    due = datetime.now(timezone.utc) - t >= timedelta(hours=hrs)
                if due:
                    self.start({"scheduled": True, "discovery": True})
            except Exception:
                pass

    def stop_scheduler(self):
        self._sched_stop.set()
