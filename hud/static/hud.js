/* VIYON HUD — vanilla JS + Canvas. Reactor centerpiece, live panels, WebSocket. */
(() => {
  "use strict";

  const COLORS = {
    cyan: "#22d3ee",
    hot: "#7df9ff",
    deep: "#0a4d6b",
    amber: "#ffb020",
    dim: "#1d3c4c",
  };

  // Fallback roster (matches core.router.AGENT_REGISTRY) so the HUD renders pre-data.
  const ROSTER = [
    ["NOVA", "💻"], ["FORGE", "🏗️"], ["SHIELD", "🛡️"], ["PULSE", "🔬"],
    ["ATLAS", "🖥️"], ["ECHO", "📡"], ["NEXUS", "📊"], ["VISTA", "🎨"],
    ["TEMPO", "📋"], ["SAGE", "💬"], ["LUNA", "🌙"], ["GHOST", "👁️"],
  ];

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const MOCK = new URLSearchParams(location.search).get("mock") === "1";

  // -- shared state ----------------------------------------------------------
  const state = {
    cpu: 0, mem: 0, disk: 0, battery: null,
    netUp: 0, netDown: 0,
    weather: null, listening: false, alert: false,
    activeAgent: null, lastCommand: "",
    agents: ROSTER.map(([name, emoji]) => ({ name, emoji, status: "idle" })),
  };
  const cpuHist = new Array(60).fill(0);
  const netHist = new Array(48).fill(0);
  let lastLoggedCommand = "";

  // -- DOM -------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const els = {
    barCpu: $("bar-cpu"), barMem: $("bar-mem"), barDisk: $("bar-disk"),
    valCpu: $("val-cpu"), valMem: $("val-mem"), valDisk: $("val-disk"),
    valUp: $("val-up"), valDown: $("val-down"),
    clock: $("clock"), date: $("date"), weather: $("weather"),
    battery: $("battery"), link: $("link-state"),
    cmdlog: $("cmdlog"), cmdform: $("cmdform"), cmdinput: $("cmdinput"),
    activeEmoji: $("active-emoji"), activeName: $("active-name"), activeStatus: $("active-status"),
    grid: $("agent-grid"), coreState: $("core-state"),
    spark: $("spark-cpu"), ambL: $("ambient-left"), ambR: $("ambient-right"),
    reactor: $("reactor"), boot: $("boot"), bootText: $("boot-text"),
  };

  // -- canvas DPR helper -----------------------------------------------------
  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  // -- agent chips (built once) ---------------------------------------------
  function buildGrid() {
    els.grid.innerHTML = "";
    state.agents.forEach((a) => {
      const chip = document.createElement("div");
      chip.className = "agent-chip";
      chip.dataset.status = a.status;
      chip.dataset.name = a.name;
      chip.title = a.name;
      chip.textContent = a.emoji;
      els.grid.appendChild(chip);
    });
  }

  // -- panel updates (only on data ticks) ------------------------------------
  function setBar(barEl, valEl, pct) {
    barEl.style.width = Math.max(0, Math.min(100, pct)) + "%";
    barEl.classList.toggle("hot", pct >= 85);
    valEl.textContent = pct.toFixed(0) + "%";
  }

  function logCommand(text) {
    if (!text || text === lastLoggedCommand) return;
    lastLoggedCommand = text;
    const li = document.createElement("li");
    li.className = "fresh";
    const ts = new Date().toLocaleTimeString("en-GB");
    li.innerHTML = `<span class="ts">${ts}</span>${text}`;
    els.cmdlog.appendChild(li);
    [...els.cmdlog.querySelectorAll("li")].forEach((n) => { if (n !== li) n.classList.remove("fresh"); });
    while (els.cmdlog.children.length > 7) els.cmdlog.removeChild(els.cmdlog.firstChild);
  }

  function updatePanels() {
    setBar(els.barCpu, els.valCpu, state.cpu);
    setBar(els.barMem, els.valMem, state.mem);
    setBar(els.barDisk, els.valDisk, state.disk);
    els.valUp.textContent = state.netUp.toFixed(0);
    els.valDown.textContent = state.netDown.toFixed(0);
    els.weather.textContent = state.weather || "—";
    els.battery.textContent = state.battery == null ? "—" : state.battery + "%";

    document.body.classList.toggle("alert", !!state.alert);
    document.body.classList.toggle("listening", !!state.listening);

    const active = state.agents.find((a) => a.name === state.activeAgent);
    if (active) {
      els.activeEmoji.textContent = active.emoji;
      els.activeName.textContent = active.name;
      els.activeStatus.textContent = (active.status || "working").toUpperCase();
    } else {
      els.activeEmoji.textContent = state.listening ? "🎙️" : "◈";
      els.activeName.textContent = state.listening ? "LISTENING" : "STANDBY";
      els.activeStatus.textContent = state.listening ? "AWAITING COMMAND" : "IDLE";
    }
    els.coreState.textContent = state.alert ? "CONFIRM" : state.listening ? "LISTENING" : "CORE";

    // sync chip statuses
    state.agents.forEach((a) => {
      const chip = els.grid.querySelector(`[data-name="${a.name}"]`);
      if (chip) chip.dataset.status = a.status;
    });

    logCommand(state.lastCommand);
    drawSpark();
    drawAmbient();
  }

  // -- sparkline + ambient bars ---------------------------------------------
  function drawSpark() {
    const { ctx, w, h } = fit(els.spark);
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    cpuHist.forEach((v, i) => {
      const x = (i / (cpuHist.length - 1)) * w;
      const y = h - (v / 100) * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = COLORS.cyan;
    ctx.lineWidth = 1.5;
    ctx.shadowColor = COLORS.cyan;
    ctx.shadowBlur = 6;
    ctx.stroke();
  }

  function drawAmbient() {
    [els.ambL, els.ambR].forEach((cv, side) => {
      const { ctx, w, h } = fit(cv);
      ctx.clearRect(0, 0, w, h);
      const bars = 26;
      for (let i = 0; i < bars; i++) {
        const seed = netHist[(i + side * 7) % netHist.length] || 0;
        const cpuMix = cpuHist[(i * 2) % cpuHist.length] / 100;
        const mag = Math.min(1, seed / 400 + cpuMix * 0.6 + 0.05);
        const bh = mag * (w - 14);
        const y = (i / bars) * h + 3;
        ctx.fillStyle = `rgba(34,211,238,${0.18 + mag * 0.5})`;
        const bx = side === 0 ? 0 : w - bh;
        ctx.fillRect(bx, y, bh, (h / bars) - 4);
      }
    });
  }

  // -- the reactor -----------------------------------------------------------
  const reCtx = els.reactor.getContext("2d");
  let bootStart = null;

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  function drawReactor(now) {
    const dpr = window.devicePixelRatio || 1;
    const W = els.reactor.clientWidth, H = els.reactor.clientHeight;
    if (els.reactor.width !== Math.floor(W * dpr)) {
      els.reactor.width = Math.floor(W * dpr);
      els.reactor.height = Math.floor(H * dpr);
    }
    reCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    reCtx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const R = Math.min(W, H) / 2 * 0.92;
    const t = now / 1000;

    // boot spin-up (skipped under reduced motion)
    let boot = 1;
    if (!reduceMotion) {
      if (bootStart == null) bootStart = now;
      boot = Math.min(1, (now - bootStart) / 1200);
    }
    const eb = easeOut(boot);
    const spin = reduceMotion ? 0 : t;

    reCtx.save();
    reCtx.translate(cx, cy);
    reCtx.globalAlpha = eb;

    // concentric rings rotating at different speeds
    const rings = [
      { r: 0.92, w: 1, speed: 0.05, dash: [2, 10], color: COLORS.deep },
      { r: 0.82, w: 1.5, speed: -0.12, dash: [], color: "rgba(34,211,238,0.5)" },
      { r: 0.66, w: 1, speed: 0.22, dash: [4, 8], color: "rgba(34,211,238,0.35)" },
      { r: 0.5, w: 2, speed: -0.3, dash: [], color: "rgba(34,211,238,0.6)" },
      { r: 0.36, w: 1, speed: 0.5, dash: [1, 6], color: COLORS.cyan },
    ];
    rings.forEach((ring) => {
      reCtx.save();
      reCtx.rotate(spin * ring.speed * Math.PI);
      reCtx.beginPath();
      reCtx.arc(0, 0, R * ring.r * eb, 0, Math.PI * 2);
      reCtx.setLineDash(ring.dash);
      reCtx.lineWidth = ring.w;
      reCtx.strokeStyle = ring.color;
      reCtx.shadowColor = COLORS.cyan;
      reCtx.shadowBlur = 8;
      reCtx.stroke();
      reCtx.restore();
    });

    // tick marks on the outer ring
    reCtx.save();
    reCtx.rotate(spin * 0.08);
    for (let i = 0; i < 60; i++) {
      const a = (i / 60) * Math.PI * 2;
      const r1 = R * 0.86, r2 = R * (i % 5 === 0 ? 0.9 : 0.88);
      reCtx.beginPath();
      reCtx.moveTo(Math.cos(a) * r1, Math.sin(a) * r1);
      reCtx.lineTo(Math.cos(a) * r2, Math.sin(a) * r2);
      reCtx.strokeStyle = "rgba(34,211,238,0.3)";
      reCtx.lineWidth = 1;
      reCtx.stroke();
    }
    reCtx.restore();

    // agent nodes on the 0.82 ring
    const nodeR = R * 0.82 * eb;
    const working = [];
    state.agents.forEach((a, i) => {
      const ang = (i / state.agents.length) * Math.PI * 2 - Math.PI / 2;
      const nx = Math.cos(ang) * nodeR, ny = Math.sin(ang) * nodeR;
      const isWork = a.status === "working";
      const isDone = a.status === "done";
      if (isWork) working.push([nx, ny]);

      const pulse = isWork && !reduceMotion ? 1 + 0.35 * Math.sin(t * 6 + i) : 1;
      const dot = (isWork ? 7 : 5) * pulse;
      reCtx.beginPath();
      reCtx.arc(nx, ny, dot, 0, Math.PI * 2);
      reCtx.fillStyle = isWork ? COLORS.hot : isDone ? COLORS.cyan : COLORS.dim;
      reCtx.shadowColor = isWork ? COLORS.hot : COLORS.cyan;
      reCtx.shadowBlur = isWork ? 16 : isDone ? 8 : 0;
      reCtx.fill();

      // emoji label just outside the node
      reCtx.shadowBlur = 0;
      reCtx.font = `${Math.round(R * 0.045)}px "Share Tech Mono", monospace`;
      reCtx.textAlign = "center";
      reCtx.textBaseline = "middle";
      reCtx.globalAlpha = eb * (isWork || isDone ? 1 : 0.55);
      reCtx.fillText(a.emoji, Math.cos(ang) * (nodeR + R * 0.08), Math.sin(ang) * (nodeR + R * 0.08));
      reCtx.globalAlpha = eb;
    });

    // working → thin line to the core
    working.forEach(([nx, ny]) => {
      reCtx.beginPath();
      reCtx.moveTo(nx, ny);
      reCtx.lineTo(0, 0);
      reCtx.strokeStyle = "rgba(125,249,255,0.5)";
      reCtx.lineWidth = 1;
      reCtx.setLineDash([3, 5]);
      reCtx.lineDashOffset = reduceMotion ? 0 : -t * 20;
      reCtx.shadowColor = COLORS.hot;
      reCtx.shadowBlur = 6;
      reCtx.stroke();
      reCtx.setLineDash([]);
    });

    // breathing scale when listening
    let breathe = 1;
    if (state.listening && !reduceMotion) breathe = 1 + 0.08 * Math.sin(t * 2.2);

    // core triangle (VIYON CORE)
    const alert = state.alert;
    const coreColor = alert ? COLORS.amber : COLORS.hot;
    if (alert && !reduceMotion) {
      // amber flash ring
      reCtx.beginPath();
      reCtx.arc(0, 0, R * 0.3 * eb, 0, Math.PI * 2);
      reCtx.strokeStyle = `rgba(255,176,32,${0.4 + 0.4 * Math.abs(Math.sin(t * 5))})`;
      reCtx.lineWidth = 3;
      reCtx.shadowColor = COLORS.amber;
      reCtx.shadowBlur = 20;
      reCtx.stroke();
    }
    reCtx.save();
    reCtx.rotate(reduceMotion ? 0 : t * 0.4);
    reCtx.scale(breathe * eb, breathe * eb);
    const cr = R * 0.13;
    reCtx.beginPath();
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2 - Math.PI / 2;
      const x = Math.cos(a) * cr, y = Math.sin(a) * cr;
      i ? reCtx.lineTo(x, y) : reCtx.moveTo(x, y);
    }
    reCtx.closePath();
    const grad = reCtx.createLinearGradient(-cr, -cr, cr, cr);
    grad.addColorStop(0, coreColor);
    grad.addColorStop(1, alert ? "#7a4a00" : COLORS.deep);
    reCtx.fillStyle = grad;
    reCtx.shadowColor = coreColor;
    reCtx.shadowBlur = 30 + (state.listening ? 18 * Math.abs(Math.sin(t * 2.2)) : 0);
    reCtx.fill();
    reCtx.lineWidth = 1.5;
    reCtx.strokeStyle = coreColor;
    reCtx.stroke();
    reCtx.restore();

    reCtx.restore();
  }

  function loop(now) {
    drawReactor(now);
    requestAnimationFrame(loop);
  }

  // -- boot sequence ---------------------------------------------------------
  function runBoot() {
    if (reduceMotion) {
      document.body.classList.remove("booting");
      document.body.classList.add("boot-done");
      return;
    }
    const msg = "VIYON ONLINE";
    let i = 0;
    const typer = setInterval(() => {
      els.bootText.textContent = msg.slice(0, ++i);
      if (i >= msg.length) clearInterval(typer);
    }, 55);
    setTimeout(() => document.body.classList.remove("booting"), 1200);
    setTimeout(() => document.body.classList.add("boot-done"), 1900);
  }

  // -- ingest a telemetry payload -------------------------------------------
  function ingest(p) {
    if (!p) return;
    if (typeof p.cpu === "number") state.cpu = p.cpu;
    if (typeof p.mem === "number") state.mem = p.mem;
    if (typeof p.disk === "number") state.disk = p.disk;
    if ("battery" in p) state.battery = p.battery;
    if ("net_up" in p) state.netUp = p.net_up;
    if ("net_down" in p) state.netDown = p.net_down;
    if ("weather" in p) state.weather = p.weather;
    if ("listening" in p) state.listening = p.listening;
    if ("alert" in p) state.alert = p.alert;
    if ("active_agent" in p) state.activeAgent = p.active_agent;
    if ("last_command" in p) state.lastCommand = p.last_command || "";
    if (Array.isArray(p.agents) && p.agents.length) state.agents = p.agents;
    if (p.time) els.clock.textContent = p.time;
    if (p.date) els.date.textContent = p.date;

    cpuHist.push(state.cpu); cpuHist.shift();
    netHist.push((state.netUp || 0) + (state.netDown || 0)); netHist.shift();
    updatePanels();
  }

  // -- data sources ----------------------------------------------------------
  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let ws;
    const open = () => {
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => { els.link.textContent = "ONLINE"; els.link.style.color = COLORS.cyan; };
      ws.onmessage = (ev) => { try { ingest(JSON.parse(ev.data)); } catch (_) {} };
      ws.onclose = () => {
        els.link.textContent = "RECONNECT…"; els.link.style.color = COLORS.amber;
        setTimeout(open, 1500);
      };
      ws.onerror = () => ws.close();
    };
    open();

    els.cmdform.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = els.cmdinput.value.trim();
      if (!text) return;
      els.cmdinput.value = "";
      ingest({ last_command: text });
      try {
        await fetch("/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      } catch (_) {}
    });
  }

  function runMock() {
    els.link.textContent = "MOCK"; els.link.style.color = COLORS.amber;
    const samples = [
      "viyon, what's my cpu doing", "research vector databases and build a demo",
      "open safari", "scaffold a fastapi project", "scan my project for vulns",
    ];
    let phase = 0;
    setInterval(() => {
      phase++;
      const agents = ROSTER.map(([name, emoji]) => ({ name, emoji, status: "idle" }));
      const w = phase % 6 === 0 ? Math.floor(Math.random() * 12) : -1;
      let active = null;
      if (w >= 0) { agents[w].status = "working"; active = agents[w].name; }
      const now = new Date();
      ingest({
        cpu: 18 + Math.random() * 40 + (w >= 0 ? 25 : 0),
        mem: 52 + Math.random() * 12,
        disk: 61,
        battery: 88,
        net_up: Math.random() * 120,
        net_down: Math.random() * 600,
        time: now.toLocaleTimeString("en-GB"),
        date: now.toDateString().toUpperCase(),
        weather: "CLEAR +18°C",
        agents,
        active_agent: active,
        last_command: phase % 12 === 0 ? samples[(phase / 12) % samples.length | 0] : state.lastCommand,
        listening: phase % 18 < 3,
        alert: phase % 30 < 2,
      });
    }, 500);

    els.cmdform.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = els.cmdinput.value.trim();
      if (!text) return;
      els.cmdinput.value = "";
      ingest({ last_command: text });
    });
  }

  // -- init ------------------------------------------------------------------
  buildGrid();
  updatePanels();
  runBoot();
  requestAnimationFrame(loop);
  window.addEventListener("resize", () => { drawSpark(); drawAmbient(); });
  if (MOCK) runMock(); else connectWS();
})();
