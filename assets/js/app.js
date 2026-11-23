/* Project page interactions. All data comes from assets/data/data.js (window.NBM), built from the paper's recorded outputs. */
(() => {
  "use strict";
  const D = window.NBM;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  /* ---------------------------------------------------------------- theme */
  $("#themeBtn").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("nbm-theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
  });

  /* run `start` while `el` is on screen, `stop` when it leaves */
  function whenVisible(el, start, stop) {
    if (!("IntersectionObserver" in window)) { start(); return; }
    new IntersectionObserver((es) => es.forEach((e) => (e.isIntersecting ? start() : stop())), { threshold: 0.25 }).observe(el);
  }
  const PLAY = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>';
  const PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
  const PREV = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M15 5l-7 7 7 7"/></svg>';
  const NEXT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M9 5l7 7-7 7"/></svg>';

  /* ---------------------------------------------------------------- chess board (SVG, 45 units per square = piece size) */
  const FILES = "abcdefgh";
  const PIECES = {};
  for (const [k, v] of Object.entries(D.pieces)) PIECES[k] = v.replace(/\sid="[^"]*"/g, "");
  const sqXY = (s) => [FILES.indexOf(s[0]) * 45, (8 - Number(s[1])) * 45];
  function parseFen(fen) {
    const out = {};
    fen.split(" ")[0].split("/").forEach((row, r) => {
      let f = 0;
      for (const ch of row) {
        if (/\d/.test(ch)) f += Number(ch);
        else { out[FILES[f] + (8 - r)] = ch; f += 1; }
      }
    });
    return out;
  }
  function makeBoard(svg) {
    svg.innerHTML = "";
    const g = (cls) => { const e = document.createElementNS(NS, "g"); e.setAttribute("class", cls); svg.appendChild(e); return e; };
    const sqs = g("squares"), heat = g("heat"), marks = g("marks"), pieces = g("pieces"), top = g("top");
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
      const rc = document.createElementNS(NS, "rect");
      rc.setAttribute("x", f * 45); rc.setAttribute("y", r * 45); rc.setAttribute("width", 45); rc.setAttribute("height", 45);
      rc.dataset.light = (r + f) % 2 === 0 ? "1" : "0";
      sqs.appendChild(rc);
    }
    const B = { svg, sqs, heat, marks, pieces, top, grey: false };
    B.paintSquares = (grey) => {
      B.grey = grey;
      $$("rect", sqs).forEach((rc) => rc.setAttribute("fill", rc.dataset.light === "1" ? (grey ? "#ececec" : "#f0d9b5") : (grey ? "#cfcfcf" : "#b58863")));
    };
    B.paintSquares(false);
    B.setFen = (fen) => {
      pieces.innerHTML = "";
      for (const [s, p] of Object.entries(parseFen(fen))) {
        const e = document.createElementNS(NS, "g");
        e.setAttribute("class", "piece"); e.dataset.sq = s;
        const [x, y] = sqXY(s);
        e.style.transform = `translate(${x}px, ${y}px)`;
        e.innerHTML = PIECES[p];
        pieces.appendChild(e);
      }
    };
    B.mark = (s, color, opacity = 0.85, fade = true) => {
      const [x, y] = sqXY(s);
      const rc = document.createElementNS(NS, "rect");
      rc.setAttribute("x", x); rc.setAttribute("y", y); rc.setAttribute("width", 45); rc.setAttribute("height", 45);
      rc.setAttribute("fill", color); rc.setAttribute("class", "mk");
      rc.style.opacity = fade ? 0 : opacity;
      marks.appendChild(rc);
      if (fade) requestAnimationFrame(() => requestAnimationFrame(() => { rc.style.opacity = opacity; }));
      return rc;
    };
    B.outline = (s, color, dash = "", w = 3) => {
      const [x, y] = sqXY(s);
      const rc = document.createElementNS(NS, "rect");
      rc.setAttribute("x", x + w / 2); rc.setAttribute("y", y + w / 2); rc.setAttribute("width", 45 - w); rc.setAttribute("height", 45 - w);
      rc.setAttribute("fill", "none"); rc.setAttribute("stroke", color); rc.setAttribute("stroke-width", w);
      if (dash) rc.setAttribute("stroke-dasharray", dash);
      top.appendChild(rc);
    };
    B.frame = (color) => {
      top.innerHTML = "";
      if (!color) return;
      const rc = document.createElementNS(NS, "rect");
      rc.setAttribute("x", 2.5); rc.setAttribute("y", 2.5); rc.setAttribute("width", 355); rc.setAttribute("height", 355);
      rc.setAttribute("fill", "none"); rc.setAttribute("stroke", color); rc.setAttribute("stroke-width", 5); rc.setAttribute("rx", 4);
      top.appendChild(rc);
    };
    B.clearMarks = () => { marks.innerHTML = ""; heat.innerHTML = ""; };
    B.slide = (from, to) => {
      const p = $(`.piece[data-sq="${from}"]`, pieces);
      const cap = $(`.piece[data-sq="${to}"]`, pieces);
      if (cap) cap.style.opacity = 0.35;
      if (!p) return;
      const [x, y] = sqXY(to);
      pieces.appendChild(p); // on top
      p.style.transform = `translate(${x}px, ${y}px)`;
    };
    return B;
  }
  const BLUE = "#2f6fd6", RED = "#e0452f";

  /* ---------------------------------------------------------------- teaser: the model's moves on Fig. 1 / Fig. 3 puzzles */
  const NOTES = {
    rook: ["", "", "rook sacrifice", "", "", "", "checkmate"],
    queen: ["queen sacrifice: ignores the free rook on e3", "", "the queen again", "", "checkmate"],
  };
  const TB = makeBoard($("#teaserBoard"));
  const T = { line: "rook", idx: 0, timer: null, playing: !REDUCED, visible: false, steps: [] };
  function teaserSteps(key) {
    const L = D.teaser[key], st = [];
    const status = (html) => { $("#teaserStatus").innerHTML = html; };
    const list = (k, now) => $$("#teaserMoves span").forEach((s, i) => {
      s.classList.toggle("done", i < k || (i === k && now === "done"));
      s.classList.toggle("now", i === k && now === "now");
    });
    let call = 0;
    L.plies.forEach((p, i) => {
      const solver = p.side === "solver";
      if (solver) call += 1;
      const c = call, note = NOTES[key][i];
      st.push({ dur: solver ? 650 : 350, fn: () => { TB.frame(null); TB.clearMarks(); TB.setFen(p.fen); list(i, "now");
        status(solver ? `<span class="badge model">model call ${c}</span><span>sees the board&hellip;</span>` : `<span class="badge">opponent</span><span>replies</span>`); } });
      if (solver) {
        st.push({ dur: 1500, fn: () => { TB.mark(p.from, BLUE); TB.mark(p.to, RED);
          status(`<span class="badge model">model call ${c}</span><span>draws <b>${p.san}</b>${note && note !== "checkmate" ? ` &mdash; ${note}` : ""}</span>`); } });
      } else {
        st.push({ dur: 600, fn: () => { TB.mark(p.from, "#7a7a7a", 0.45); TB.mark(p.to, "#7a7a7a", 0.45);
          status(`<span class="badge">opponent</span><span>replies <b>&hellip;${p.san}</b></span>`); } });
      }
      st.push({ dur: 600, fn: () => { TB.slide(p.from, p.to); list(i, "done"); } });
    });
    st.push({ dur: 3400, fn: () => { TB.clearMarks(); TB.setFen(L.final); TB.frame(css("--good") || "#0f9d6b");
      status(`<span class="badge mate">checkmate</span><span>every move drawn by the model, one call per move</span>`); } });
    return st;
  }
  function teaserLoad(key) {
    clearTimeout(T.timer);
    T.line = key; T.idx = 0; T.steps = teaserSteps(key);
    const L = D.teaser[key];
    $("#teaserMoves").innerHTML = L.plies.map((p) => `<span class="${p.side === "solver" ? "solver" : "opp"}">${p.san}</span>`).join("");
    $("#teaserNote").textContent = `held-out Lichess puzzle, rated ${L.rating}`;
    $$("#idea .tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.line === key));
    T.steps[0].fn(); T.idx = 1;
    if (!T.playing) { for (; T.idx < 2; T.idx++) T.steps[T.idx].fn(); }
    teaserTick();
  }
  function teaserTick() {
    clearTimeout(T.timer);
    if (!T.playing || !T.visible) return;
    const s = T.steps[T.idx - 1];
    T.timer = setTimeout(() => {
      if (T.idx >= T.steps.length) { T.idx = 0; }
      T.steps[T.idx].fn(); T.idx += 1; teaserTick();
    }, s ? s.dur : 500);
  }
  function teaserPlayBtn() { $("#teaserPlay").innerHTML = T.playing ? PAUSE : PLAY; $("#teaserPlay").setAttribute("aria-label", T.playing ? "Pause" : "Play"); }
  $("#teaserPlay").addEventListener("click", () => {
    T.playing = !T.playing; teaserPlayBtn();
    if (T.playing) teaserTick(); else clearTimeout(T.timer);
  });
  $("#teaserRestart").addEventListener("click", () => teaserLoad(T.line));
  $$("#idea .tabs button").forEach((b) => b.addEventListener("click", () => teaserLoad(b.dataset.line)));
  teaserPlayBtn();
  teaserLoad("rook");
  whenVisible($("#teaserBoard"), () => { T.visible = true; teaserTick(); }, () => { T.visible = false; clearTimeout(T.timer); });

  /* ---------------------------------------------------------------- generic image stepper */
  function stepper(el, frames, { interval = 1100, hold = 2200, alt = "" } = {}) {
    el.innerHTML = `<div class="frame"><img alt="${alt}"><span class="tag" hidden></span></div>
      <div class="lab"><span class="txt"></span><span style="display:flex;gap:6px;flex:none">
      <button class="iconbtn prev" aria-label="Previous">${PREV}</button><button class="iconbtn play" aria-label="Pause">${PAUSE}</button>
      <button class="iconbtn next" aria-label="Next">${NEXT}</button></span></div><div class="dots"></div>`;
    const img = $("img", el), tag = $(".tag", el), txt = $(".txt", el), dots = $(".dots", el);
    dots.innerHTML = frames.map(() => "<i></i>").join("");
    frames.forEach((f) => { const im = new Image(); im.src = f.src; });
    const S = { i: 0, playing: !REDUCED, visible: false, timer: null };
    const show = (i) => {
      S.i = (i + frames.length) % frames.length;
      const f = frames[S.i];
      img.src = f.src; txt.innerHTML = f.label || "";
      tag.hidden = !f.tag; tag.textContent = f.tag || ""; tag.className = "tag " + (f.tagClass || "");
      $$("i", dots).forEach((d, k) => d.classList.toggle("on", k === S.i));
    };
    const tick = () => {
      clearTimeout(S.timer);
      if (!S.playing || !S.visible) return;
      const last = S.i === frames.length - 1;
      S.timer = setTimeout(() => { show(S.i + 1); tick(); }, last ? hold : (frames[S.i].dur || interval));
    };
    const btn = $(".play", el);
    const sync = () => { btn.innerHTML = S.playing ? PAUSE : PLAY; btn.setAttribute("aria-label", S.playing ? "Pause" : "Play"); };
    btn.addEventListener("click", () => { S.playing = !S.playing; sync(); tick(); });
    $(".prev", el).addEventListener("click", () => { S.playing = false; sync(); clearTimeout(S.timer); show(S.i - 1); });
    $(".next", el).addEventListener("click", () => { S.playing = false; sync(); clearTimeout(S.timer); show(S.i + 1); });
    $$("i", dots).forEach((d, k) => d.addEventListener("click", () => { S.playing = false; sync(); clearTimeout(S.timer); show(k); }));
    sync(); show(0);
    whenVisible(el, () => { S.visible = true; tick(); }, () => { S.visible = false; clearTimeout(S.timer); });
    return { show, el, frames };
  }

  /* ---------------------------------------------------------------- method panels */
  stepper($("#editStepper"), [
    { src: D.edit.cond, tag: "input x", label: "the board the model sees", dur: 1300 },
    { src: D.edit.output, tag: "model output", tagClass: "decide", label: "unaltered 20-step sample: the move drawn on the image", dur: 2000 },
    { src: D.edit.target, tag: "training target M(x, a)", tagClass: "imagine", label: "what training targets look like: the same edit", dur: 1600 },
  ], { hold: 1600, alt: "Action as an image edit" });
  stepper($("#loopStepper"), D.loop.frames.map((f) => ({
    src: f.img, tag: f.kind === "in" ? `call ${f.call}: input` : `call ${f.call}: output`, tagClass: f.kind === "in" ? "" : "decide",
    label: f.kind === "in" ? `call <b>${f.call}</b> of ${D.loop.calls}: the current state` : `the model marks the next cell (red disc); the move is applied`,
    dur: f.kind === "in" ? 700 : 900,
  })), { hold: 2600, alt: "Closed-loop input and output frames" });
  $("#loopStepper").insertAdjacentHTML("beforeend", `<p class="tiny" style="margin:8px 0 0">Held-out test map; the finished route costs ${D.loop.cost}, the optimum.</p>`);
  stepper($("#videoStepper"), D.video_clip.frames.map((src, i) => {
    const lab = D.video_clip.labels[i];
    const isD = lab.startsWith("Decide"), isI = lab.startsWith("Imagine");
    return { src, tag: isD ? "Decide" : isI ? "Imagine" : "input", tagClass: isD ? "decide" : isI ? "imagine" : "",
      label: isD ? lab.replace("Decide · ", "") : isI ? lab.replace("Imagine · ", "") : "the board the clip starts from", dur: isD ? 1400 : 900 };
  }), { hold: 2600, alt: "Frames generated by the video model" });
  $("#videoStepper").insertAdjacentHTML("beforeend", `<p class="tiny" style="margin:8px 0 0">One generated clip, unaltered; every drawn move and imagined board is correct.</p>`);

  /* ---------------------------------------------------------------- terrain maps */
  const TC = D.terrain.codes;
  const COST = { [TC.floor]: 1, [TC.mud]: 2, [TC.sand]: 4 };
  const TILE = { [TC.floor]: "#f3efe6", [TC.wall]: "#353c4a", [TC.mud]: "#7d5532", [TC.sand]: "#e6c675" };
  function terrainDefs(id) {
    return `<defs>
      <pattern id="${id}mud" width="16" height="10" patternUnits="userSpaceOnUse"><rect width="16" height="10" fill="${TILE[TC.mud]}"/>
        <path d="M0 6 Q4 2 8 6 T16 6" fill="none" stroke="#a87a4f" stroke-width="1.6"/></pattern>
      <pattern id="${id}sand" width="10" height="10" patternUnits="userSpaceOnUse"><rect width="10" height="10" fill="${TILE[TC.sand]}"/>
        <circle cx="3" cy="3" r="1.1" fill="#b48a3a"/><circle cx="8" cy="7.5" r="1" fill="#c49a48"/></pattern>
      <pattern id="${id}wall" width="20" height="10" patternUnits="userSpaceOnUse"><rect width="20" height="10" fill="${TILE[TC.wall]}"/>
        <path d="M0 0.5H20M0 5.5H20M5 0.5V5.5M15 5.5V10.5" stroke="#4a5263" stroke-width="1"/></pattern>
    </defs>`;
  }
  function legendHTML() {
    const t = (fill, extra = "") => `<svg class="tile" viewBox="0 0 18 18" style="border:0">${terrainDefs("lg")}<rect width="18" height="18" fill="${fill}" ${extra}/></svg>`;
    return `<b style="color:var(--ink)">Hidden costs:</b>
      <span>${t(TILE[TC.floor])}floor 1</span><span>${t("url(#lgmud)")}mud 2</span><span>${t("url(#lgsand)")}sand 4</span><span>${t("url(#lgwall)")}wall</span>
      <span><svg class="tile" viewBox="0 0 18 18" style="border:0"><rect x="1.5" y="1.5" width="15" height="15" rx="3" fill="rgba(224,69,47,.22)" stroke="#e0452f" stroke-width="2.4"/></svg>the model's next cell</span>
      <span><svg class="tile" viewBox="0 0 18 18" style="border:0"><path d="M1 9H17" stroke="#666" stroke-width="2.4" stroke-dasharray="4 3"/></svg>alternative route</span>`;
  }
  $("#terrainLegend").innerHTML = legendHTML();
  function mapCard(m, k) {
    const C = 40, n = m.n, W = n * C, id = `m${k}`;
    const ctr = ([r, c]) => [c * C + C / 2, r * C + C / 2];
    let tiles = "";
    m.grid.forEach((row, r) => row.forEach((t, c) => {
      const fill = t === TC.mud ? `url(#${id}mud)` : t === TC.sand ? `url(#${id}sand)` : t === TC.wall ? `url(#${id}wall)` : TILE[TC.floor];
      tiles += `<rect x="${c * C}" y="${r * C}" width="${C}" height="${C}" fill="${fill}" stroke="rgba(0,0,0,.08)" stroke-width="1"/>`;
    }));
    const pts = (P) => P.map((p) => ctr(p).join(",")).join(" ");
    const [gx, gy] = ctr(m.goal);
    const flag = `<g transform="translate(${gx - 8},${gy - 13})"><line x1="0" y1="0" x2="0" y2="26" stroke="#1f1f1f" stroke-width="2.4"/>
      <rect x="0" y="0" width="17" height="11" fill="#fff" stroke="#1f1f1f" stroke-width="1"/>
      <rect x="0" y="0" width="4.25" height="5.5"/><rect x="8.5" y="0" width="4.25" height="5.5"/><rect x="4.25" y="5.5" width="4.25" height="5.5"/><rect x="12.75" y="5.5" width="4.25" height="5.5"/></g>`;
    const card = document.createElement("div");
    card.className = "card mapcard";
    const moves = m.path.length - 1, altMoves = m.alt.length - 1;
    const altName = m.alt_kind === "fewest" ? "fewest-steps route" : "route avoiding mud and sand";
    card.innerHTML = `<svg viewBox="-2 -2 ${W + 4} ${W + 4}" role="img" aria-label="Terrain map ${m.id}">${terrainDefs(id)}
      <rect x="-2" y="-2" width="${W + 4}" height="${W + 4}" rx="6" fill="#9a958b"/>${tiles}
      <polyline class="alt" points="${pts(m.alt)}" fill="none" stroke="#3d3d3d" stroke-width="4" stroke-dasharray="9 7" stroke-linecap="round" stroke-linejoin="round" opacity="0" style="transition:opacity .5s"/>
      <polyline class="halo" fill="none" stroke="#fff" stroke-width="13" stroke-linecap="round" stroke-linejoin="round" opacity=".85"/>
      <polyline class="trail" fill="none" stroke="#0f9d6b" stroke-width="7.5" stroke-linecap="round" stroke-linejoin="round"/>
      <rect class="next" width="${C - 4}" height="${C - 4}" rx="6" fill="rgba(224,69,47,.25)" stroke="#e0452f" stroke-width="4" opacity="0"/>
      ${flag}
      <circle class="agent" r="${C * 0.3}" fill="#2563eb" stroke="#fff" stroke-width="3.5" style="transition:cx .28s,cy .28s"/>
      <rect class="done" x="1" y="1" width="${W - 2}" height="${W - 2}" rx="4" fill="none" stroke="#0f9d6b" stroke-width="5" opacity="0"/>
    </svg>
    <div class="cap"><b class="run">cost 0</b><br><span class="muted"></span></div>`;
    const S = $("svg", card), trail = $(".trail", S), halo = $(".halo", S), next = $(".next", S), agent = $(".agent", S);
    const alt = $(".alt", S), done = $(".done", S), run = $(".run", card), sub = $(".muted", card);
    const st = { k: 0, phase: 0, timer: null, visible: false };
    const draw = () => {
      const P = m.path.slice(0, st.k + 1);
      const p = pts(P);
      trail.setAttribute("points", p); halo.setAttribute("points", p);
      const [ax, ay] = ctr(m.path[st.k]);
      agent.setAttribute("cx", ax); agent.setAttribute("cy", ay);
      let cost = 0;
      for (let i = 1; i <= st.k; i++) cost += COST[m.grid[m.path[i][0]][m.path[i][1]]];
      const fin = st.k === moves;
      run.textContent = fin ? `cost ${m.cost} in ${moves} moves ✓` : `move ${st.k} · cost so far ${cost}`;
      run.style.color = fin ? "var(--good)" : "";
      sub.textContent = fin ? `${altName} (dashed): cost ${m.alt_cost} in ${altMoves} moves` : "one model call per move";
      alt.style.opacity = fin ? 0.9 : 0; done.style.opacity = fin ? 1 : 0;
    };
    const step = () => {
      clearTimeout(st.timer);
      if (!st.visible) return;
      if (st.k === moves) {
        next.setAttribute("opacity", 0);
        st.timer = setTimeout(() => { st.k = 0; draw(); step(); }, 3800);
        return;
      }
      const [nx, ny] = ctr(m.path[st.k + 1]);
      next.setAttribute("x", nx - C / 2 + 2); next.setAttribute("y", ny - C / 2 + 2); next.setAttribute("opacity", 1);
      st.timer = setTimeout(() => { next.setAttribute("opacity", 0); st.k += 1; draw(); st.timer = setTimeout(step, 180); }, 480);
    };
    if (REDUCED) st.k = moves;
    draw();
    whenVisible(card, () => { if (REDUCED) return; st.visible = true; step(); }, () => { st.visible = false; clearTimeout(st.timer); });
    return card;
  }
  D.terrain.maps.forEach((m, k) => $("#maps").appendChild(mapCard(m, k)));

  /* ---------------------------------------------------------------- denoising explorer */
  const DN = D.denoise;
  const CAND_COL = ["#8c6bb1", "#3f8fd2", "#0f9d6b", "#d98b2b"];
  const ansIdx = DN.candidates.indexOf(DN.answer);
  $("#dnCond").src = DN.cond;
  const setOne = () => {
    const enh = $("#dnEnh").checked;
    $("#dnOne").src = enh ? DN.one_step_enh : DN.one_step;
    $("#dnOneCap").innerHTML = enh ? `one-step output, contrast &times;${DN.one_step_gain}` : "one-step output, unaltered";
  };
  $("#dnEnh").addEventListener("change", setOne); setOne();
  const counts = {};
  DN.samples8.forEach((u) => { counts[u] = (counts[u] || 0) + 1; });
  const nm = (u) => DN.names[DN.candidates.indexOf(u)] || u;
  $("#dnText").innerHTML = `The answer is <b>${DN.answer_san}</b>. Eight full samples of this board split evenly: ${Object.entries(counts).map(([u, c]) => `${nm(u)} &times;${c}`).join(", ")}. The strongest marks of the one-step output read <b class="ok">${DN.answer_san}</b>.`;
  const drawsEl = $("#draws");
  drawsEl.innerHTML = DN.draws.map((d, i) => `<div class="draw">
      <div class="sprite" style="background-image:url(${d.sprite})" role="img" aria-label="Clean-image estimate, noise seed ${d.k}"></div>
      <div class="who"><span>noise seed ${d.k}</span></div>
      <div class="bars">${DN.names.map((n, j) => `<div class="bar"><span style="color:${CAND_COL[j]}">${n}</span><div class="tr"><div class="fl" style="background:${CAND_COL[j]}"></div></div></div>`).join("")}</div>
      <div class="final"></div></div>`).join("");
  const setStep = (s) => {
    $("#dnStep").value = s;
    $("#dnOut").textContent = `step ${s + 1} / 20 · σ ${DN.sigmas[s].toFixed(2)}`;
    $$(".draw", drawsEl).forEach((el, i) => {
      const d = DN.draws[i];
      $(".sprite", el).style.backgroundPosition = `${(s / 19) * 100}% 0`;
      $$(".fl", el).forEach((f, j) => { f.style.width = `${Math.min(1, d.strength[s][j]) * 100}%`; });
      const fin = $(".final", el);
      if (s === 19) {
        const ok = d.final === DN.answer;
        fin.innerHTML = `commits to ${nm(d.final)} <span class="${ok ? "ok" : "no"}">${ok ? "✓" : "✗"}</span>`;
      } else if (s === 0) {
        const j = d.strength[0].indexOf(Math.max(...d.strength[0]));
        fin.innerHTML = `<span class="muted">first estimate: several faint marks</span>`;
        void j;
      } else fin.innerHTML = "";
    });
  };
  DN.draws.forEach((d) => { const im = new Image(); im.src = d.sprite; });
  setStep(0);
  let dnTimer = null;
  const dnStop = () => { clearInterval(dnTimer); dnTimer = null; $("#dnPlay").innerHTML = PLAY; };
  $("#dnStep").addEventListener("input", (e) => { dnStop(); setStep(Number(e.target.value)); });
  $("#dnPlay").addEventListener("click", () => {
    if (dnTimer) { dnStop(); return; }
    let s = Number($("#dnStep").value);
    if (s >= 19) s = -1;
    $("#dnPlay").innerHTML = PAUSE;
    dnTimer = setInterval(() => { if (s >= 19) { dnStop(); return; } s += 1; setStep(s); if (s >= 19) dnStop(); }, 380);
  });
  let dnAuto = false;
  whenVisible($("#draws"), () => { if (!dnAuto && !REDUCED) { dnAuto = true; setTimeout(() => $("#dnPlay").click(), 900); } }, () => {});

  /* ---------------------------------------------------------------- Fig. 6 strip */
  function strip(task) {
    const E = D.strip[task];
    const mark = (ok) => `<span class="${ok ? "ok" : "no"}">${ok ? "✓" : "✗"}</span>`;
    const ref = task === "c4" ? "reference: any value-preserving column" : `reference: ${E.answer}`;
    $("#strip").innerHTML = `
      <figure><img src="${E.cond}" alt="input"><figcaption><b>input</b> · ${E.side}<br><span class="tiny">${ref}</span></figcaption></figure>
      <figure class="gold"><img src="${E.one_step_enh}" alt="one-step output, contrast-enhanced"><figcaption><b>one step</b> (contrast &times;${E.gain})<br>reads ${E.read} ${mark(E.read_ok)}</figcaption></figure>
      <div class="sep"></div>
      ${E.samples.map((s, k) => `<figure><img src="${s.img}" alt="20-step sample, seed ${k}"><figcaption>seed ${k}<br>${s.move} ${mark(s.ok)}</figcaption></figure>`).join("")}`;
    $$("#stripTabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.task === task));
  }
  $$("#stripTabs button").forEach((b) => b.addEventListener("click", () => strip(b.dataset.task)));
  strip("chess");

  /* ---------------------------------------------------------------- minimal pair (Fig. 5) */
  const PB = makeBoard($("#pairBoard"));
  const PS = { which: "orig", ov: "move" };
  const heatCol = (v) => { // #f6d6a8 -> #e0702f -> #b3175f
    const a = [246, 214, 168], b = [224, 112, 47], c = [179, 23, 95];
    const t = Math.max(0, Math.min(1, v));
    const [p, q, u] = t < 0.5 ? [a, b, t * 2] : [b, c, (t - 0.5) * 2];
    return `rgb(${p.map((x, i) => Math.round(x + (q[i] - x) * u)).join(",")})`;
  };
  function pairDraw() {
    const P = D.pair[PS.which];
    PB.clearMarks(); PB.frame(null); PB.setFen(P.fen);
    PB.paintSquares(PS.ov === "occ");
    if (PS.ov === "move") { PB.mark(P.from, BLUE, 0.85, false); PB.mark(P.to, RED, 0.85, false); }
    else {
      for (const [s, v] of Object.entries(P.occ)) if (v >= 0.1) PB.mark(s, heatCol(v), 0.9, false);
      PB.outline(P.from, BLUE, "", 3.5); PB.outline(P.to, RED, "", 3.5);
    }
    if (PS.which === "white") PB.outline("c3", "#1f1f1f", "5 4", 2.5);
    $("#pairScale").hidden = PS.ov !== "occ";
    const txt = PS.which === "orig" ? `model plays <b>${P.san}</b>, a rook sacrifice that forces mate in three` :
      `the bishop on c3 guards h8: model plays <b>${P.san}</b>, mate in one`;
    $("#pairStatus").innerHTML = `<span class="badge model">${PS.which === "orig" ? "original" : "+ bishop c3"}</span><span>${txt} <span class="ok">✓</span></span>`;
    $("#pairReal").src = P.decide_frame;
    $("#pairRealCap").textContent = "The video model's unaltered first Decide frame for this board (green dots: the legal squares of the moving piece).";
  }
  $$("[data-pair]").forEach((b) => b.addEventListener("click", () => {
    PS.which = b.dataset.pair; $$("[data-pair]").forEach((x) => x.setAttribute("aria-pressed", x === b)); pairDraw();
  }));
  $$("[data-ov]").forEach((b) => b.addEventListener("click", () => {
    PS.ov = b.dataset.ov; $$("[data-ov]").forEach((x) => x.setAttribute("aria-pressed", x === b)); pairDraw();
  }));
  pairDraw();

  /* ---------------------------------------------------------------- Connect Four clip */
  const C4 = D.c4, COLS = "abcdefg";
  stepper($("#c4Stepper"), C4.frames.map((src, i) => {
    const lab = C4.labels[i];
    const isD = lab === "Decide";
    let label = i === 0 ? `${C4.mover} to move. Only column ${COLS[C4.best[0]]} keeps the win; the one-move heuristic plays ${COLS[C4.heur]}.` :
      isD ? (i === 1 ? `Decide: the disc lands in column <b>${COLS[C4.video_col]}</b> ✓ (the move that is played)` : "Decide: the next move of the imagined line") : "Imagine: the board after it";
    return { src, tag: i === 0 ? "input" : lab, tagClass: isD ? "decide" : i ? "imagine" : "", label, dur: isD ? 1400 : 900 };
  }), { hold: 2400, alt: "Connect Four clip generated by the video model" });

  /* ---------------------------------------------------------------- Go */
  $("#goCond").src = D.go.cond; $("#goOne").src = D.go.one_step;
  $("#goCondCap").innerHTML = `${D.go.side} · KataGo: <b>${D.go.best}</b>`;
  $("#goText").innerHTML = `Four full 20-step samples of this position play ${D.strip.go.samples.map((x) => `${x.move}${x.ok ? " ✓" : ""}`).join(", ")}: each noise draw commits to a single point, while the single step already marks KataGo's move most strongly.`;
  $("#goOneCap").innerHTML = `one-step output (contrast-enhanced): reads <b>${D.go.read}</b> <span class="ok">✓</span>`;

  /* ---------------------------------------------------------------- N-queens loop: vector redraw of the recorded drawings */
  const NQ = D.nqueens;
  function nqSVG(call, raw) {
    const N = NQ.N, C = 40, W = N * C;
    const placed = [];
    call.state.split("").forEach((ch, r) => { if (ch !== ".") placed.push([r, Number(ch)]); });
    const att = (a, b) => a[0] === b[0] || a[1] === b[1] || Math.abs(a[0] - b[0]) === Math.abs(a[1] - b[1]);
    const all = placed.concat(call.plan);
    const bad = new Set();
    for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) if (att(all[i], all[j])) { bad.add(i); bad.add(j); }
    let s = "";
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) s += `<rect x="${c * C}" y="${r * C}" width="${C}" height="${C}" fill="${(r + c) % 2 ? "#cdb595" : "#f2e6d2"}"/>`;
    const q = (r, c, fill, op = 1) => `<g transform="translate(${c * C + 2.5},${r * C + 2.5}) scale(.78)" opacity="${op}">${PIECES.q.replace(/fill:#000000/g, `fill:${fill}`).replace(/fill="#000"/g, `fill="${fill}"`)}</g>`;
    const [cr, cc] = call.cell;
    s += `<rect x="${cc * C + 1.5}" y="${cr * C + 1.5}" width="${C - 3}" height="${C - 3}" fill="rgba(15,157,107,.28)" stroke="#0f9d6b" stroke-width="3"/>`;
    placed.forEach(([r, c]) => { s += q(r, c, "#222"); });
    call.plan.forEach(([r, c], i) => { if (r !== cr || c !== cc) s += q(r, c, bad.has(placed.length + i) ? "#e0452f" : "#3f8fd2", 0.75); });
    s += q(cr, cc, "#0f9d6b");
    return `<svg viewBox="0 0 ${W} ${W}" style="width:100%;display:block;border-radius:6px">${s}</svg>`;
  }
  const nqEl = $("#nqStepper");
  nqEl.innerHTML = `<div class="seg" role="group" aria-label="View" style="margin-bottom:10px"><button aria-pressed="true" data-nq="vec">Read-out</button><button aria-pressed="false" data-nq="raw">Raw model drawing</button></div><div class="inner"></div>`;
  let nqMode = "vec";
  const nqFrames = () => NQ.calls.map((c, i) => {
    const solved = c.solved;
    const svg = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" ${nqSVG(c).slice(5)}`);
    return { src: nqMode === "raw" ? NQ.frames[i] : svg, tag: `call ${i + 1}`,
      label: `drawn board ${solved ? '<span class="ok">is a valid solution</span>' : '<span class="no">is not a valid solution</span>'}; its first new queen (green) is still completable and is placed`, dur: 1500 };
  });
  const nqBuild = () => stepper($(".inner", nqEl), nqFrames(), { hold: 2600, alt: "N-queens loop" });
  nqBuild();
  $$("[data-nq]", nqEl).forEach((b) => b.addEventListener("click", () => {
    nqMode = b.dataset.nq; $$("[data-nq]", nqEl).forEach((x) => x.setAttribute("aria-pressed", x === b)); nqBuild();
  }));
  $(".inner", nqEl).insertAdjacentHTML("afterend", `<p class="tiny" style="margin:8px 0 0">Dark: queens already on the board. Blue / red: the rest of the model's drawing (red = attacking). Read-out = strongest mark per empty row.</p>`);

  /* ---------------------------------------------------------------- billiards: drawn shot, then the simulator replay */
  const BIL = D.billiards;
  const cv = $("#bilCanvas"), cx = cv.getContext("2d");
  const SCALE = 232, X0 = 24, Y0 = 12, K = 2;
  const px = (x, y) => [(X0 + x * SCALE) * K, (Y0 + (1 - y) * SCALE) * K];
  const imgs = {};
  BIL.shots.forEach((s) => { imgs[s.id] = new Image(); imgs[s.id].src = s.decision; });
  const B = { i: 0, t: 0, phase: "decide", timer: null, visible: false };
  function table() {
    cx.fillStyle = "#1e2024"; cx.fillRect(0, 0, cv.width, cv.height);
    cx.fillStyle = "#683e1e"; cx.fillRect(12 * K, 0, 488 * K, 256 * K);
    cx.fillStyle = "#247648"; cx.fillRect(X0 * K, Y0 * K, 464 * K, 232 * K);
    cx.fillStyle = "#0c0c0c";
    BIL.pockets.forEach(([x, y], k) => { const [u, v] = px(x, y); cx.beginPath(); cx.arc(u, v, BIL.pocket_r[k] * SCALE * K, 0, 7); cx.fill(); });
    cx.fillStyle = "#424248"; cx.fillRect(24 * K, 264 * K, 464 * K, 16 * K);
  }
  function simFrame(s, t) {
    table();
    const fr = s.frames, n = fr[0].length;
    cx.fillStyle = "#00e1ff"; cx.globalAlpha = 0.55; cx.fillRect(24 * K, 266 * K, 464 * s.power * K, 12 * K); cx.globalAlpha = 1;
    cx.setLineDash([7, 7]); cx.strokeStyle = "rgba(255,255,255,.75)"; cx.lineWidth = 2.2; cx.beginPath();
    for (let k = 0; k <= t; k++) { if (!s.alive[k][0]) break; const [u, v] = px(...fr[k][0]); k ? cx.lineTo(u, v) : cx.moveTo(u, v); }
    cx.stroke(); cx.setLineDash([]);
    for (let b = n - 1; b >= 0; b--) {
      if (!s.alive[t][b]) continue;
      const [u, v] = px(...fr[t][b]);
      cx.beginPath(); cx.arc(u, v, BIL.R * SCALE * K, 0, 7); cx.fillStyle = BIL.colors[s.colors[b]]; cx.fill();
      cx.lineWidth = 1.4; cx.strokeStyle = "rgba(0,0,0,.45)"; cx.stroke();
    }
    const pk = s.pocket[1];
    if (pk >= 0 && !s.alive[t][1]) {
      const [u, v] = px(...BIL.pockets[pk]);
      cx.beginPath(); cx.arc(u, v, BIL.pocket_r[pk] * SCALE * K + 8, 0, 7); cx.strokeStyle = "#f2c14e"; cx.lineWidth = 5; cx.stroke();
    }
  }
  function bilLabel(s) {
    $("#bilLab").innerHTML = B.phase === "decide"
      ? `<span><span class="pill">model output</span> the drawn shot: aim line, cue stick, power bar</span>`
      : `<span><span class="pill">simulator</span> decoded shot ${s.angle.toFixed(1)}°, power ${s.power.toFixed(2)}: ${s.success ? '<span class="ok">red ball potted ✓</span>' : '<span class="no">missed</span>'}</span>`;
  }
  function bilRun() {
    clearTimeout(B.timer);
    if (!B.visible) return;
    const s = BIL.shots[B.i];
    if (B.phase === "decide") {
      const im = imgs[s.id];
      const drawIm = () => { cx.drawImage(im, 0, 0, cv.width, cv.height); };
      if (im.complete) drawIm(); else im.onload = drawIm;
      bilLabel(s);
      B.timer = setTimeout(() => { B.phase = "sim"; B.t = 0; bilRun(); }, 2300);
    } else {
      simFrame(s, B.t); bilLabel(s);
      if (B.t < s.frames.length - 1) { B.t += 1; B.timer = setTimeout(bilRun, 1000 / 30); }
      else B.timer = setTimeout(() => { B.phase = "decide"; bilRun(); }, 2200);
    }
  }
  $("#bilTabs").innerHTML = BIL.shots.map((s, i) => `<button role="tab" aria-selected="${i === 0}" data-shot="${i}">${s.shot}</button>`).join("");
  $$("#bilTabs button").forEach((b) => b.addEventListener("click", () => {
    B.i = Number(b.dataset.shot); B.phase = "decide";
    $$("#bilTabs button").forEach((x) => x.setAttribute("aria-selected", x === b)); bilRun();
  }));
  imgs[BIL.shots[0].id].onload = () => { if (!B.visible) cx.drawImage(imgs[BIL.shots[0].id], 0, 0, cv.width, cv.height); };
  bilLabel(BIL.shots[0]);
  whenVisible(cv, () => { B.visible = true; bilRun(); }, () => { B.visible = false; clearTimeout(B.timer); });

  /* ---------------------------------------------------------------- BibTeX copy */
  $("#copyBib").addEventListener("click", async () => {
    const t = $("#bib").textContent;
    try { await navigator.clipboard.writeText(t); }
    catch (e) { const r = document.createRange(); r.selectNodeContents($("#bib")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand("copy"); }
    $("#copyBib").textContent = "Copied"; setTimeout(() => { $("#copyBib").textContent = "Copy"; }, 1600);
  });
  $$('a[aria-disabled="true"]').forEach((a) => a.addEventListener("click", (e) => e.preventDefault()));
})();
