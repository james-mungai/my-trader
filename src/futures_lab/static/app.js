const tokenInput = document.querySelector("#apiToken");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const resetBtn = document.querySelector("#resetBtn");

const els = Object.fromEntries(
  [
    "runtimeLine",
    "regimePill",
    "midPrice",
    "spread",
    "dataAge",
    "rangePos",
    "rangePct",
    "flow10",
    "bookImb",
    "actionPill",
    "confidence",
    "mode",
    "entry",
    "takeProfit",
    "stopLoss",
    "notional",
    "reason",
    "riskPill",
    "riskReason",
    "riskBlockers",
    "targetPill",
    "pnl",
    "trades",
    "openSide",
    "openEntry",
    "openTp",
    "lastTrade",
  ].map((id) => [id, document.querySelector(`#${id}`)])
);

function headers() {
  const token = tokenInput.value.trim();
  return token ? { "X-API-Key": token } : {};
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function money(value) {
  return value === null || value === undefined ? "--" : `$${Number(value).toFixed(2)}`;
}

function price(value) {
  return value === null || value === undefined ? "--" : Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function pct(value) {
  return value === null || value === undefined ? "--" : `${(Number(value) * 100).toFixed(3)}%`;
}

function num(value, digits = 2) {
  return value === null || value === undefined ? "--" : Number(value).toFixed(digits);
}

function setPill(el, text, kind) {
  el.textContent = text || "--";
  el.className = `pill ${kind || ""}`.trim();
}

function paint(data) {
  const market = data.market;
  const decision = data.decision;
  const risk = data.risk;
  const paper = data.paper;

  els.runtimeLine.textContent = market
    ? `${market.symbol} · ${market.connected ? "stream connected" : "stream stopped"} · observed ${num(market.observed_seconds, 0)}s`
    : "Runtime not started";

  setPill(els.regimePill, market?.regime || "unknown", market?.regime === "sideways" ? "good" : market?.regime === "volatile" ? "bad" : "warn");
  els.midPrice.textContent = price(market?.mid_price);
  els.spread.textContent = `${num(market?.spread_bps, 3)} bps`;
  els.dataAge.textContent = `${num(market?.data_age_seconds, 2)}s`;
  els.rangePos.textContent = pct(market?.range_position_180s);
  els.rangePct.textContent = pct(market?.range_180s_pct);
  els.flow10.textContent = pct(market?.taker_buy_ratio_10s);
  els.bookImb.textContent = num(market?.book_imbalance_top, 3);

  const action = decision?.action || "wait";
  setPill(els.actionPill, action, action.includes("propose") ? "good" : "warn");
  els.confidence.textContent = pct(decision?.confidence);
  els.mode.textContent = decision?.mode || "--";
  els.entry.textContent = price(decision?.entry_price);
  els.takeProfit.textContent = price(decision?.take_profit_price);
  els.stopLoss.textContent = price(decision?.stop_loss_price);
  els.notional.textContent = money(decision?.notional_usd);
  els.reason.textContent = decision?.reason || "--";

  setPill(els.riskPill, risk?.allowed ? "allowed" : "blocked", risk?.allowed ? "good" : "bad");
  els.riskReason.textContent = risk?.reason || "--";
  els.riskBlockers.innerHTML = "";
  for (const blocker of risk?.blockers || []) {
    const li = document.createElement("li");
    li.textContent = blocker;
    els.riskBlockers.appendChild(li);
  }

  const pnl = paper?.realized_pnl_usd ?? 0;
  setPill(els.targetPill, paper?.daily_target_hit ? "target hit" : paper?.daily_max_loss_hit ? "max loss" : "active", paper?.daily_target_hit ? "good" : paper?.daily_max_loss_hit ? "bad" : "");
  els.pnl.textContent = money(pnl);
  els.pnl.style.color = pnl > 0 ? "var(--good)" : pnl < 0 ? "var(--bad)" : "var(--text)";
  els.trades.textContent = paper?.trades_today ?? "--";
  els.openSide.textContent = paper?.open_position?.side || "--";
  els.openEntry.textContent = price(paper?.open_position?.entry_price);
  els.openTp.textContent = price(paper?.open_position?.take_profit_price);
  els.lastTrade.textContent = paper?.last_trade ? `${paper.last_trade.exit_reason} ${money(paper.last_trade.net_pnl_usd)}` : "--";
}

async function refresh() {
  try {
    const data = await api("/latest");
    paint(data);
  } catch (error) {
    els.runtimeLine.textContent = error.message;
  }
}

startBtn.addEventListener("click", async () => {
  await api("/runtime/start", { method: "POST" });
  await refresh();
});

stopBtn.addEventListener("click", async () => {
  await api("/runtime/stop", { method: "POST" });
  await refresh();
});

resetBtn.addEventListener("click", async () => {
  await api("/paper/reset", { method: "POST" });
  await refresh();
});

refresh();
setInterval(refresh, 1000);
