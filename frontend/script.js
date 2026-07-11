// Purely cosmetic ticker tape — static demo values, no live data source.
const tickers = [
  { sym: "AAPL", price: "228.41", delta: "+1.2%", up: true },
  { sym: "TSLA", price: "241.09", delta: "-0.8%", up: false },
  { sym: "NVDA", price: "134.77", delta: "+2.4%", up: true },
  { sym: "MSFT", price: "452.30", delta: "+0.3%", up: true },
  { sym: "AMZN", price: "198.55", delta: "-0.4%", up: false },
  { sym: "GOOGL", price: "179.02", delta: "+0.9%", up: true },
  { sym: "META", price: "512.88", delta: "+1.6%", up: true },
  { sym: "NIFTY50", price: "24,812", delta: "+0.5%", up: true },
  { sym: "SENSEX", price: "81,245", delta: "-0.2%", up: false },
];

function buildTicker() {
  const el = document.getElementById("ticker");
  if (!el) return;

  const row = tickers
    .map(
      (t) =>
        `<span>${t.sym} <span class="${t.up ? "up" : "down"}">${t.price} ${t.delta}</span></span>`
    )
    .join("");

  // duplicate content once so the CSS scroll loop has no visible seam
  el.innerHTML = row + row;
}

buildTicker();
