# StockVision — placeholder frontend

Static HTML/CSS/JS landing page. No build step, no dependencies.

## Deploy to Vercel

**Option A — Vercel dashboard**
1. Push this folder to a GitHub repo.
2. Go to vercel.com → Add New Project → import the repo.
3. Framework preset: "Other" (Vercel will detect it as static). Leave build command empty, output directory as root.
4. Deploy.

**Option B — Vercel CLI**
```bash
npm i -g vercel
cd stockvision
vercel
```
Follow the prompts — no config needed since there's no build step.

## Files
- `index.html` — page structure and content
- `styles.css` — all styling
- `script.js` — populates the scrolling ticker tape (static demo data only)

Swap the content in `index.html` and the demo prices in `script.js` whenever the real backend/model is ready.
