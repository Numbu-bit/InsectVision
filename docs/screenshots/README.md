# UI screenshots (phase 3)

Rendered from the running app with a real Chromium browser at three viewport widths.
The theme is locked to the light/green interface presented in phases 1 and 2.

| File | Viewport | Shows |
|---|---|---|
| `laptop-scan.png` | 1440 px | Desktop workspace: species strip on top, scanner left, results placeholder right |
| `laptop-result.png` | 1440 px | Same, after analysing `beetle.jpg` -- verdict banner, boxed photo, confidence |
| `tablet-scan.png` | 820 px | Single wider column |
| `phone-scan.png` | 390 px (full page) | Stacked capture buttons, species list on top |
| `phone-result.png` | 390 px (full page) | Result view on a phone |

Regenerate with `scratchpad/jsdom_env/screenshot.js` (puppeteer-core + local Chrome/Edge) or
simply resize the browser window on <http://127.0.0.1:8000>.
