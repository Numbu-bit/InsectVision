# UI screenshots (phase 3)

Rendered from the running app with a real Chromium browser at three viewport widths.
The interface is the light green-and-white theme from phases 1 and 2, laid out as a
"stage + sidebar" dashboard: the stage on the left shows the photo, the live camera or
the analysed result with an instrument strip of counts under it; the sidebar holds the
verdict, the detections, the species reference and the model facts.

| File | Viewport | Shows |
|---|---|---|
| `laptop-scan.png` | 1440 px | Empty stage, ready for a photo |
| `laptop-result.png` | 1440 px | `beetle.jpg` analysed: boxed photo, counts, verdict, detection card |
| `laptop-not-an-insect.png` | 1440 px | A tree photo: neutral verdict, nothing counted |
| `laptop-camera.png` | 1440 px | Live camera tab before permission is granted |
| `tablet-scan.png` | 820 px | Single column |
| `phone-scan.png` | 390 px (full page) | Stacked layout |
| `phone-result.png` | 390 px (full page) | Result on a phone |

Regenerate by resizing the browser window on <http://127.0.0.1:8000>, or with the
puppeteer-core script used during development.
