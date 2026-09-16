# Browser interface

The static interface is served by FastAPI at `/` and `/static`.

| File | What it does |
|---|---|
| `index.html` | Defines the page structure: supported species reference, camera/file upload controls, crop preview, analysis button, rejected-photo view, cascade detection view, classifier-only top-match view, and status messages. |
| `app.js` | Runs the browser workflow. It loads health information, accepts camera/gallery/drag-and-drop images, enforces the client-side 10 MB check, handles cropping and zooming, sends the cropped image to `/api/v1/analyse`, draws detection boxes, and renders predictions/errors. The server repeats security-sensitive validation. |
| `style.css` | Provides the mobile-first layout, colours, buttons, crop overlay, result states, loading spinner, responsive sizing, and visible keyboard focus styles. |

## Browser flow

`index.html` starts with the upload state. `app.js` calls `/api/v1/health` and hides or enables parts of the page based on the returned mode. After a file is decoded, the user can use the crop box or submit the full image. Results are rendered differently for cascade mode and classifier-only mode.

The frontend has no build step or framework. Changes can be tested by running the FastAPI server and refreshing the browser.
