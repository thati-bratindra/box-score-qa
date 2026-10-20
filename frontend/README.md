# Frontend

Angular 18 chat UI for the Q&A pipeline. It calls the backend at `http://localhost:8000/api/chat` (see [`src/app/services/base.service.ts`](src/app/services/base.service.ts)), so start the backend first or the UI will show an error banner when you send a message.

## Run it

Requires Node 18, 20, or 22 (`node --version`).

```bash
npm install
npm start
```

Then open `http://localhost:4200/`. The dev server reloads automatically when source files change.

## Other commands

* `npm run build` builds to `dist/`.
* `npm test` runs the unit tests via Karma.
