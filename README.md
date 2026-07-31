# Systematic Trading

Live MANU ticks from LSE → in-memory last price + bid/ask.

```powershell
copy env.example .env   # set LSE_API_KEY
docker compose up --build
```

- http://localhost:8000/docs
- `GET /health` · `GET /tick`
