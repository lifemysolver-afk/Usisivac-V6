# Gemini CLI - Konfiguracija za Usisivac V6

## Instalacija Gemini CLI

```bash
npm install -g @anthropic-ai/claude-code  # Cline alternative
npm install -g @anthropic-ai/gemini-cli   # Ako postoji
# ILI koristi Google AI Studio CLI:
pip install google-generativeai
```

## Pokretanje iz terminala

### Non-stop mod (beskonacna petlja)
```bash
cd /path/to/Usisivac-V6
python orchestrator/orchestrator.py "Predict house prices" --domain tabular --delay 30
```

### Jedan prolaz
```bash
python orchestrator/orchestrator.py "Classify text sentiment" --domain nlp --once
```

### Sa podacima
```bash
python orchestrator/orchestrator.py "Predict churn" --domain tabular --data "Customer data with 50 features" --data-path data/customers.csv
```

### A2A serveri (svi agenti na HTTP)
```bash
python orchestrator/a2a_servers.py
# Orchestrator  http://localhost:8081
# CoderAgent    http://localhost:8082
# Guardian      http://localhost:8083
# Research      http://localhost:8084
# Critic        http://localhost:8085
# Cleaner       http://localhost:8086
# Feature       http://localhost:8087
# Relay         http://localhost:8088
```

## Gemini CLI u VS Code terminalu

1. Otvori VS Code terminal (Ctrl+`)
2. Pokreni Usisivac V6:
```bash
python orchestrator/orchestrator.py "Your problem here"
```
3. Gemini CLI prati work_log.md:
```bash
tail -f logs/work_log.md
```
4. Relay poruke:
```bash
curl -X POST http://localhost:8088/relay \
  -H "Content-Type: application/json" \
  -d '{"from":"gemini","to":"claude","message":"Analyze drift results"}'
```

## Environment Variables

Kopiraj `.env.example` u `.env` i popuni kljuceve:
```bash
cp .env.example .env
# Edituj .env sa svojim API kljucevima
```
