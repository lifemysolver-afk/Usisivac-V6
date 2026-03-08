
# Usisivac V6 — Univerzalni Autonomni Multi-Agent Sistem

<p align="center">
  <b>"Usisivač" koji ne samo da prikuplja znanje, već ga pretvara u produkcioni kod — sa nultom simulacijom.</b>
</p>

---

**Usisivac V6** je vrhunac autonomije u okviru **Trinity Protocol** ekosistema. Ovo je univerzalni, autonomni multi-agent sistem dizajniran za rešavanje bilo kog data science problema — od takmičarskog mašinskog učenja do kompleksnih industrijskih izazova. Sistem radi u non-stop petlji, pokreće se iz Gemini CLI terminala, i konstantno uči i unapređuje svoja rešenja.

Ključne karakteristike:
- **Univerzalnost**: Nije ograničen na Kaggle; primenjiv na bilo koji data science problem (Tabular, NLP, CV).
- **Autonomija**: Radi u non-stop petlji, samostalno istražuje, kodira, čisti, izvršava i audituje.
- **Anti-Simulacija**: Ugrađen **ANTI_SIMULATION_v3** mehanizam koji garantuje da se svaka akcija (trening, ingest, audit) stvarno izvršava, sa kriptografskim dokazom (`proof_registry.jsonl`).
- **Neural Discussion Engine**: Proponent (Groq) brani, Opponent (Mistral) napada, Moderator presudi — pre nego što znanje uđe u RAG.
- **LopticaModule**: 3-6-2 state machine sa KnowledgeBase (SQLite), ConflictResolver, FeedbackTracker i VetoBoard (5-persona quorum).
- **BrainMassIngest**: ChromaDB masovni ingest svih `.py`, `.md`, `.json`, `.yaml` fajlova iz projekta.
- **Neuralni Filter**: Koristi "Veliki Filter" — neuronsku mrežu koja filtrira i rangira znanje iz ChromaDB baze za maksimalnu ekstrakciju relevantnosti.
- **Free API**: Dizajniran da radi sa besplatnim API ključevima (Groq, Mistral, Gemini, OpenRouter) uz automatsku Key Rotation.
- **Tri-Way Relay**: Omogućava kolaboraciju između **Gemini CLI**, **Claude (preko Cline)** i **VS Code**.
- **Guardian Audit**: **JudgeGuard v2.1** vrši automatski audit svake iteracije, mereći `drift_score` i integritet artefakata.

## Arhitektura: Trinity Protocol

Sistem se sastoji od specijalizovanih agenata koji sarađuju putem centralnog **Orchestrator**-a u non-stop petlji.

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Non-Stop Loop)              │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ LopticaModule│──▶│  Discussion  │──▶│  Research     │    │
│  │ (3-6-2 SM)   │   │  Engine      │   │  Agent        │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ VetoBoard    │   │ Critic       │──▶│ Coder        │    │
│  │ (5 Persona)  │   │ Agent        │   │ Agent        │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Knowledge    │   │ Cleaner      │──▶│ Feature      │    │
│  │ Base (SQLite)│   │ Agent        │   │ Agent        │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│                           │                                 │
│                           ▼                                 │
│                    ┌──────────────┐                          │
│                    │  GUARDIAN    │                          │
│                    │ (drift_score)│                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

| Agent           | Port | Uloga                                               | LLM Provider |
|-----------------|------|-----------------------------------------------------|---|
| **Orchestrator**  | 8081 | Centralni mozak, pokreće pipeline u petlji          | — |
| **ResearchAgent** | 8084 | Usisava znanje, izvlači "Golden Recipe"            | Groq (Llama 3.3 70B) |
| **CriticAgent**   | 8085 | Čuvar kvaliteta, detektuje anti-patterne i zablude  | Mistral Small |
| **CoderAgent**    | 8082 | Samostalno piše Python kod na osnovu research-a     | Groq (Llama 3.3 70B) |
| **CleanerAgent**  | 8086 | Automatski čisti i normalizuje podatke              | Mistral Small |
| **FeatureAgent**  | 8087 | Dinamički izvršava feature engineering kod          | N/A (executor) |
| **Proponent**     | —    | Brani relevantnost znanja u Discussion Engine-u     | Groq (Llama 3.3 70B) |
| **Opponent**      | —    | Napada i kritikuje relevantnost znanja              | Mistral Small |
| **Moderator**     | —    | Donosi finalnu odluku: INGEST / PARTIAL / REJECT   | Groq (Llama 3.3 70B) |
| **Guardian**      | 8083 | QA Auditor, meri `drift_score`, verifikuje dokaze   | N/A (auditor) |
| **Relay**         | 8088 | Tri-way chat za Gemini ↔ Claude ↔ Cline           | — |

### Pipeline Petlja (11 koraka)

```
 1. LopticaModule    → 3-6-2 state machine + KB pretraga + VetoBoard
 2. Discussion       → Proponent brani, Opponent napada, Moderator presudi
 3. ResearchAgent    → ChromaDB RAG + Golden Recipe
 4. CriticAgent      → Kritikuje plan i nalaze
 5. CoderAgent       → Autonomno generisanje Python koda
 6. CriticAgent      → Code review (second pass)
 7. CleanerAgent     → Statisticko čišćenje podataka
 8. FeatureAgent     → Dinamičko feature engineering
 9. Guardian         → drift_score audit + proof_registry verifikacija
10. Self-Healing     → Ako drift > 0.4, feedback za korekciju
11. → LOOP           → Ponovi sa novim znanjem
```

---

## LopticaModule — Trinity Integration

LopticaModule je integrisana komponenta iz **Trinity_AIMO_Loptica** projekta koja donosi napredne mehanizme za upravljanje znanjem.

### 3-6-2 State Machine

```
RESEARCH (3 akcije) → DESIGN (6 akcija) → IMPLEMENTATION (6 akcija)
    → VALIDATION (2 akcije) → SYNTHESIS → COMPLETE
```

Svaka faza ima checkpoint limit — engine automatski napreduje kada se dostigne.

### KnowledgeBase (SQLite)

- **Solutions**: Takmičenja, rankovi, autori
- **Techniques**: Hiperparametri sa `confidence` skorom, `rich_context` (JSON) i `domain` tagom
- **ConflictResolver**: Detektuje HARD konflikte (npr. `high_learning_rate` + `no_warmup`) i SOFT duplikate
- **FeedbackTracker**: Self-learning — automatski prilagođava confidence na osnovu rezultata takmičenja
- **NotebookParser**: AST ekstrakcija hiperparametara iz `.ipynb` notebooka
- **HarvesterAnalytics**: Izveštaji, snapshoti i statistike

### VetoBoard (5-Persona Quorum)

Pet virtuelnih persona glasaju o svakoj akciji:

| Persona | Fokus |
|---|---|
| CEO | Strateška vrednost |
| CTO | Tehnička izvodljivost |
| CFO | Troškovi i resursi |
| LEGAL | Bezbednost i usklađenost |
| CRITIC | Devil's advocate |

Ako 3+ persona glasaju VETO, akcija se blokira.

### BrainMassIngest (ChromaDB)

Masovni ingest svih `.py`, `.md`, `.json`, `.yaml` fajlova iz projekta u ChromaDB kolekciju `massive_brain`. Podržava chunking, deduplication i batch upload.

```python
from loptica.brain_mass_ingest import BrainMassIngest
ingestor = BrainMassIngest()
result = ingestor.ingest("/path/to/project")
# → {"status": "OK", "chunks_added": 1247, "total_in_db": 1247}
```

---

## Neural Discussion Engine

Pre nego što znanje uđe u RAG, prolazi kroz debatu:

```
1. Proponent (Groq/Llama 3.3 70B)  → Brani relevantnost
2. Opponent (Mistral Small)         → Napada i kritikuje
3. Moderator (Groq/Llama 3.3 70B)  → Presuda: INGEST / PARTIAL / REJECT
```

Sve diskusije se čuvaju u:
- **ChromaDB** (`discussion_db`) — za semantičku pretragu prethodnih debata
- **JSONL** (`logs/discussion_log.jsonl`) — za audit trail

---

## Anti-Simulation v3

Stroga zabrana simulacije. Agenti **ne smeju** da pišu:

```
ZABRANJENE FRAZE:
- "trening završen"    (ako model nije stvarno istreniran)
- "BLEU 19.1"          (ako metrika nije stvarno izmerena)
- "audit pass"         (ako audit nije stvarno pokrenut)
- "model spreman"      (ako fajl ne postoji na disku)
```

Svaka akcija generiše SHA-256 dokaz u `logs/proof_registry.jsonl`:

```json
{
  "timestamp": "2026-03-08T14:30:00",
  "agent": "CoderAgent",
  "action": "CODE_GENERATED",
  "details": "file=src/generated/solution.py lines=245",
  "proof_hash": "a3f2b1c4d5e6..."
}
```

---

## Direktorijumska Struktura

```
Usisivac-V6/
├── core/                       # Jezgro sistema
│   ├── anti_simulation.py      # Anti-Simulation v3 enforcement
│   ├── llm_client.py           # LLM klijent (Groq + Mistral + fallback)
│   ├── key_rotator.py          # Automatska rotacija API ključeva
│   ├── rag_engine.py           # ChromaDB RAG engine
│   ├── neural_filter.py        # Neuronski filter za relevantnost
│   ├── state_manager.py        # Deljeno stanje sistema
│   └── discussion_engine.py    # ChromaDB + JSONL za diskusije
│
├── agents/                     # Specijalizovani agenti
│   ├── research_agent.py       # ResearchAgent (knowledge vacuum)
│   ├── critic_agent.py         # CriticAgent (quality guard)
│   ├── coder_agent.py          # CoderAgent (autonomous coder)
│   ├── cleaner_agent.py        # CleanerAgent (statistical cleaner)
│   ├── feature_agent.py        # FeatureAgent (dynamic executor)
│   └── discussion_agents.py    # Proponent + Opponent + Moderator
│
├── loptica/                    # LopticaModule (Trinity Integration)
│   ├── loptica_engine.py       # 3-6-2 state machine
│   ├── loptica_module.py       # Unified integration layer
│   ├── veto_board.py           # 5-persona VetoBoard quorum
│   ├── knowledge_base.py       # SQLite KB + ConflictResolver + Feedback
│   └── brain_mass_ingest.py    # ChromaDB masovni ingest
│
├── orchestrator/               # Centralni Orchestrator
│   ├── orchestrator.py         # Non-stop pipeline loop (11 koraka)
│   └── a2a_servers.py          # A2A HTTP serveri za agente
│
├── guardian/                   # Guardian audit sistem
│   └── guardian.py             # drift_score + JudgeGuard v2.1
│
├── relay/                      # Tri-way relay
│   └── triway_relay.py         # Claude ↔ Gemini ↔ Cline komunikacija
│
├── config/                     # Konfiguracije
│   ├── GEMINI.md               # Gemini CLI uputstva
│   └── antigravity_setup.py    # Antigravity IDE auto-config
│
├── tests/                      # Test suite
│   ├── test_system.py          # Core system tests (12 tests)
│   ├── test_loptica_integration.py  # Loptica integration tests (9 tests)
│   └── test_apis.py            # API provider diagnostics
│
├── logs/                       # Logovi i audit trail
│   ├── work_log.md             # Unified work log
│   ├── proof_registry.jsonl    # Kriptografski dokazi
│   └── discussion_log.jsonl    # Diskusije (Proponent vs Opponent)
│
├── data/                       # Podaci
├── src/generated/              # Generisani kod
├── reports/                    # Izveštaji i submission fajlovi
├── .vscode/settings.json       # VS Code konfiguracija
├── .clinerules                 # Cline custom instructions
├── .env.example                # Template za API ključeve
├── requirements.txt            # Python zavisnosti
├── MASTER_PROMPT.md            # Master prompt za @mcp:sequential-thinking
└── README.md                   # Ovaj fajl
```

---

## Brzi Start

### 1. Kloniranje

```bash
git clone https://github.com/kiza1234568/Usisivac-V6.git
cd Usisivac-V6
```

### 2. Instalacija

```bash
pip install -r requirements.txt
```

### 3. API Ključevi

```bash
cp .env.example .env
# Unesite svoje besplatne ključeve:
# GROQ_API_KEY=gsk_...
# MISTRAL_API_KEY=...
# GEMINI_API_KEY=...
```

### 4. Antigravity IDE Config

```bash
python config/antigravity_setup.py
```

### 5. Testiranje

```bash
# Core testovi (12 tests)
python tests/test_system.py

# Loptica integration testovi (9 tests)
python tests/test_loptica_integration.py

# API dijagnostika
python tests/test_apis.py
```

### 6. Pokretanje

```bash
# Jedan prolaz
python orchestrator/orchestrator.py "Predict customer churn" --once --domain tabular

# Non-stop mod (preporučeno)
python orchestrator/orchestrator.py "Predict customer churn" --domain tabular --max-iter 10

# Sa podacima
python orchestrator/orchestrator.py "Predict churn" --domain tabular \
  --data "train.csv with 20 features" --data-path data/train.csv

# A2A serveri (za Gemini CLI i Cline)
python orchestrator/a2a_servers.py
```

### 7. Gemini CLI (iz VS Code terminala)

```bash
gemini run "Pokreni Usisivac V6 za tabular classification problem"
```

---

## Free API Provajderi

| Provider | Model | Besplatan? | Rate Limit |
|---|---|---|---|
| **Groq** | Llama 3.3 70B | Da | 30 req/min |
| **Mistral** | Mistral Small | Da | 60 req/min |
| **Gemini** | Gemini 2.0 Flash | Da | 15 req/min |
| **OpenRouter** | Razni | Da (free tier) | Varira |

Key Rotator automatski rotira ključeve kada se dostigne rate limit.

---

## Testovi

| Test Suite | Testovi | Status |
|---|---|---|
| `test_system.py` | 12 | ALL PASS |
| `test_loptica_integration.py` | 9 | ALL PASS |
| `test_apis.py` | 4 | Groq + Mistral PASS |
| **UKUPNO** | **25** | **ALL PASS** |

---

## Tri-Way Relay i Gemini CLI

Sistem je dizajniran za interakciju sa **Gemini CLI** i **Cline** (VS Code ekstenzija za Claude).

- **Praćenje rada**: `tail -f logs/work_log.md`
- **Slanje poruke**: `curl -X POST http://localhost:8088/relay -d '{"from":"gemini","to":"cline","message":"Proveri drift score."}'`
- **Provera stanja**: `curl http://localhost:8081/status`

Sva uputstva za konfiguraciju nalaze se u `config/GEMINI.md`.

---

## Licenca

MIT License. Slobodno koristite, modifikujte i distribuirajte.

---

> **ANTI_SIMULATION_v3**: Ovaj sistem ne simulira rezultate. Svaka akcija je verifikovana kriptografskim dokazom. Ako vidite "trening završen" u logu, to znači da je model **stvarno** istreniran.

---

*Usisivac V6 @ Trinity Protocol — Stvarnost ispred simulacije.*
