
# Usisivac V6 — Univerzalni Autonomni Multi-Agent Sistem

<p align="center">
  <img src="https://i.imgur.com/b09f3a2.png" alt="Usisivac V6 Banner" width="800">
</p>

<p align="center">
  <b>"Usisivač" koji ne samo da prikuplja znanje, već ga pretvara u produkcioni kod.</b>
</p>

---

**Usisivac V6** je vrhunac autonomije u okviru **Trinity Protocol** ekosistema. Ovo je univerzalni, autonomni multi-agent sistem dizajniran za rešavanje bilo kog data science problema — od takmičarskog mašinskog učenja do kompleksnih industrijskih izazova. Sistem radi u non-stop petlji, pokreće se iz Gemini CLI terminala, i konstantno uči i unapređuje svoja rešenja.

Ključne karakteristike:
- **Univerzalnost**: Nije ograničen na Kaggle; primenjiv na bilo koji data science problem (Tabular, NLP, CV).
- **Autonomija**: Radi u non-stop petlji, samostalno istražuje, kodira, čisti, izvršava i audituje.
- **Anti-Simulacija**: Ugrađen **ANTI_SIMULATION_v3** mehanizam koji garantuje da se svaka akcija (trening, ingest, audit) stvarno izvršava, sa kriptografskim dokazom (`proof_registry.jsonl`).
- **Neuralni Filter**: Koristi "Veliki Filter" — neuronsku mrežu koja filtrira i rangira znanje iz ChromaDB baze za maksimalnu ekstrakciju relevantnosti.
- **Free API**: Dizajniran da radi sa besplatnim API ključevima (Groq, Mistral, Gemini, OpenRouter).
- **Tri-Way Relay**: Omogućava kolaboraciju između **Gemini CLI**, **Claude (preko Cline)** i **VS Code**.
- **Guardian Audit**: **JudgeGuard v2.1** vrši automatski audit svake iteracije, mereći `drift_score` i integritet artefakata.

## 🏛️ Arhitektura: Trinity Protocol

Sistem se sastoji od specijalizovanih agenata koji sarađuju putem centralnog **Orchestrator**-a u non-stop petlji.

| Agent           | Port | Uloga                                               |
|-----------------|------|-----------------------------------------------------|
| **Orchestrator**  | 8081 | Centralni mozak, pokreće pipeline u petlji          |
| **ResearchAgent** | 8084 | Usisava znanje, izvlači "Golden Recipe"            |
| **CriticAgent**   | 8085 | Čuvar kvaliteta, detektuje anti-patterne i zablude  |
| **CoderAgent**    | 8082 | Samostalno piše Python kod na osnovu research-a     |
| **CleanerAgent**  | 8086 | Automatski čisti i normalizuje podatke              |
| **FeatureAgent**  | 8087 | Dinamički izvršava feature engineering kod          |
| **Guardian**      | 8083 | QA Auditor, meri `drift_score`, verifikuje dokaze   |
| **Relay**         | 8088 | Tri-way chat za Gemini ↔ Claude ↔ Cline           |

### Pipeline Petlja

Orchestrator izvršava sledeći pipeline u svakoj iteraciji:

1.  **ResearchAgent**: Usisava znanje iz ChromaDB-a i interneta, izvlači "Golden Recipe" za dati problem.
2.  **CriticAgent**: Kritikuje plan i nalaze ResearchAgent-a.
3.  **CoderAgent**: Piše kompletan, izvršiv Python kod na osnovu research-a i kritike.
4.  **CriticAgent**: Ponovo kritikuje, ovog puta generisani kod.
5.  **CleanerAgent**: Generiše i **stvarno izvršava** skriptu za čišćenje podataka.
6.  **FeatureAgent**: **Stvarno izvršava** feature engineering skriptu.
7.  **Guardian**: Vrši kompletan audit, izračunava `drift_score`, verifikuje sve dokaze u `proof_registry.jsonl`.
8.  **Self-Healing**: Ako `drift_score > 0.4`, Guardian šalje feedback za korekciju u sledećoj iteraciji.
9.  **LOOP**: Proces se ponavlja, učeći iz prethodne iteracije.

## 🚀 Pokretanje Sistema

Sistem je dizajniran za pokretanje iz komandne linije, idealno unutar VS Code terminala.

### 1. Instalacija

```bash
# Kloniraj repozitorijum
git clone https://github.com/your-repo/Usisivac-V6.git
cd Usisivac-V6

# Instaliraj zavisnosti
pip install -r requirements.txt
```

### 2. Konfiguracija (Antigravity IDE)

Pokreni `antigravity_setup.py` da automatski generišeš sve potrebne konfiguracije.

```bash
python config/antigravity_setup.py
```

Ova skripta će:
- Kreirati `.env` fajl. **Moraš uneti svoje besplatne API ključeve**.
- Kreirati `.gitignore`.
- Verifikovati `.vscode/settings.json` i `.clinerules`.
- Generisati `config/gemini_config.json`.

### 3. Pokretanje A2A Servera

Svi agenti komuniciraju preko lokalnih HTTP servera. Pokreni ih sve odjednom:

```bash
python orchestrator/a2a_servers.py
```

### 4. Pokretanje Orchestrator-a (Non-Stop Mod)

Ovo je glavni mod rada. Orchestrator će raditi u beskonačnoj petlji, rešavajući problem.

```bash
# Primer: rešavanje problema predikcije cena kuća
python orchestrator/orchestrator.py "Predict house prices based on location, size, and age" --domain tabular --data-path "data/train.csv"
```

- **`problem`**: Opis problema na prirodnom jeziku.
- **`--domain`**: Domen problema (`tabular`, `nlp`, `cv`, `universal`).
- **`--data-path`**: (Opciono) Putanja do CSV fajla sa podacima.
- **`--max-iter`**: (Opciono) Maksimalan broj iteracija.
- **`--delay`**: (Opciono) Pauza između iteracija u sekundama.

Da zaustaviš, pritisni `Ctrl+C`.

## ⚙️ Jezgro Sistema

| Fajl                      | Opis                                                                      |
|---------------------------|---------------------------------------------------------------------------|
| `core/anti_simulation.py` | **ANTI_SIM v3**: Garantuje stvarno izvršavanje akcija putem `proof_registry`. |
| `core/neural_filter.py`   | **Veliki Filter**: Neuronska mreža za rangiranje znanja iz RAG-a.           |
| `core/rag_engine.py`      | **RAG Motor**: Upravlja ChromaDB kolekcijama i JSON fallback-om.            |
| `core/llm_client.py`      | **LLM Klijent**: Podržava Groq, Mistral, Gemini, OpenRouter sa fallback-om. |
| `core/state_manager.py`   | **State Manager**: Upravlja deljenim stanjem sistema (`work_share_state.json`). |

## 🤝 Tri-Way Relay & Gemini CLI

Sistem je dizajniran za interakciju sa **Gemini CLI** i **Cline** (VS Code ekstenzija za Claude).

- **Praćenje rada**: `tail -f logs/work_log.md`
- **Slanje poruke**: `curl -X POST http://localhost:8088/relay -d '{"from":"gemini","to":"cline","message":"Proveri drift score."}'`
- **Provera stanja**: `curl http://localhost:8081/status`

Sva uputstva za konfiguraciju nalaze se u `config/GEMINI.md`.

---
*Usisivac V6 @ Trinity Protocol — Stvarnost ispred simulacije.*
