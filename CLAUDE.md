# CLAUDE.md — Spend Analysis v3

Plataforma de classificação de gastos corporativos com loop de aprendizado humano. Consultores revisam classificações, correções alimentam uma Knowledge Base (KB) por projeto, mesclada com KB do setor (referência viva) para few-shot RAG nas classificações futuras.

**Domínio**: Spend Analysis / Procurement / Classificação taxonômica N1-N4
**Cliente**: PG Consultoria — AI Team

## Comandos

```bash
# Backend
uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt  # mesmo Python do Azure
cp local.settings.json.example local.settings.json  # preencher secrets
func start                    # Azure Functions local

# Frontend
cd frontend && npm install
cp .env.local.example .env.local
npm run dev                   # http://localhost:3000

# Testes — zero chamadas ao Grok/xAI
.venv/bin/python -m pytest tests/ -v                          # Backend (470 testes, ~9s)
cd frontend && npx jest --verbose                     # Frontend (60 testes, ~3s)
.venv/bin/python -m pytest tests/ --cov=src --cov-report=term-missing  # Coverage
```

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Backend | Azure Functions v2 (Python 3.9+) com Blueprints |
| Frontend | Next.js 14 + TypeScript + React 18 + TailwindCSS 3.4 |
| LLM | Grok/xAI API (`grok-4-1-fast-reasoning`) |
| Few-shot RAG | TF-IDF cosine similarity (`KBRetriever`) |
| ML (legado) | scikit-learn — setores `varejo` e `educacional` apenas |
| Storage | Azure File Share + Storage Queue / IndexedDB (client) |

## Pipeline de Classificação

```
Upload → sort por descrição → chunks de 500 → salva em taxonomy_jobs/
SubmitTaxonomyJob → enfileira {job_id} na Azure Storage Queue (taxonomy-jobs)
Queue trigger (ProcessTaxonomyJob):
  → Pre-flight billing (GET /v1/models) — 401/403 → ERROR imediato ("créditos xAI esgotados")
  → Merge KB setor (se use_sector_kb) + projeto (1x por job)
  → Two-Phase: (1) KB direct match ≥0.90 (2) LLM com enriched examples
  → Validação hierarquia (cascade: exact → shift → fuzzy → n4-reverse)
  → Consolida → "Não Identificado" nos vazios → CLASSIFIED
  → Deadline cooperativo 25min: para limpo, limpa lease e re-enfileira a continuação (resume por chunk)
Cleanup timer (1x/hora): safety net — re-enfileira PENDING órfãos e PROCESSING sem progresso > 1h (máx 3 retomadas → ERROR)

Revisão → ApproveClassifications → alimenta KB projeto → Excel → COMPLETED
Promoção para setor = ação separada (PromoteToSectorKB)
```

**Lifecycle:** `PENDING → PROCESSING → CLASSIFIED → APPROVED → COMPLETED` (+ `CANCELLED`, `ERROR`)

## Estrutura

```
├── function_app.py              # Entry point — registra blueprints
├── blueprints/                  # 8 blueprints (projects, classification, review, knowledge, models, copilot, worker, health)
├── src/                         # Lógica de negócio (ver src/CLAUDE.md)
│   ├── file_lock.py             # Thread-safe status.json ops (filelock)
│   └── queue_helpers.py         # enqueue_job() → Azure Storage Queue
├── models/
│   ├── sectors/{name}/          # sector_config.json + knowledge_base.json + kb_versions/
│   ├── projects/{id}/           # project_config.json + knowledge_base.json + kb_versions/
│   └── taxonomy_jobs/           # Fila de jobs async
├── tests/                       # pytest (470 testes)
└── frontend/                    # Next.js (ver frontend/CLAUDE.md)
```

## Variáveis de Ambiente

```bash
GROK_API_KEY=                    # Obrigatória
GROK_API_ENDPOINT=https://api.x.ai/v1
GROK_MODEL_NAME=grok-4-1-fast-reasoning
MODELS_DIR_PATH=                 # Opcional: env → /mount/models → ./models/
DIRECT_LINE_SECRET=              # Copilot Studio
# Autenticação (obrigatórias em produção)
AZURE_AD_TENANT_ID=              # Azure Entra ID tenant
AZURE_AD_CLIENT_ID=              # App registration client ID
SKIP_AUTH=true                   # Apenas dev local (bloqueado em Azure via WEBSITE_SITE_NAME)
ALLOWED_ORIGINS=http://localhost:3000  # CSV de origens permitidas para CORS
ADMIN_EMAILS=                    # CSV de emails com role admin (opcional)
ALLOWED_GROUP_ID=                # ID do grupo de segurança Azure AD (opcional — se setado, bloqueia tokens sem claim groups)
# Frontend
# NEXT_PUBLIC_API_URL, NEXT_PUBLIC_AZURE_AD_CLIENT_ID, NEXT_PUBLIC_AZURE_AD_TENANT_ID
```

## Convenções Gerais

### Python
- Módulos em `src/` com responsabilidade única
- `normalize_text()` em `preprocessing.py` é source of truth — não duplicar
- `safe_json_dumps()` para serialização JSON com dados numéricos
- `INCOMPLETE_VALUES` em `utils.py` — frozenset centralizada para detectar campos vazios/não identificados
- File locking via `src/file_lock.py` — usar `locked_status()` para ops atômicas em status.json
- Exceções de domínio em `src/exceptions.py` (`NotFoundError`, `ValidationError`, `ConflictError`, `ExternalServiceError`)
- Respostas HTTP padronizadas via `src/api_helpers.py` (`json_response`, `error_response`, `@handle_errors`, `@rate_limit`)
- Sanitização de IDs via `src/validation.py` (`safe_resource_id`) — previne path traversal em projectId, sectorName, jobId, entryId
- Autenticação JWT via `src/auth.py` (`@require_auth`, `@require_admin`) — Azure Entra ID com JWKS
- TypedDict definitions em `src/types.py` — types compartilhados entre módulos
- Logging via `logging` (nunca `print()`)
- Type checking: `pyproject.toml` com config mypy gradual (`ignore_missing_imports=true`)
- Imports: `from src.module import func`

### TypeScript
- Tipos centralizados em `frontend/src/lib/types.ts`
- Hooks em `frontend/src/hooks/` para lógica de estado
- API client em `frontend/src/lib/api.ts`
- Design tokens em `frontend/src/lib/design-tokens.ts`
- Erro de job chega à UI: `GetTaxonomyJobStatus` retorna `error` → `useTaxonomySession` expõe `processingError`/`clearProcessingError` → `ProcessingOverlay` exibe em estilo destrutivo com botão "Fechar"

### Nomenclatura
- Níveis: N1 (mais alto) → N4 (mais granular)
- Setores: slug auto-gerado lowercase (`"naval"`, `"naval-offshore"`)
- Project IDs: `"{setor}-{nome}"` (ex: `"naval-wartsila"`)
- Status jobs: `PENDING`, `PROCESSING`, `CLASSIFIED`, `APPROVED`, `COMPLETED`, `ERROR`, `CANCELLED`
- Parâmetros API: **camelCase** (`projectId`, `entryId`, `pageSize`) — não usar snake_case

### Gotchas Críticos (cross-cutting)
- **Hierarquia é lista** (não dict) — preserva N4s duplicados. Não converter para dict
- **KB dedup por `description_norm` apenas** — não reverter para `(description_norm, N4)`, causa duplicatas
- **Sector KB = referência viva** — merge automático condicionado a `use_sector_kb` (default true)
- **`use_sector_kb` condicional em 3 pontos**: worker, reclassificação, cobertura KB
- **Status CLASSIFIED (não COMPLETED)** no worker — `ApproveClassifications` define COMPLETED
- **KBRetriever criado 1x por job** — não recriar por chunk (TF-IDF indexing é o custo)
- **Hierarquia inválida falha alto** — `hierarchy_file_base64` que não parseia → `ValidationError` 400 no Create/UpdateProject; submit bloqueia projeto `own` com hierarquia vazia. Nunca engolir o parse e deixar job rodar sem taxonomia (incidente 2026-06-12, ver `docs/postmortems/`)

> Para gotchas específicos de backend ver `src/CLAUDE.md`, de frontend ver `frontend/CLAUDE.md`

## Regras de Commit

- **NUNCA** adicionar `"Co-Authored-By"` nos commits
- Mensagens em português, descritivas
- Prefixos: `Fix`, `Ajuste`, `Adicionando`, `Refactor`

## Deployment

- **Frontend**: Push para `main` → deploy automático via GitHub Actions (Azure Static Web Apps)
- **Backend**: Deploy manual — `func azure functionapp publish pg-ai-pi-spendai-api --python` (RG `CopilotPG`, subscription "Copilot subscription" — é o app que o frontend chama. O legado `az-pg-spend-analysis-ai-agent` está em outra subscription e fora de uso)

## Segurança

- **Autenticação**: Azure Entra ID (MSAL) com JWT Bearer tokens. `@require_auth` e `@require_admin` decorators
- **Sanitização**: `safe_resource_id()` em todos os IDs de recurso (path traversal prevention)
- **Rate limiting**: `@rate_limit` no SubmitTaxonomyJob (10 req/min/IP)
- **Circuit breaker**: Grok API fail-fast após 5 falhas consecutivas (60s recovery)
- **Billing fail-fast**: 401/403 da xAI → `BillingError` → job ERROR imediato com mensagem explícita (pre-flight + in-flight), sem queimar chunks em fallback
- **HTTPS Only**: habilitado no Azure Functions (HTTP → 301)
- **Security headers**: X-Frame-Options DENY, X-Content-Type-Options nosniff, Referrer-Policy, CSP com connect-src restrito
- **CORS**: dinâmico via `ALLOWED_ORIGINS` env var

## Problemas Conhecidos

- Endpoints usam `AuthLevel.ANONYMOUS` — auth real é feita pelos decorators `@require_auth`/`@require_admin`
- Cache ML in-memory — não persiste entre instâncias Functions
- Circuit breaker e rate limiter são in-memory (per-instance) — não distribuídos
- Runtime Azure é **Python 3.11** (Flex Consumption, `pg-ai-pi-spendai-api`) — o venv local deve usar 3.11 (`uv venv -p 3.11 .venv`); o 3.13 citado em `docs/migracao-consumption.md` é do app legado
- **flock em CIFS (Azure File Share) pode retornar EACCES** em contenção de lock (POSIX permite "EACCES or EAGAIN") — `src/file_lock.py` trata como "lock ocupado" e re-tenta; `filelock` é **pinado** no requirements.txt porque o build remoto do deploy resolve a versão na hora e versões diferentes tratam EACCES de forma diferente (incidente 2026-06-12, ver `docs/postmortems/`)
- **`visibilityTimeout` da queue (45min, host.json) é também o delay de retry de mensagem falhada** — worker que morre sem limpar estado deixa o job PROCESSING sem dono por até 45min (UI mostra 99%); locks de KB/config (`knowledge_base.py`, `project_manager.py`) ainda usam `FileLock` cru, sem o retry de EACCES (follow-up pendente)
