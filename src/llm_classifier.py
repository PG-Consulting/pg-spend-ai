"""
LLM Classifier for Standard/UNSPSC Mode.
Uses Azure OpenAI to classify items when the ML model is uncertain or when "Padrão" mode is selected.
"""

import os
import json
import logging
import random
import time
import requests
from enum import Enum
from typing import List, Dict, Optional, Tuple, Union

from src.types import ClassificationResultDict, HierarchyEntryDict, KBEntryDict
from src.exceptions import BillingError
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 2  # Backoff exponencial em chamadas à API
LLM_MAX_CONCURRENT_CALLS = 15  # Max chamadas LLM simultâneas (global, cross-chunk)
_LLM_SEMAPHORE = threading.Semaphore(LLM_MAX_CONCURRENT_CALLS)

# Status codes fatais de billing/auth: retry não resolve (créditos esgotados ou
# chave inválida). Evidência ao vivo: a xAI retorna 403 code "permission-denied"
# com "used all available credits or reached its monthly spending limit".
BILLING_FATAL_STATUS_CODES = (401, 403)


def _billing_error_message(status_code) -> str:
    return (
        f"Créditos da API xAI esgotados ou chave inválida (HTTP {status_code}). "
        "Recarregue créditos no console.x.ai e re-submeta o job."
    )


def check_llm_health() -> None:
    """Pre-flight barato: GET {endpoint}/models (não consome tokens).

    Levanta BillingError quando a API responde 401/403 (créditos esgotados ou
    chave inválida — comprovado ao vivo que a xAI retorna 403 nesse caso).
    Outras falhas (rede, 5xx) NÃO bloqueiam o job: o pipeline tem retry e
    fallback próprios para erros transitórios.
    """
    config = get_azure_openai_config()
    if not config["api_key"] or config["api_key"] == "SUA-CHAVE-AQUI":
        # Sem chave: classify_items_with_llm já degrada com fallback explícito.
        return
    endpoint = f"{config['endpoint'].rstrip('/')}/models"
    try:
        response = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=15,
        )
    except Exception as e:
        logger.warning(f"check_llm_health: falha de rede (não bloqueante): {e}")
        return
    if response.status_code in BILLING_FATAL_STATUS_CODES:
        raise BillingError(_billing_error_message(response.status_code))


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple circuit breaker for external API calls.

    CLOSED: normal operation.
    OPEN: fail fast after failure_threshold consecutive failures.
    HALF_OPEN: one test request after recovery_timeout seconds.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitState.CLOSED

    def can_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN

    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# UNSPSC Segment/Family definitions for prompt context
# We use a simplified subset to guide the model, or rely on its internal knowledge (GPT-4 handles UNSPSC well)
UNSPSC_CONTEXT = """
Classifique os itens abaixo na taxonomia UNSPSC (United Nations Standard Products and Services Code).
Retorne o Segmento (N1) e a Família (N2) mais adequados.
Exemplos:
- "Caneta Esferográfica" -> N1: "Material de Escritório", N2: "Instrumentos de Escrita"
- "Licença Microsoft Office" -> N1: "Software", N2: "Software de Negócios"
- "Serviço de Limpeza Predial" -> N1: "Serviços Prediais", N2: "Limpeza e Manutenção"
- "Consultoria Financeira" -> N1: "Serviços Financeiros", N2: "Consultoria"
"""


def _format_hierarchy_compact(
    custom_hierarchy: Union[Dict[str, HierarchyEntryDict], List[HierarchyEntryDict]],
) -> str:
    """
    Formata hierarquia customizada em formato árvore com labels explícitos [N1]/[N2]/[N3]/[N4].
    Labels eliminam ambiguidade de nível (reduz deslocamento de ~14% para ~5%).
    Aceita lista de dicts ou dict keyed por N4 (backward compat).

    Output:
        [N1] Operação e Manutenção
          [N2] Materiais e Serviços OEM
            [N3] OEM - ABB
              [N4] Materiais OEM
              [N4] Serviços OEM
    """
    # Aceitar lista ou dict (backward compat)
    if isinstance(custom_hierarchy, dict):
        entries = custom_hierarchy.values()
    else:
        entries = custom_hierarchy

    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for h in entries:
        n1 = h.get("N1", "") or "Outros"
        n2 = h.get("N2", "") or "Outros"
        n3 = h.get("N3", "") or "Outros"
        n4 = h.get("N4", "")
        if n4 and n4 not in tree[n1][n2][n3]:  # dedup na mesma N3
            tree[n1][n2][n3].append(n4)

    lines = []
    for n1 in sorted(tree.keys()):
        lines.append(f"[N1] {n1}")
        for n2 in sorted(tree[n1].keys()):
            lines.append(f"  [N2] {n2}")
            for n3 in sorted(tree[n1][n2].keys()):
                lines.append(f"    [N3] {n3}")
                for n4 in sorted(tree[n1][n2][n3]):
                    lines.append(f"      [N4] {n4}")
    return "\n".join(lines)


def get_azure_openai_config() -> Dict[str, str]:
    """Retrieves Azure OpenAI config from environment variables."""
    return {
        "endpoint": os.getenv("GROK_API_ENDPOINT", "https://api.x.ai/v1"),
        "api_key": os.getenv("GROK_API_KEY", ""),
        "deployment": os.getenv("GROK_MODEL_NAME", "grok-4-1-fast-reasoning"),
    }


def classify_items_with_llm(
    descriptions: List[str],
    sector: str = "Padrão",
    client_context: str = "",
    custom_hierarchy: Optional[
        Union[Dict[str, HierarchyEntryDict], List[HierarchyEntryDict]]
    ] = None,
    few_shot_examples: Optional[List[KBEntryDict]] = None,
    user_instruction: Optional[str] = None,
    use_web_search: bool = False,
) -> Tuple[List[ClassificationResultDict], Optional[Dict[str, int]]]:
    """
    Classifies a list of item descriptions using Azure OpenAI.

    Args:
        descriptions: List of text descriptions to classify.
        sector: The industry sector to provide context (e.g. 'Varejo', 'Educacional').
        client_context: Additional context about the client/project (e.g. 'Dengo - Chocolate').
        custom_hierarchy: Optional dictionary containing the allowed categories (N1-N4).
        few_shot_examples: Optional list of confirmed examples from the consultant KB.
            Each entry is a dict with keys: description, N1, N2, N3, N4.
        user_instruction: Optional freeform instruction from the consultant (highest priority).

    Returns:
        Tuple (results, total_usage):
        - results: list of dicts with keys N1, N2, N3, N4, confidence, explanation
        - total_usage: dict agregado {prompt_tokens, completion_tokens,
          reasoning_tokens, total_tokens} ou None quando não houve chamada à API
    """
    config = get_azure_openai_config()

    # Validation
    if (
        not config["endpoint"]
        or not config["api_key"]
        or config["api_key"] == "SUA-CHAVE-AQUI"
    ):
        logging.warning(
            "Azure OpenAI keys not configured. Skipping LLM classification."
        )
        return [_create_empty_result() for _ in descriptions], None

    # Prepare batch prompt (process in chunks if needed, here we assume small batches or separate calls)
    # For a list, we might want to process all at once if small, or loop.
    # To keep response structured, we'll ask for JSON.

    results = [None] * len(descriptions)

    # Process in larger batches (100 items) and use parallel threads
    chunk_size = (
        100  # Batch grande melhora consistência (itens similares no mesmo contexto)
    )
    chunks = []
    for i in range(0, len(descriptions), chunk_size):
        chunks.append((i, descriptions[i : i + chunk_size]))

    logging.info(
        f"Starting aggressive parallel LLM classification ({len(chunks)} chunks)..."
    )

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }

    with ThreadPoolExecutor(max_workers=LLM_MAX_CONCURRENT_CALLS) as executor:
        future_to_chunk = {
            executor.submit(
                _call_openai_api,
                chunk_items,
                config,
                sector,
                client_context,
                custom_hierarchy,
                few_shot_examples,
                user_instruction,
                use_web_search,
            ): chunk_start
            for chunk_start, chunk_items in chunks
        }

        for future in as_completed(future_to_chunk):
            chunk_start = future_to_chunk[future]
            try:
                chunk_results, chunk_usage = future.result()
                # Aggregate token usage
                if chunk_usage:
                    for k in total_usage:
                        total_usage[k] += chunk_usage.get(k, 0)
                # Place results back in correct order
                for offset, res in enumerate(chunk_results):
                    if chunk_start + offset < len(results):
                        results[chunk_start + offset] = res
            except BillingError:
                # Fatal (créditos esgotados): NÃO converter em fallback manual —
                # propaga para o worker marcar o job ERROR com mensagem explícita.
                raise
            except Exception as e:
                logging.error(f"Chunk starting at {chunk_start} failed: {e}")
                failed_chunk_items = next(
                    (items for start, items in chunks if start == chunk_start), []
                )
                for offset in range(len(failed_chunk_items)):
                    idx = chunk_start + offset
                    if idx < len(results):
                        results[idx] = _create_manual_fallback(
                            "Erro no processamento paralelo"
                        )

    # Log aggregated token usage
    if total_usage["total_tokens"] > 0:
        logger.info(
            "TOKEN USAGE TOTAL: input=%d, output=%d, reasoning=%d, total=%d, items=%d, llm_calls=%d",
            total_usage["prompt_tokens"],
            total_usage["completion_tokens"],
            total_usage["reasoning_tokens"],
            total_usage["total_tokens"],
            len(descriptions),
            len(chunks),
        )

    final_results = [
        r
        if r is not None
        else _create_manual_fallback(
            "Falha no mapeamento", "Falha Crítica no Processamento"
        )
        for r in results
    ]
    return final_results, total_usage


def _create_empty_result() -> ClassificationResultDict:
    return {
        "N1": "Não Identificado",
        "N2": "Não Identificado",
        "N3": "Não Identificado",
        "N4": "Não Identificado",
        "LLM_Explanation": "LLM não configurado",
        "confidence": 0.0,
    }


def _call_openai_api(
    items: List[str],
    config: Dict[str, str],
    sector: str = "Padrão",
    client_context: str = "",
    custom_hierarchy: Optional[
        Union[Dict[str, HierarchyEntryDict], List[HierarchyEntryDict]]
    ] = None,
    few_shot_examples: Optional[List[KBEntryDict]] = None,
    user_instruction: Optional[str] = None,
    use_web_search: bool = False,
) -> List[ClassificationResultDict]:
    """Helper to call the API for a chunk of items."""
    _LLM_SEMAPHORE.acquire()
    try:
        return _call_openai_api_inner(
            items,
            config,
            sector,
            client_context,
            custom_hierarchy,
            few_shot_examples,
            user_instruction,
            use_web_search,
        )
    finally:
        _LLM_SEMAPHORE.release()


def _call_openai_api_inner(
    items: List[str],
    config: Dict[str, str],
    sector: str = "Padrão",
    client_context: str = "",
    custom_hierarchy: Optional[
        Union[Dict[str, HierarchyEntryDict], List[HierarchyEntryDict]]
    ] = None,
    few_shot_examples: Optional[List[KBEntryDict]] = None,
    user_instruction: Optional[str] = None,
    use_web_search: bool = False,
):
    """Actual API call logic, called under semaphore."""

    if not _CIRCUIT_BREAKER.can_attempt():
        logger.warning(
            "Circuit breaker OPEN — skipping Grok API call, returning fallback"
        )
        return [_create_manual_fallback(item) for item in items], None

    # Construct the system message
    client_info = f"para o cliente: {client_context}" if client_context else ""

    # Construct the system message using the User's specific "Consultant" persona
    client_name = (
        client_context if client_context else "ACNE"
    )  # Default placeholder if not provided

    # System message SEPARADO para hierarquia customizada vs padrão
    if custom_hierarchy:
        compact_tree = _format_hierarchy_compact(custom_hierarchy)
        system_message = (
            f"Você é um especialista em categorização de spend corporativo. "
            f"Sua tarefa é categorizar cada item segundo o Contexto/Cliente: '{client_name}'. "
            "ATENÇÃO: Se o contexto acima contiver 'Regras' ou instruções específicas, "
            "aplique-as com prioridade máxima sobre seu conhecimento geral.\n\n"
            "FORMATO DE SAÍDA: Retorne APENAS JSON array (sem markdown).\n"
            'Exemplo: [{"item_id": "item_0", "item": "...", "N1": "...", "N2": "...", "N3": "...", "N4": "...", "confidence": 0.9}]\n\n'
            f"ÁRVORE DE CATEGORIAS DO CLIENTE (cada linha prefixada com [N1], [N2], [N3] ou [N4]):\n"
            f"{compact_tree}\n\n"
            "RESTRIÇÕES OBRIGATÓRIAS:\n"
            "1. Classifique APENAS usando as categorias da árvore acima.\n"
            "2. Cada linha tem um prefixo [N1], [N2], [N3] ou [N4] indicando o nível.\n"
            "   Para cada item, retorne o caminho completo: o [N1] que contém o [N2], que contém o [N3], que contém o [N4].\n"
            "3. Copie EXATAMENTE os nomes como aparecem na árvore.\n"
            "4. Quando o MESMO N4 aparece sob diferentes N3, analise a descrição "
            "para determinar o N3 correto.\n"
            "5. Se nenhuma categoria se encaixa, use 'Não Identificado' em todos os níveis.\n"
            "6. NUNCA invente categorias fora da árvore."
        )
    else:
        system_message = (
            f"Você é um especialista em categorização de spend corporativo, com experiência em estruturas de classificação como UNSPSC e em modelos customizados de categorias de Compras. "
            f"Sua tarefa é categorizar automaticamente cada item da base de gastos segundo o Contexto/Cliente: '{client_name}'. "
            "ATENÇÃO: Se o contexto acima contiver 'Regras' ou instruções específicas, aplique-as com prioridade máxima sobre seu conhecimento geral.\n"
            "utilizando uma árvore de categorias que está disponibilizada abaixo. "
            "Caso fique em dúvida na classificação ou não conseguiu identificar na árvore, coloque 'Não Identificado' em todos os níveis (N1-N4). "
            "Os dados serão processados para Excel, então avalie item por linha.\n\n"
            "REGRAS DE OURO PARA RACIOCÍNIO:\n"
            "Preste extrema atenção na descrição do item, pois palavras-chave mudam radicalmente a categoria.\n"
            "Exemplo Clássico do 'Tubo':\n"
            "- Se 'Tubo' contiver 'PVC' -> MRO > Materiais de Construção > Produtos Sanitários e Hidráulicos\n"
            "- Se 'Tubo' contiver 'AÇO' ou 'CARBONO' -> Industrial > Materiais Industriais > Tubulações Industriais\n\n"
            "Analise cada palavra antes de decidir para desambiguar contextos.\n"
            "IMPORTANTE: Retorne a resposta APENAS no formato JSON abaixo (array de objetos), sem markdown. "
            "Exemplo de Saída:\n"
            '[{"item_id": "item_0", "item": "Tubo PVC 10mm", "N1": "MRO", "N2": "Materiais de Construção", "N3": "Produtos Sanitários", "N4": "Tubos", "confidence": 0.95}, ...]'
        )

    # Add few-shot examples if provided
    if few_shot_examples:
        examples_text = "\n".join(
            [
                f'- "{ex["description"]}" → N1: {ex["N1"]}, N2: {ex["N2"]}, N3: {ex["N3"]}, N4: {ex["N4"]}'
                for ex in few_shot_examples
                if ex.get("N1") and ex.get("N4")
            ]
        )
        if examples_text:
            system_message += (
                f"\n\nEXEMPLOS CONFIRMADOS PELO CONSULTOR (use como referência prioritária):\n"
                f"{examples_text}\n"
                "Estes exemplos foram validados manualmente pelo consultor. "
                "Para itens similares, priorize estas classificações."
            )

    # Add user instruction if provided
    if user_instruction:
        system_message += (
            f"\n\nINSTRUÇÃO ESPECÍFICA DO CONSULTOR (prioridade máxima, sobrepõe qualquer outro critério):\n"
            f"{user_instruction}\n"
        )

    system_message += (
        "\n\nIDENTIFICAÇÃO DOS ITENS: cada entrada tem um `item_id` único. "
        "Retorne exatamente um objeto por entrada, copiando `item_id` e `item` "
        "sem alteração. Não omita, duplique nem invente IDs."
    )

    # Web search foi descontinuado pela xAI: a Live Search retorna HTTP 410
    # ("deprecated, switch to Agent Tools API") e o antigo
    # tools:[{"type":"web_search"}] é rejeitado com HTTP 422. Quando ligado,
    # isso fazia 100% dos itens caírem em fallback silenciosamente. Degradamos
    # com segurança: classificamos normalmente, sem prometer internet ao modelo.
    if use_web_search:
        logger.debug(
            "use_web_search solicitado, mas web search foi descontinuado pela "
            "xAI; classificando sem busca na internet (degradação segura)"
        )

    user_items = [
        {"item_id": f"item_{idx}", "item": item}
        for idx, item in enumerate(items)
    ]
    user_content = (
        "Classifique todos os itens. Preserve exatamente os campos `item_id` e `item`:\n"
        + json.dumps(user_items, ensure_ascii=False)
    )

    payload = {
        "model": config["deployment"],
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
    }

    # NÃO injetar tools:[{"type":"web_search"}] — a xAI rejeita com HTTP 422
    # ("unknown variant `web_search`"). Web search foi descontinuado; ver acima.

    # xAI (Grok) API endpoint
    endpoint = f"{config['endpoint'].rstrip('/')}/chat/completions"

    # DEBUG: Validating configuration
    if not config["api_key"] or len(config["api_key"]) < 10:
        logging.warning("CRITICAL: Grok API Key missing or too short!")

    max_retries = LLM_MAX_RETRIES
    response = None
    for attempt in range(max_retries + 1):
        try:
            logging.info(
                f"Sending request to Grok with input size {len(items)} (Attempt {attempt + 1})"
            )
            response = requests.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config['api_key']}",
                },
                json=payload,
                timeout=90,
            )

            if response.status_code == 200:
                break

            # Rate limit: respeitar Retry-After header
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logging.warning(
                    f"Rate limited (429). Waiting {retry_after}s before retry..."
                )
                if attempt < max_retries:
                    time.sleep(retry_after)
                    continue

            # Billing/auth fatal: 401/403 não são transitórios (créditos
            # esgotados ou chave inválida) — retry não resolve. Fail fast em
            # vez de cair em fallback manual silencioso.
            if response.status_code in BILLING_FATAL_STATUS_CODES:
                logging.error(
                    f"Grok API billing/auth error {response.status_code}: {response.text}"
                )
                _CIRCUIT_BREAKER.record_failure()
                raise BillingError(_billing_error_message(response.status_code))

            logging.error(
                f"Grok API Error {response.status_code} (Attempt {attempt + 1}): {response.text}"
            )
            if attempt < max_retries:
                time.sleep(2**attempt + random.uniform(0, 1))
        except BillingError:
            raise  # fatal — propaga para o worker marcar o job ERROR
        except Exception as e:
            logging.error(
                f"Exception calling Azure OpenAI (Attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                time.sleep(2**attempt + random.uniform(0, 1))
            else:
                _CIRCUIT_BREAKER.record_failure()
                return [_create_manual_fallback(item) for item in items], None

    _empty_usage = None
    try:
        if response is None or response.status_code != 200:
            _CIRCUIT_BREAKER.record_failure()
            code = response.status_code if response else "N/A"
            return [
                _create_manual_fallback(item, f"Erro {code} após retentativas")
                for item in items
            ], _empty_usage

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Extract token usage to return alongside results
        usage = data.get("usage", {})
        details = usage.get("completion_tokens_details") or {}
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": details.get("reasoning_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "items": len(items),
        }

        # Clean Markdown code blocks if present
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()

        # Parse JSON output
        parsed = json.loads(content)

        # Check for error responses from Azure OpenAI
        if isinstance(parsed, dict) and "error" in parsed:
            error_msg = parsed.get("error", "Unknown error")
            logging.error(f"Azure OpenAI returned error: {error_msg}")
            return [
                _create_manual_fallback(item, f"API Error: {error_msg}")
                for item in items
            ], token_usage

        # Handle different response formats:
        # 1. Direct dict with N1-N4 keys (single item)
        # 2. Dict with a list value (batch with wrapper key)
        # 3. Direct list (batch without wrapper)

        if isinstance(parsed, dict):
            # Check if it's a single-item response with N1-N4 keys
            if "N1" in parsed or "N2" in parsed:
                # Single item response - wrap in list
                parsed = [parsed]
            else:
                # Try to find a list value in the dict (e.g., {"items": [...]})
                for key, val in parsed.items():
                    if isinstance(val, list):
                        parsed = val
                        break
                else:
                    # No list found and no N1-N4 keys - unexpected format
                    logging.warning(
                        f"LLM returned unexpected dict format: {list(parsed.keys())}"
                    )
                    return [
                        _create_manual_fallback(item, "Formato inesperado")
                        for item in items
                    ], token_usage

        if not isinstance(parsed, list):
            logging.warning(
                "LLM returned unexpected format (not list after processing)"
            )
            return [
                _create_manual_fallback(item, "Formato inesperado") for item in items
            ], token_usage

        # Map by the per-call ID. Accept older responses only when an exact
        # description uniquely identifies one input and one response row.
        response_rows = [row for row in parsed if isinstance(row, dict)]
        has_ids = any("item_id" in row for row in response_rows)
        matches = [None] * len(items)

        if has_ids:
            id_counts = Counter(
                row["item_id"]
                for row in response_rows
                if isinstance(row.get("item_id"), str)
            )
            for idx, item_text in enumerate(items):
                item_id = f"item_{idx}"
                if id_counts[item_id] != 1:
                    continue
                match = next(
                    row for row in response_rows if row.get("item_id") == item_id
                )
                if "item" in match and match["item"] != item_text:
                    continue
                matches[idx] = match
        else:
            input_counts = Counter(items)
            rows_by_item = defaultdict(list)
            unlabelled_rows = []
            for row in response_rows:
                if isinstance(row.get("item"), str):
                    rows_by_item[row["item"]].append(row)
                elif "item" not in row:
                    unlabelled_rows.append(row)

            for idx, item_text in enumerate(items):
                item_rows = rows_by_item[item_text]
                if input_counts[item_text] == 1 and len(item_rows) == 1:
                    matches[idx] = item_rows[0]
                elif (
                    len(items) == 1
                    and len(parsed) == 1
                    and len(unlabelled_rows) == 1
                ):
                    matches[idx] = unlabelled_rows[0]

        formatted_results = []
        for item_text, match in zip(items, matches):
            if match and (match.get("N1") or match.get("N2")):
                formatted_results.append(
                    {
                        "N1": match.get("N1", ""),
                        "N2": match.get("N2", ""),
                        "N3": match.get("N3", ""),
                        "N4": match.get("N4", ""),
                        "LLM_Explanation": "Classificado via Azure OpenAI (UNSPSC)",
                        "confidence": match.get("confidence", 0.8),
                    }
                )
            else:
                formatted_results.append(
                    _create_manual_fallback(item_text, "Item não retornado pelo LLM")
                )

        _CIRCUIT_BREAKER.record_success()
        return formatted_results, token_usage

    except Exception as e:
        logging.error(f"Exception calling Azure OpenAI: {e}")
        return [_create_manual_fallback(item) for item in items], None


def _create_manual_fallback(
    item_text: str, reason: str = "Erro na API"
) -> ClassificationResultDict:
    return {
        "N1": "Não Identificado",
        "N2": "Não Identificado",
        "N3": "Não Identificado",
        "N4": "Não Identificado",
        "LLM_Explanation": reason,
        "confidence": 0.0,
    }


def _post_with_semaphore(endpoint, headers, payload, timeout=90):
    """HTTP POST respeitando o semáforo global de rate limiting."""
    _LLM_SEMAPHORE.acquire()
    try:
        return requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    finally:
        _LLM_SEMAPHORE.release()


def map_categories_with_llm(
    source_categories: List[str], target_categories: List[str]
) -> Dict[str, str]:
    """
    Map a list of source categories to the closest target categories
    using LLM semantic matching. Using aggressive parallelism for Azure timeouts.
    """
    config = get_azure_openai_config()
    if (
        not config["api_key"]
        or len(source_categories) == 0
        or len(target_categories) == 0
    ):
        return {}

    target_list_str = "\n".join([f"- {t}" for t in target_categories])

    system_message = (
        "Você é um especialista em taxonomias de compras. "
        "Mapeie termos de origem para a taxonomia de destino do cliente.\n"
        "TAXONOMIA DE DESTINO:\n"
        f"{target_list_str}\n\n"
        "RESPONDA APENAS COM JSON:\n"
        "{\n"
        '  "Termo Origem": "Termo Destino"\n'
        "}"
    )

    mappings = {}
    chunk_size = 20  # 10x fewer API calls for semantic mapping

    # Process in parallel like the main classification
    chunk_items = []
    for i in range(0, len(source_categories), chunk_size):
        chunk_items.append(source_categories[i : i + chunk_size])

    logging.info(f"Starting parallel semantic mapping ({len(chunk_items)} chunks)...")

    endpoint = f"{config['endpoint'].rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }

    with ThreadPoolExecutor(max_workers=LLM_MAX_CONCURRENT_CALLS) as executor:
        futures = []
        for chunk in chunk_items:
            payload = {
                "model": config["deployment"],
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Mapeie: {', '.join(chunk)}"},
                ],
                "temperature": 0.0,
            }
            futures.append(
                executor.submit(
                    _post_with_semaphore,
                    endpoint,
                    headers,
                    payload,
                )
            )

        for future in futures:
            try:
                response = future.result()
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    if "```" in content:
                        content = (
                            content.replace("```json", "").replace("```", "").strip()
                        )
                    mappings.update(json.loads(content))
            except Exception as e:
                logging.error(f"Parallel mapping chunk failed: {e}")

    return mappings
