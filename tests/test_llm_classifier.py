"""Tests for src.llm_classifier — fallback chunk size and prompt correctness."""

import json

import pytest
from unittest.mock import Mock, patch

from src.llm_classifier import classify_items_with_llm
import inspect


FAKE_CONFIG = {
    "endpoint": "https://fake.api/v1",
    "api_key": "fake-key-1234567890",
    "deployment": "grok-test",
}


class TestFallbackChunkSize:
    """Quando um chunk falha, o fallback deve usar o tamanho real do chunk que falhou,
    não o tamanho do primeiro chunk."""

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier._call_openai_api")
    def test_last_chunk_fallback_uses_correct_size(self, mock_api, mock_config):
        """Se o último chunk (menor) falhar, fallback deve gerar apenas len(last_chunk) items,
        não len(first_chunk) items."""
        # 150 items => chunk_size=100 => 2 chunks: [0:100] (100 items) + [100:150] (50 items)
        descriptions = [f"item_{i}" for i in range(150)]

        def side_effect(items, *args, **kwargs):
            # First chunk (100 items) succeeds
            if len(items) == 100:
                results = [
                    {
                        "N1": "Cat",
                        "N2": "Sub",
                        "N3": "Grp",
                        "N4": "Det",
                        "confidence": 0.9,
                        "LLM_Explanation": "ok",
                    }
                    for _ in items
                ]
                return results, {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                }
            # Second chunk (50 items) fails
            raise RuntimeError("API Error on last chunk")

        mock_api.side_effect = side_effect
        results, _ = classify_items_with_llm(descriptions)

        assert len(results) == 150
        # First 100 should be successful classifications
        for i in range(100):
            assert results[i]["N1"] == "Cat", f"Item {i} should be classified"
        # Last 50 should be fallback
        for i in range(100, 150):
            assert results[i]["N1"] == "Não Identificado", (
                f"Item {i} should be fallback"
            )
            assert results[i]["confidence"] == 0.0

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier._call_openai_api")
    def test_single_chunk_fallback(self, mock_api, mock_config):
        """Se o único chunk falhar, fallback cobre todos os items."""
        descriptions = [f"item_{i}" for i in range(50)]
        mock_api.side_effect = RuntimeError("API Error")

        results, _ = classify_items_with_llm(descriptions)

        assert len(results) == 50
        for r in results:
            assert r["N1"] == "Não Identificado"
            assert r["confidence"] == 0.0


class TestPromptNoOldModelReference:
    """Prompt LLM nao deve referenciar modelo antigo grok-4-0709."""

    def test_no_grok_4_0709_in_prompt(self):
        """O source code de _call_openai_api nao deve conter referencia ao modelo antigo."""
        from src.llm_classifier import _call_openai_api_inner

        source = inspect.getsource(_call_openai_api_inner)
        assert "grok-4-0709" not in source, (
            "Prompt ainda referencia modelo antigo 'grok-4-0709'. "
            "Remover referencia hardcoded a modelo especifico."
        )

    def test_prompt_contains_disambiguation_instruction(self):
        """A instrucao de desambiguacao deve existir sem referencia a modelo."""
        from src.llm_classifier import _call_openai_api_inner

        source = inspect.getsource(_call_openai_api_inner)
        assert "desambiguar contextos" in source, (
            "Instrucao de desambiguacao deve estar presente no prompt."
        )


class TestRetryJitter:
    """Retry deve incluir jitter para evitar thundering herd."""

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.requests.post")
    @patch("src.llm_classifier.time.sleep")
    def test_retry_sleep_has_jitter(self, mock_sleep, mock_post, mock_config):
        """Sleep entre retries deve ser >= base (2^attempt)."""
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        results = classify_items_with_llm(["item teste"])

        # Deve ter chamado sleep 2 vezes (attempt 0 e 1, não no último)
        assert mock_sleep.call_count == 2
        # Cada sleep deve ser >= base (2^attempt) e <= base+1 (jitter range)
        for i, call in enumerate(mock_sleep.call_args_list):
            sleep_value = call[0][0]
            base = 2**i
            assert sleep_value >= base, f"Sleep {sleep_value} should be >= base {base}"
            assert sleep_value <= base + 1, (
                f"Sleep {sleep_value} should be <= {base + 1}"
            )


class TestMapCategoriesSemaphore:
    """map_categories_with_llm deve respeitar _LLM_SEMAPHORE."""

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier._LLM_SEMAPHORE")
    @patch("src.llm_classifier.requests.post")
    def test_map_categories_acquires_semaphore(self, mock_post, mock_sem, mock_config):
        """Cada chamada HTTP em map_categories deve adquirir/liberar o semáforo."""
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"Cat A": "Cat B"}'}}]
        }

        from src.llm_classifier import map_categories_with_llm

        map_categories_with_llm(["Cat A"], ["Cat B"])

        # Semáforo deve ter sido adquirido e liberado pelo menos 1 vez
        assert mock_sem.acquire.call_count >= 1
        assert mock_sem.release.call_count >= 1


class TestResponseMapping:
    def setup_method(self):
        from src.llm_classifier import _CIRCUIT_BREAKER

        _CIRCUIT_BREAKER.record_success()

    @staticmethod
    def classify(items, rows):
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps(rows, ensure_ascii=False)}}],
            "usage": {},
        }
        with patch(
            "src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG
        ), patch("src.llm_classifier.requests.post", return_value=response) as post:
            results, _ = classify_items_with_llm(items)
        return results, post.call_args.kwargs["json"]

    def test_reversed_rows_are_mapped_by_item_id(self):
        items = ["Parafuso M8", "Licença anual de software"]
        rows = [
            {"item_id": "item_1", "item": items[1], "N1": "Software", "N4": "Licença"},
            {"item_id": "item_0", "item": items[0], "N1": "Materiais", "N4": "Parafuso"},
        ]

        results, payload = self.classify(items, rows)

        assert [result["N4"] for result in results] == ["Parafuso", "Licença"]
        encoded_items = payload["messages"][1]["content"].split("\n", 1)[1]
        assert json.loads(encoded_items) == [
            {"item_id": "item_0", "item": items[0]},
            {"item_id": "item_1", "item": items[1]},
        ]
        assert "item_id" in payload["messages"][0]["content"]

    def test_omitted_id_falls_back_without_shifting_other_rows(self):
        items = ["Parafuso M8", "Licença anual de software"]

        results, _ = self.classify(
            items,
            [{"item_id": "item_1", "item": items[1], "N1": "Software", "N4": "Licença"}],
        )

        assert results[0]["N1"] == "Não Identificado"
        assert results[0]["confidence"] == 0.0
        assert results[1]["N4"] == "Licença"

    def test_duplicate_id_is_ambiguous_even_if_one_row_has_contradictory_text(self):
        items = ["Parafuso M8", "Licença anual de software"]

        results, _ = self.classify(
            items,
            [
                {"item_id": "item_0", "item": items[0], "N1": "A", "N4": "A"},
                {"item_id": "item_0", "item": items[1], "N1": "B", "N4": "B"},
                {"item_id": "item_1", "item": items[1], "N1": "Software", "N4": "Licença"},
            ],
        )

        assert results[0]["N1"] == "Não Identificado"
        assert results[1]["N4"] == "Licença"

    def test_id_with_contradictory_item_does_not_fall_back_to_text(self):
        items = ["Parafuso M8", "Licença anual de software"]

        results, _ = self.classify(
            items,
            [
                {"item_id": "item_0", "item": items[1], "N1": "Wrong", "N4": "Wrong"},
                {"item_id": "item_1", "item": items[1], "N1": "Software", "N4": "Licença"},
            ],
        )

        assert results[0]["N1"] == "Não Identificado"
        assert results[1]["N4"] == "Licença"

    @pytest.mark.parametrize("invalid_id", [None, 0, ["item_0"], "unknown"])
    def test_invalid_id_does_not_fall_back_to_matching_text(self, invalid_id):
        item = "Parafuso M8"

        results, _ = self.classify(
            [item], [{"item_id": invalid_id, "item": item, "N1": "Wrong", "N4": "Wrong"}]
        )

        assert results[0]["N1"] == "Não Identificado"

    def test_missing_id_in_mixed_response_is_not_matched_by_text(self):
        items = ["Parafuso M8", "Licença anual de software"]

        results, _ = self.classify(
            items,
            [
                {"item_id": "item_0", "item": items[0], "N1": "Materiais", "N4": "Parafuso"},
                {"item": items[1], "N1": "Software", "N4": "Licença"},
            ],
        )

        assert results[0]["N4"] == "Parafuso"
        assert results[1]["N1"] == "Não Identificado"

    def test_repeated_descriptions_are_disambiguated_by_id(self):
        items = ["Assinatura mensal", "Assinatura mensal"]

        results, _ = self.classify(
            items,
            [
                {"item_id": "item_1", "item": items[1], "N1": "B", "N4": "B"},
                {"item_id": "item_0", "item": items[0], "N1": "A", "N4": "A"},
            ],
        )

        assert [result["N4"] for result in results] == ["A", "B"]

    def test_legacy_exact_text_maps_uniquely_without_ids(self):
        items = ["Parafuso M8", "Licença anual de software"]

        results, _ = self.classify(
            items,
            [
                {"item": items[1], "N1": "Software", "N4": "Licença"},
                {"item": items[0], "N1": "Materiais", "N4": "Parafuso"},
            ],
        )

        assert [result["N4"] for result in results] == ["Parafuso", "Licença"]

    def test_legacy_single_unlabelled_object_remains_valid(self):
        results, _ = self.classify(
            ["Licença anual de software"],
            {"N1": "Software", "N4": "Licença"},
        )

        assert results[0]["N4"] == "Licença"

    def test_repeated_legacy_descriptions_are_ambiguous(self):
        items = ["Assinatura mensal", "Assinatura mensal"]

        results, _ = self.classify(
            items,
            [
                {"item": items[0], "N1": "A", "N4": "A"},
                {"item": items[1], "N1": "B", "N4": "B"},
            ],
        )

        assert [result["N1"] for result in results] == [
            "Não Identificado",
            "Não Identificado",
        ]

    def test_duplicate_legacy_response_rows_are_ambiguous(self):
        item = "Parafuso M8"

        results, _ = self.classify(
            [item],
            [
                {"item": item, "N1": "A", "N4": "A"},
                {"item": item, "N1": "B", "N4": "B"},
            ],
        )

        assert results[0]["N1"] == "Não Identificado"

    def test_legacy_substring_match_is_rejected(self):
        items = ["Parafuso M8", "Parafuso M8 zincado"]

        results, _ = self.classify(
            items,
            [{"item": items[1], "N1": "Fixadores", "N4": "Parafuso zincado"}],
        )

        assert results[0]["N1"] == "Não Identificado"
        assert results[1]["N4"] == "Parafuso zincado"


class TestWebSearchDoesNotBreakClassification:
    """Regressão: use_web_search=True NÃO pode injetar o parâmetro `tools`
    [{"type": "web_search"}], que a API xAI/Grok rejeita com HTTP 422
    ("unknown variant `web_search`"). Esse param fazia 100% dos itens caírem
    em fallback ('Não Identificado') silenciosamente, sem erro visível.
    Web search foi descontinuado pela xAI (Live Search → 410 Gone), então o
    comportamento correto é degradar com segurança: classificar normalmente."""

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.requests.post")
    def test_web_search_does_not_send_invalid_tools_param(self, mock_post, mock_config):
        from src.llm_classifier import _call_openai_api_inner, _CIRCUIT_BREAKER

        # garante breaker fechado (testes anteriores podem tê-lo aberto)
        _CIRCUIT_BREAKER.record_success()

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '[{"item": "Parafuso M8", "N1": "Fixadores", '
                        '"N2": "Parafusos", "N3": "", "N4": "", "confidence": 0.9}]'
                    }
                }
            ],
            "usage": {},
        }

        results, _ = _call_openai_api_inner(
            ["Parafuso M8"],
            FAKE_CONFIG,
            use_web_search=True,
        )

        # A chamada HTTP deve ter sido feita...
        assert mock_post.call_count == 1
        # ...e o payload NÃO pode conter tools:[{type:web_search}] (422 na xAI)
        payload = mock_post.call_args.kwargs["json"]
        tools = payload.get("tools", []) or []
        assert not any(
            isinstance(t, dict) and t.get("type") == "web_search" for t in tools
        ), "payload não deve enviar o tool web_search (xAI rejeita com HTTP 422)"

        # E a classificação deve funcionar normalmente (não 100% fallback)
        assert results[0]["N1"] == "Fixadores"
        assert results[0]["confidence"] != 0.0


class TestBillingFailFast:
    """HTTP 401/403 da xAI são fatais (créditos esgotados ou chave inválida):
    retry não resolve, e o fallback manual silencioso mascarava o erro do
    consultor (evidência ao vivo: xAI retorna 403 "permission-denied" com
    "used all available credits or reached its monthly spending limit").
    Devem levantar BillingError imediatamente, sem retry."""

    def setup_method(self):
        from src.llm_classifier import _CIRCUIT_BREAKER

        _CIRCUIT_BREAKER.record_success()  # garante breaker fechado entre testes

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.time.sleep")
    @patch("src.llm_classifier.requests.post")
    def test_403_raises_billing_error_without_retry(
        self, mock_post, mock_sleep, mock_config
    ):
        from src.exceptions import BillingError
        from src.llm_classifier import _call_openai_api_inner

        mock_post.return_value.status_code = 403
        mock_post.return_value.text = (
            '{"code": "permission-denied", "error": "Your team has either used '
            'all available credits or reached its monthly spending limit."}'
        )

        with pytest.raises(BillingError, match="Créditos da API xAI"):
            _call_openai_api_inner(["item"], FAKE_CONFIG)

        assert mock_post.call_count == 1, "403 é fatal — não deve ser retentado"

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.time.sleep")
    @patch("src.llm_classifier.requests.post")
    def test_401_raises_billing_error_without_retry(
        self, mock_post, mock_sleep, mock_config
    ):
        from src.exceptions import BillingError
        from src.llm_classifier import _call_openai_api_inner

        mock_post.return_value.status_code = 401
        mock_post.return_value.text = "Unauthorized"

        with pytest.raises(BillingError, match="HTTP 401"):
            _call_openai_api_inner(["item"], FAKE_CONFIG)

        assert mock_post.call_count == 1, "401 é fatal — não deve ser retentado"

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.time.sleep")
    @patch("src.llm_classifier.requests.post")
    def test_billing_error_propagates_through_classify_items_with_llm(
        self, mock_post, mock_sleep, mock_config
    ):
        """O ThreadPoolExecutor de classify_items_with_llm NÃO pode engolir
        BillingError no fallback manual — deve propagar para o worker."""
        from src.exceptions import BillingError

        mock_post.return_value.status_code = 403
        mock_post.return_value.text = "permission-denied"

        with pytest.raises(BillingError):
            classify_items_with_llm(["item a", "item b"])

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.time.sleep")
    @patch("src.llm_classifier.requests.post")
    def test_500_still_returns_fallback_not_billing_error(
        self, mock_post, mock_sleep, mock_config
    ):
        """Comportamento preservado: 5xx continua com retry + fallback manual
        (transitório), sem BillingError."""
        from src.llm_classifier import _call_openai_api_inner

        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        results, usage = _call_openai_api_inner(["item"], FAKE_CONFIG)

        assert len(results) == 1
        assert results[0]["N1"] == "Não Identificado"
        assert mock_post.call_count == 3  # 1 + LLM_MAX_RETRIES


class TestCheckLLMHealth:
    """Pre-flight check barato (GET /models, zero tokens) que detecta créditos
    esgotados ANTES de processar chunks — comprovado ao vivo que retorna 403."""

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.requests.get")
    def test_health_403_raises_billing_error(self, mock_get, mock_config):
        from src.exceptions import BillingError
        from src.llm_classifier import check_llm_health

        mock_get.return_value.status_code = 403

        with pytest.raises(BillingError, match="Créditos da API xAI"):
            check_llm_health()

        called_url = mock_get.call_args[0][0]
        assert called_url.endswith("/models"), "deve usar GET /models (grátis)"

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.requests.get")
    def test_health_200_does_not_raise(self, mock_get, mock_config):
        from src.llm_classifier import check_llm_health

        mock_get.return_value.status_code = 200
        check_llm_health()  # não deve levantar

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier.requests.get")
    def test_health_network_error_does_not_block(self, mock_get, mock_config):
        """Falha de rede no pre-flight NÃO bloqueia o job (pipeline tem
        retry/fallback próprios)."""
        from src.llm_classifier import check_llm_health

        mock_get.side_effect = ConnectionError("network down")
        check_llm_health()  # não deve levantar

    @patch(
        "src.llm_classifier.get_azure_openai_config",
        return_value={
            "endpoint": "https://fake.api/v1",
            "api_key": "",
            "deployment": "x",
        },
    )
    @patch("src.llm_classifier.requests.get")
    def test_health_without_key_skips(self, mock_get, mock_config):
        """Sem chave configurada, classify já degrada com fallback — health não chama API."""
        from src.llm_classifier import check_llm_health

        check_llm_health()
        mock_get.assert_not_called()


class TestTokenUsageReturn:
    """(f) classify_items_with_llm deve retornar (results, total_usage) com os
    tokens agregados de todos os batches — antes o total era logado e descartado."""

    @patch("src.llm_classifier.get_azure_openai_config", return_value=FAKE_CONFIG)
    @patch("src.llm_classifier._call_openai_api")
    def test_returns_results_and_aggregated_usage(self, mock_api, mock_config):
        descriptions = [f"item_{i}" for i in range(150)]  # 2 batches (100 + 50)

        def side_effect(items, *args, **kwargs):
            results = [
                {
                    "N1": "Cat",
                    "N2": "Sub",
                    "N3": "Grp",
                    "N4": "Det",
                    "confidence": 0.9,
                    "LLM_Explanation": "ok",
                }
                for _ in items
            ]
            usage = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 10,
                "total_tokens": 160,
            }
            return results, usage

        mock_api.side_effect = side_effect
        results, total_usage = classify_items_with_llm(descriptions)

        assert len(results) == 150
        assert total_usage["prompt_tokens"] == 200
        assert total_usage["completion_tokens"] == 100
        assert total_usage["reasoning_tokens"] == 20
        assert total_usage["total_tokens"] == 320

    @patch(
        "src.llm_classifier.get_azure_openai_config",
        return_value={"endpoint": "", "api_key": "", "deployment": ""},
    )
    def test_unconfigured_key_returns_results_and_none_usage(self, mock_config):
        results, usage = classify_items_with_llm(["a"])
        assert len(results) == 1
        assert usage is None
