# Análise de Deploy - Chatterbox TTS Service

## Contexto

Análise de opções de deploy para um serviço de TTS baseado no Chatterbox, com foco em **custo-benefício para baixa carga**.

### Características do serviço
- **RAM necessária:** ~10 GB (ambos modelos carregados)
- **Latência Inglês (Turbo):** ~1-2s (70 tokens/s)
- **Latência Português (Multilingual):** ~5-8s (15 tokens/s)

---

## Cenário de baixa carga

**Premissas:**
- ~200 requests/dia (~6.000/mês)
- ~5 seg de geração por request (GPU T4)
- ~30.000 segundos de compute/mês (~8 horas)
- Poucos usuários, uso esporádico durante o dia
- Sem uso durante horário de sono

---

## Tabela Comparativa

| Serviço | Custo/mês | Cold Start | UX | Vantagens | Desvantagens |
|---------|-----------|------------|-----|-----------|--------------|
| **[RunPod Serverless](https://www.runpod.io/pricing)** | **~$3-5** | ~500ms (FlashBoot) | ⭐⭐⭐⭐⭐ | Cold start muito baixo, preço ótimo | Menos features que Modal |
| **[Modal](https://modal.com/pricing)** | **~$5** (sem warm) | 2-4s | ⭐⭐⭐⭐ | SDK Python excelente, DX top | Cold start maior |
| **[Replicate](https://replicate.com/pricing)** | **~$7** | Variável | ⭐⭐⭐⭐ | Zero config, eles gerenciam tudo | Menos controle |
| **[Fly.io](https://fly.io)** (CPU) | **~$15-20** | 30-60s | ⭐⭐ | Simples, escala a zero | Sem GPU, lento |
| **Google Cloud Run** (CPU) | **~$10-20** | 30-60s | ⭐⭐ | Integração GCP | Complexo, sem GPU barato |

---

## Detalhes de Pricing

### RunPod Serverless
- **T4 GPU:** ~$0.40/hr ($0.000111/sec)
- **Flex Workers:** escalam a zero, paga só quando usa
- **FlashBoot:** cold start de ~500ms
- Sem taxas de ingress/egress

### Modal
- **T4 GPU:** ~$0.59/hr ($0.000164/sec)
- Cold start típico: 2-4 segundos
- SDK Python nativo, excelente DX
- Opção `keep_warm` para manter instâncias

### Replicate
- **T4 GPU:** ~$0.81/hr ($0.000225/sec)
- Billing por segundo, mínimo 1 segundo
- Zero configuração de infra
- Eles gerenciam scaling automaticamente

---

## Custo com instância "warm" (elimina cold start)

Se quiser UX instantânea (0 cold start), precisa manter 1 worker ligado:

| Serviço | Custo warm 12h/dia | Custo warm 24/7 |
|---------|-------------------|-----------------|
| **RunPod** (T4) | ~$145/mês | ~$290/mês |
| **Modal** (T4) | ~$210/mês | ~$420/mês |
| **Replicate** | N/A (não oferece) | N/A |

---

## Recomendações

### Para começar (baixo custo, aceita cold start)
**→ RunPod Serverless** - $3-5/mês
- Cold start de 500ms é aceitável para maioria dos casos
- Melhor custo-benefício

### Se cold start de 500ms ainda incomodar
**→ Modal com `keep_warm=1`** - ~$150/mês
- Mantém 1 instância durante horário comercial
- Zero cold start durante o dia

### Se quiser zero trabalho
**→ Replicate** - $7/mês
- Você só sobe o modelo e pronto
- Eles cuidam de toda a infra

---

## Fontes

- [Modal Pricing](https://modal.com/pricing)
- [Replicate Pricing](https://replicate.com/pricing)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [RunPod Serverless Docs](https://docs.runpod.io/serverless/pricing)
- [Modal T4 GPU Article](https://modal.com/blog/nvidia-t4-price-article)

---

*Análise gerada em Janeiro/2026*
