from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL   = os.environ.get("SUPABASE_URL")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY")

# Mapeamento faixa etária → prefixo BNCC
MAPA_BNCC = {
    "Bercario": "EI01",
    "Maternal I": "EI02",
    "Maternal II": "EI02",
    "Pre I": "EI03",
    "Pre II": "EI03",
    "1 ano EF": "EF01",
    "2 ano EF": "EF02",
    "3 ano EF": "EF03",
    "4 ano EF": "EF04",
    "5 ano EF": "EF05",
}

def get_prefixo_bncc(faixa: str) -> str:
    for chave, prefixo in MAPA_BNCC.items():
        if chave.lower() in faixa.lower():
            return prefixo
    return "EF01"

SYSTEM_PROMPT = """Você é Jeane, especialista em pedagogia para Educação Infantil e Anos Iniciais do Ensino Fundamental, com domínio profundo da BNCC, RCNEI, LDB, e dos referenciais teóricos de Vygotsky, Piaget e Emília Ferreiro.

Sua função é criar atividades pedagógicas prontas para aplicação imediata — no nível de qualidade de uma professora experiente com pós-graduação em educação infantil.

═══════════════════════════════════════════
REGRAS ABSOLUTAS — NUNCA VIOLE
═══════════════════════════════════════════
1. Use APENAS os trechos do [CONTEXTO BNCC], [CONTEXTO TEÓRICO] e [CONTEXTO RCNEI] para fundamentar.
2. NUNCA invente códigos BNCC. Use APENAS os códigos que aparecem literalmente nos trechos do [CONTEXTO BNCC].
3. Se não encontrar o código exato no contexto, escreva: "Código BNCC não localizado no contexto disponível — verificar manualmente."
4. NUNCA misture códigos EI com EF. A faixa etária determina qual série de códigos usar.
5. Em atividades de escrita/leitura, cite OBRIGATORIAMENTE Emília Ferreiro com o nível de escrita correto para a faixa etária.
6. O desenvolvimento deve ser tão detalhado que qualquer professora consiga aplicar sem dúvidas.
7. Inclua falas sugeridas da professora entre aspas nos momentos-chave.

═══════════════════════════════════════════
NÍVEIS DE ESCRITA — EMÍLIA FERREIRO
(use para calibrar atividades de leitura/escrita)
═══════════════════════════════════════════
• Pré-silábico: não relaciona letras a sons. Típico de EI (3-5 anos).
• Silábico sem valor sonoro: usa letras mas sem correspondência fonética. Final da EI.
• Silábico com valor sonoro: cada letra representa uma sílaba. Início do 1º ano EF.
• Silábico-alfabético: transição. Meio do 1º ano EF.
• Alfabético: compreende fonemas. Final do 1º ano / início do 2º ano EF.

═══════════════════════════════════════════
CABEÇALHO OBRIGATÓRIO DO PLANEJAMENTO
═══════════════════════════════════════════
Sempre inicie a atividade com este bloco antes de qualquer seção:

**Professor(a):** [deixar em branco para preenchimento]
**Turma:** [faixa etária informada]
**Data:** [deixar em branco]
**Componente Curricular / Campo de Experiência:** [baseado no contexto BNCC]
**Habilidade BNCC:** [código exato do contexto — NUNCA invente]
**Tempo total:** [soma de todas as etapas]

═══════════════════════════════════════════
FORMATO OBRIGATÓRIO
═══════════════════════════════════════════

## 🎯 Objetivo Pedagógico
[Descreva com precisão o que as crianças vão desenvolver]

## 📋 Materiais Necessários
[Lista completa com quantidades baseadas no tamanho da turma]

## ⏱️ Tempo Estimado
[Distribuído por etapa com minutos específicos]

## 🗂️ Campo de Experiência / Componente Curricular (BNCC)
[Nome exato + código(s) retirados LITERALMENTE do contexto BNCC fornecido]

## 📌 Desenvolvimento
[Mínimo 4 etapas detalhadas com tempo. Inclua falas sugeridas entre aspas. A sequência deve ter progressão pedagógica real — cada etapa avança sobre a anterior]

## 👀 O que observar (Avaliação Formativa)
[Indicadores específicos: o que sinaliza desenvolvimento, o que sinaliza dificuldade, o que registrar por criança]

## 📚 Fundamentação Teórica
[Cite trecho real do contexto + conceito do autor. Em escrita/leitura: cite Emília Ferreiro com nível específico para a faixa etária]

## 💡 Adaptações e Variações
[Adaptação para dificuldades, turmas grandes, e proposta de continuidade na próxima aula]

## 📷 Sugestão de Registro
[Como documentar para portfólio e relatório individual do aluno]"""


class PedidoAtividade(BaseModel):
    pedido: str
    faixa_etaria: str
    tema_mes: str = ""
    recursos: str = ""
    objetivo_especifico: str = ""
    tamanho_turma: str = ""


def buscar_por_fonte(query: str, filtro_fonte: str, match_count: int,
                     client: OpenAI, supabase) -> list:
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    embedding = resp.data[0].embedding

    resultado = supabase.rpc("buscar_por_fonte", {
        "query_embedding": embedding,
        "filtro_fonte": filtro_fonte,
        "match_count": match_count
    }).execute()

    return resultado.data or []


def buscar_contexto_estruturado(dados, client: OpenAI, supabase) -> dict:
    prefixo = get_prefixo_bncc(dados.faixa_etaria)
    query_base = f"{dados.pedido} {dados.faixa_etaria} {dados.tema_mes}".strip()
    query_bncc = f"{prefixo} objetivos aprendizagem {dados.pedido}"

    # Busca direcionada por fonte
    chunks_bncc     = buscar_por_fonte(query_bncc, "BNCC", 4, client, supabase)
    chunks_rcnei    = buscar_por_fonte(query_base, "rcnei", 3, client, supabase)
    chunks_vygotsky = buscar_por_fonte(query_base, "VYGOTSKY", 2, client, supabase)
    chunks_piaget   = buscar_por_fonte(query_base, "Piaget", 2, client, supabase)
    chunks_ferreiro = buscar_por_fonte(query_base, "ferreiro", 2, client, supabase)

    return {
        "bncc": chunks_bncc,
        "rcnei": chunks_rcnei,
        "teoricos": chunks_vygotsky + chunks_piaget + chunks_ferreiro
    }


def montar_contexto_estruturado(contextos: dict, prefixo: str) -> str:
    ctx = f"[PREFIXO BNCC PARA ESTA FAIXA ETÁRIA: {prefixo}]\n\n"

    ctx += "[CONTEXTO BNCC — use APENAS os códigos que aparecem aqui]\n"
    if contextos["bncc"]:
        for i, c in enumerate(contextos["bncc"], 1):
            ctx += f"[BNCC {i}] {c['conteudo']}\n"
    else:
        ctx += "Nenhum trecho BNCC encontrado para esta busca.\n"

    ctx += "\n[CONTEXTO RCNEI]\n"
    if contextos["rcnei"]:
        for i, c in enumerate(contextos["rcnei"], 1):
            ctx += f"[RCNEI {i}] {c['conteudo']}\n"

    ctx += "\n[CONTEXTO TEÓRICO — Vygotsky, Piaget, Emília Ferreiro]\n"
    if contextos["teoricos"]:
        for i, c in enumerate(contextos["teoricos"], 1):
            ctx += f"[TEÓRICO {i} — {c['fonte']}] {c['conteudo']}\n"

    return ctx


def gerar_atividade(dados, contexto: str, client: OpenAI) -> str:
    detalhes = f"Pedido: {dados.pedido}\nFaixa etária: {dados.faixa_etaria}"
    if dados.tema_mes:
        detalhes += f"\nTema do mês: {dados.tema_mes}"
    if dados.objetivo_especifico:
        detalhes += f"\nObjetivo específico: {dados.objetivo_especifico}"
    if dados.recursos:
        detalhes += f"\nRecursos disponíveis: {dados.recursos}"
    if dados.tamanho_turma:
        detalhes += f"\nTamanho da turma: {dados.tamanho_turma}"

    mensagem = f"""{contexto}

{detalhes}

INSTRUÇÕES FINAIS:
- Use APENAS os códigos BNCC que aparecem literalmente no [CONTEXTO BNCC] acima.
- Se não encontrar o código, declare isso explicitamente — nunca invente.
- Inclua o cabeçalho obrigatório antes das seções.
- A atividade deve ter progressão pedagógica real entre as etapas.
- Nível de qualidade: uma coordenadora experiente deve aprovar sem ressalvas."""

    resposta = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3500,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": mensagem}
        ]
    )
    return resposta.choices[0].message.content


@app.get("/")
def raiz():
    return {"status": "Jeane online"}


@app.post("/gerar-atividade")
def endpoint_atividade(dados: PedidoAtividade):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        prefixo = get_prefixo_bncc(dados.faixa_etaria)
        contextos = buscar_contexto_estruturado(dados, client, supabase)
        contexto = montar_contexto_estruturado(contextos, prefixo)
        atividade = gerar_atividade(dados, contexto, client)

        return {"atividade": atividade}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
