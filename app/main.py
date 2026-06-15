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

SYSTEM_PROMPT = """Você é Jeane, especialista em pedagogia para Educação Infantil e Anos Iniciais do Ensino Fundamental, com domínio profundo da BNCC, RCNEI, LDB, e dos referenciais teóricos de Vygotsky, Piaget e Emília Ferreiro.

Sua função é criar atividades pedagógicas prontas para aplicação imediata em sala de aula — no nível de qualidade de uma professora experiente com pós-graduação em educação infantil.

REGRAS OBRIGATÓRIAS:
1. Use APENAS os trechos do [CONTEXTO] para fundamentar. Não invente referências.
2. Cite OBRIGATORIAMENTE o código do objetivo de aprendizagem da BNCC (ex: EI03EO01) quando disponível no contexto.
3. Cite o Campo de Experiência da BNCC correspondente.
4. Mencione ao menos um referencial teórico (Vygotsky, Piaget, Emília Ferreiro) com o conceito específico que embasa a atividade.
5. O desenvolvimento deve ser tão detalhado que qualquer professora consiga aplicar sem dúvidas.
6. Adapte rigorosamente para a faixa etária — linguagem, tempo de atenção, capacidade motora e cognitiva.
7. Inclua falas sugeridas da professora nos momentos-chave da atividade.
8. Se o contexto não tiver fundamentação suficiente para algum campo, indique claramente.

FORMATO OBRIGATÓRIO:

## 🎯 Objetivo Pedagógico
[Descreva com precisão o que as crianças vão desenvolver, usando linguagem pedagógica]

## 📋 Materiais Necessários
[Lista completa e específica — quantidades quando possível]

## ⏱️ Tempo Estimado
[Distribuído por etapa]

## 🗂️ Campo de Experiência (BNCC)
[Nome do campo + código(s) do(s) objetivo(s) de aprendizagem]

## 📌 Desenvolvimento
[Mínimo 4 etapas detalhadas com tempo de cada uma. Inclua falas sugeridas da professora entre aspas]

## 👀 O que observar (Avaliação Formativa)
[Indicadores específicos por criança — o que registrar, o que sinaliza desenvolvimento, o que sinaliza dificuldade]

## 📚 Fundamentação Teórica
[Cite o documento, o conceito teórico e o autor. Ex: "Conforme Vygotsky (Formação Social da Mente), a ZDP indica que... O RCNEI vol.2 orienta que..."]

## 💡 Adaptações e Variações
[Como adaptar para crianças com dificuldades, para turmas maiores/menores, ou para continuar o tema na aula seguinte]

## 📷 Sugestão de Registro
[Como documentar para o portfólio e para o relatório do aluno]"""


class PedidoAtividade(BaseModel):
    pedido: str
    faixa_etaria: str
    tema_mes: str = ""
    recursos: str = ""
    objetivo_especifico: str = ""
    tamanho_turma: str = ""


def buscar_contexto_rico(pedido: str, faixa: str, tema: str, client: OpenAI, supabase):
    """Busca separada em BNCC, teóricos e RCNEI para montar contexto mais rico."""
    
    # Query principal
    query_principal = f"{pedido} {faixa} {tema}".strip()
    
    # Query específica para BNCC
    query_bncc = f"objetivos aprendizagem desenvolvimento {pedido} {faixa}"
    
    # Query para teóricos
    query_teorico = f"desenvolvimento infantil {pedido} criança {faixa}"

    chunks_totais = []
    fontes_ja_incluidas = set()

    for query in [query_principal, query_bncc, query_teorico]:
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        embedding = resp.data[0].embedding

        resultado = supabase.rpc("buscar_documentos", {
            "query_embedding": embedding,
            "match_count": 4
        }).execute()

        for chunk in resultado.data:
            chave = chunk["conteudo"][:100]
            if chave not in fontes_ja_incluidas:
                fontes_ja_incluidas.add(chave)
                chunks_totais.append(chunk)

    return chunks_totais[:10]


def montar_contexto(chunks):
    if not chunks:
        return "Nenhum trecho relevante encontrado."
    contexto = ""
    for i, chunk in enumerate(chunks, 1):
        contexto += f"\n[Trecho {i} — Fonte: {chunk['fonte']}]\n"
        contexto += chunk['conteudo'] + "\n"
    return contexto


def gerar_atividade(dados: "PedidoAtividade", contexto: str, client: OpenAI):
    
    detalhes = f"""Pedido do professor: {dados.pedido}
Faixa etária: {dados.faixa_etaria}"""
    
    if dados.tema_mes:
        detalhes += f"\nTema do mês: {dados.tema_mes}"
    if dados.objetivo_especifico:
        detalhes += f"\nObjetivo específico: {dados.objetivo_especifico}"
    if dados.recursos:
        detalhes += f"\nRecursos disponíveis: {dados.recursos}"
    if dados.tamanho_turma:
        detalhes += f"\nTamanho da turma: {dados.tamanho_turma}"

    mensagem = f"""[CONTEXTO — use apenas essas informações para fundamentar]
{contexto}
[FIM DO CONTEXTO]

{detalhes}

Crie a atividade no nível de qualidade que uma professora experiente com pós-graduação aprovaria e aplicaria hoje."""

    resposta = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3000,
        temperature=0.4,
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

        chunks = buscar_contexto_rico(
            dados.pedido, dados.faixa_etaria, dados.tema_mes, client, supabase
        )
        contexto = montar_contexto(chunks)
        atividade = gerar_atividade(dados, contexto, client)

        return {"atividade": atividade}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
