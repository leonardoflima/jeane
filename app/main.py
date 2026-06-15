from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
import os

app = FastAPI()

# Permite que o site frontend acesse o backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pega as chaves das variáveis de ambiente (configuradas no Vercel)
OPENAI_KEY   = os.environ.get("OPENAI_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

SYSTEM_PROMPT = """Você é Jeane, uma especialista em pedagogia para Educação Infantil e Anos Iniciais do Ensino Fundamental.

Sua função é criar atividades pedagógicas fundamentadas, prontas para uso em sala de aula.

REGRAS OBRIGATÓRIAS:
1. Use APENAS os trechos do [CONTEXTO] para fundamentar a atividade. Não invente referências.
2. Se o contexto não tiver informação suficiente, diga claramente quais partes não puderam ser fundamentadas.
3. Cite sempre a fonte (nome do documento) ao referenciar algo do contexto.
4. Adapte a linguagem para ser prática e direta — o professor deve conseguir aplicar a atividade no mesmo dia.

FORMATO OBRIGATÓRIO DA RESPOSTA:

## 🎯 Objetivo Pedagógico
[O que as crianças vão desenvolver com essa atividade]

## 📋 Materiais Necessários
[Lista de materiais]

## ⏱️ Tempo Estimado
[Duração sugerida]

## 📌 Desenvolvimento
[Passo a passo detalhado da atividade]

## 👀 O que observar (Avaliação)
[O que o professor deve observar em cada criança durante a atividade]

## 📚 Fundamentação
[Referência ao documento e trecho que embasou a atividade]

## 💡 Sugestão de Registro
[Como o professor pode registrar e documentar essa atividade]"""


class PedidoAtividade(BaseModel):
    pedido: str
    faixa_etaria: str


def buscar_contexto(pedido: str, client: OpenAI, supabase):
    resposta = client.embeddings.create(
        model="text-embedding-ada-002",
        input=pedido
    )
    embedding = resposta.data[0].embedding

    resultado = supabase.rpc("buscar_documentos", {
        "query_embedding": embedding,
        "match_count": 5
    }).execute()

    return resultado.data


def montar_contexto(chunks):
    if not chunks:
        return "Nenhum trecho relevante encontrado."
    contexto = ""
    for i, chunk in enumerate(chunks, 1):
        contexto += f"\n[Trecho {i} — Fonte: {chunk['fonte']}]\n"
        contexto += chunk['conteudo'] + "\n"
    return contexto


def gerar_atividade(pedido: str, faixa_etaria: str, contexto: str, client: OpenAI):
    mensagem = f"""[CONTEXTO - use apenas essas informações para fundamentar a atividade]
{contexto}
[FIM DO CONTEXTO]

Pedido do professor: {pedido}
Faixa etária: {faixa_etaria}

Crie a atividade seguindo o formato do sistema."""

    resposta = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
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
        client = OpenAI(api_key=OPENAI_KEY)
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        chunks = buscar_contexto(dados.pedido, client, supabase)
        contexto = montar_contexto(chunks)
        atividade = gerar_atividade(dados.pedido, dados.faixa_etaria, contexto, client)

        return {"atividade": atividade}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
