import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from supabase import create_client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

openai_client = OpenAI(api_key=OPENAI_KEY)
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── HELPERS ──────────────────────────────────────────────────

def gerar_embedding(texto: str):
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texto[:8000]
    )
    return resp.data[0].embedding

def buscar_chunks(embedding, match_count=5, categoria_filtro=None):
    try:
        if categoria_filtro:
            result = supabase.rpc("buscar_por_fonte", {
                "query_embedding": embedding,
                "filtro_fonte": categoria_filtro,
                "match_count": match_count
            }).execute()
        else:
            result = supabase.rpc("buscar_documentos", {
                "query_embedding": embedding,
                "match_count": match_count
            }).execute()
        return result.data or []
    except Exception:
        return []

def buscar_por_categoria(embedding, categoria: str, match_count=4):
    """Busca chunks filtrando pelo campo categoria (não fonte)."""
    try:
        result = supabase.table("documentos").select(
            "conteudo, fonte, categoria"
        ).eq("categoria", categoria).limit(50).execute()

        # Sem busca vetorial por categoria, retorna amostra representativa
        return result.data[:match_count] if result.data else []
    except Exception:
        return []

# ── ENDPOINTS ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Jeane online"}


@app.post("/gerar-atividade")
async def gerar_atividade(req: Request):
    body = await req.json()
    pedido       = body.get("pedido", "")
    faixa_etaria = body.get("faixa_etaria", "")
    objetivo     = body.get("objetivo", "")
    tema_mes     = body.get("tema_mes", "")
    recursos     = body.get("recursos", "")
    tamanho      = body.get("tamanho_turma", "")

    if not pedido:
        raise HTTPException(status_code=400, detail="pedido é obrigatório")

    query = f"{faixa_etaria} {pedido} {objetivo}"
    emb   = gerar_embedding(query)

    bncc     = buscar_chunks(emb, match_count=3, categoria_filtro="BNCC")
    teoricos = buscar_chunks(emb, match_count=2, categoria_filtro="VYGOTSKY")
    rcnei    = buscar_chunks(emb, match_count=2, categoria_filtro="rcnei")

    def fmt(chunks):
        return "\n\n".join([f"[{c['fonte']}]\n{c['conteudo']}" for c in chunks]) if chunks else "Não localizado."

    MAPA_FAIXA = {
        "Berçário": {"ciclo": "Educação Infantil", "codigos": "EI01", "campo": "O eu, o outro e o nós"},
        "G1": {"ciclo": "Educação Infantil", "codigos": "EI01", "campo": "Corpo, gestos e movimentos"},
        "G2": {"ciclo": "Educação Infantil", "codigos": "EI02", "campo": "Traços, sons, cores e formas"},
        "G3": {"ciclo": "Educação Infantil", "codigos": "EI02/EI03", "campo": "Escuta, fala, pensamento e imaginação"},
        "G4": {"ciclo": "Educação Infantil — Etapa 1", "codigos": "EI03", "campo": "Espaços, tempos, quantidades, relações e transformações"},
        "G5": {"ciclo": "Educação Infantil — Etapa 2", "codigos": "EI03", "campo": "Todos os campos de experiência"},
        "1º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF01", "campo": "Língua Portuguesa / Matemática"},
        "2º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF02", "campo": "Língua Portuguesa / Matemática"},
        "3º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF03", "campo": "Língua Portuguesa / Matemática / Ciências"},
        "4º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF04", "campo": "Todas as disciplinas"},
        "5º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF05", "campo": "Todas as disciplinas"},
    }

    info_faixa = MAPA_FAIXA.get(faixa_etaria, {"ciclo": faixa_etaria, "codigos": "EF/EI", "campo": ""})

    system_prompt = f"""Você é Jeane, assistente pedagógica especialista em {info_faixa['ciclo']}.

REGRAS ABSOLUTAS:
1. Use SOMENTE os trechos BNCC fornecidos para citar códigos. Se não encontrar o código exato, escreva: "Verificar manualmente na BNCC — habilidade relacionada a [descreva]"
2. Nunca invente códigos BNCC
3. Para {faixa_etaria}: use apenas campos de experiência EI (ex: {info_faixa['campo']}) — nunca misture com componentes EF
4. A fundamentação teórica deve citar autor, obra e conceito específico
5. O desenvolvimento deve ter progressão real: introdução → aprofundamento → consolidação → avaliação

CONTEXTO BNCC DISPONÍVEL:
{fmt(bncc)}

CONTEXTO TEÓRICO:
{fmt(teoricos)}

CONTEXTO RCNEI:
{fmt(rcnei)}"""

    user_prompt = f"""Crie um plano de atividade completo:

Pedido: {pedido}
Faixa etária / Turma: {faixa_etaria}
{f'Objetivo específico: {objetivo}' if objetivo else ''}
{f'Tema do mês: {tema_mes}' if tema_mes else ''}
{f'Recursos disponíveis: {recursos}' if recursos else ''}
{f'Tamanho da turma: {tamanho} alunos' if tamanho else ''}

Estruture assim:
## Cabeçalho
Professor(a): [deixar em branco para preenchimento]
Turma: {faixa_etaria}
Data: [deixar em branco]
Componente Curricular / Campo de Experiência: [preencher]
Habilidade BNCC: [código ou "Verificar manualmente"]
Tempo total: [X minutos]

## 🎯 Objetivo Pedagógico
## 📋 Materiais Necessários
## ⏱️ Tempo Estimado
## 🗂 Campo de Experiência / Componente Curricular (BNCC)
## 📌 Desenvolvimento
(com falas reais da professora entre aspas e progressão pedagógica clara)
## 👀 O que observar (Avaliação Formativa)
## 📚 Fundamentação Teórica
(autor + obra + conceito específico)
## 💡 Adaptações e Variações
## 📷 Sugestão de Registro"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.5,
        max_tokens=2000
    )

    return {"atividade": resp.choices[0].message.content}


@app.post("/gerar-atividade-aluno")
async def gerar_atividade_aluno(req: Request):
    body = await req.json()
    pedido             = body.get("pedido", "")
    faixa_etaria       = body.get("faixa_etaria", "")
    linhas_pontilhadas = body.get("linhas_pontilhadas", False)
    espaco_desenho     = body.get("espaco_desenho", True)
    tema               = body.get("tema", "")

    if not pedido or not faixa_etaria:
        raise HTTPException(status_code=400, detail="pedido e faixa_etaria são obrigatórios")

    emb        = gerar_embedding(f"atividade educação infantil {faixa_etaria} {pedido}")
    contexto   = buscar_por_categoria(emb, "atividade_ei", match_count=4)
    ctx_txt    = "\n\n".join([c["conteudo"] for c in contexto]) if contexto else ""

    system_prompt = """Você é uma especialista em Educação Infantil criando folhas de atividade para alunos.
Retorne SOMENTE um JSON válido, sem texto antes ou depois, sem markdown, sem blocos de código.

Tipos de exercício disponíveis:
- "escrever": linhas para escrever (campo: "linhas": número)
- "completar": itens para completar (campo: "itens": ["palavra1 ___", "palavra2 ___"])
- "circule": opções para circular (campo: "opcoes": ["op1","op2","op3","op4"])
- "desenho": espaço para desenhar
- "caixa": caixa livre

Regras por turma:
- Berçário, G1, G2: apenas "desenho" e "circule"
- G3: "desenho", "circule", "caixa"
- G4, G5: todos os tipos
- Gere 3 a 5 exercícios
- Enunciados curtos e simples para crianças

Formato JSON:
{
  "titulo": "Título curto e divertido",
  "exercicios": [
    {"tipo": "circule", "enunciado": "Enunciado aqui", "opcoes": ["op1","op2","op3"]},
    {"tipo": "desenho", "enunciado": "Enunciado aqui"},
    {"tipo": "escrever", "enunciado": "Enunciado aqui", "linhas": 2}
  ]
}"""

    user_prompt = f"""Crie atividade para turma {faixa_etaria}.
Tema: {pedido}
{f'Tema do mês: {tema}' if tema else ''}
{'Incluir linhas pontilhadas para escrita.' if linhas_pontilhadas else ''}

Referência pedagógica:
{ctx_txt if ctx_txt else 'Use seu conhecimento pedagógico.'}

Retorne apenas o JSON."""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        texto = resp.choices[0].message.content.strip()
        texto = texto.replace("```json", "").replace("```", "").strip()
        dados = json.loads(texto)
        return dados
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Erro JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
