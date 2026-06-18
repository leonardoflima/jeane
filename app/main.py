import os
import re
import json
import unicodedata
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

def buscar_atividades(embedding, match_count=4):
    try:
        result = supabase.rpc("buscar_documentos", {
            "query_embedding": embedding,
            "match_count": match_count * 8
        }).execute()
        chunks = result.data or []
        filtrados = [c for c in chunks if c.get("categoria") == "atividade_ei"]
        return filtrados[:match_count] if len(filtrados) >= match_count else chunks[:match_count]
    except Exception:
        return []

def normalizar_palavra(s: str) -> str:
    """Remove acentos, maiúscula, só letras, máx 8 chars."""
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return ''.join(c for c in s.upper() if c.isalpha())[:8]

def detectar_intencao(pedido: str) -> dict:
    """Detecta palavras-chave no pedido para forçar tipos específicos."""
    p = pedido.lower()
    return {
        "caca_palavras": bool(re.search(r'ca[çc]a[\s\-]?palavras?', p)),
        "tracejado":     bool(re.search(r'pontilhad|traceja|escrever? (sobre|em cima)|num[eé]ro pontilhad|letra pontilhad', p)),
        "contagem":      bool(re.search(r'conta[rg]|quantos|correspond[eê]ncia num', p)),
    }

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
        "G1":  {"ciclo": "Educação Infantil", "codigos": "EI01", "campo": "Corpo, gestos e movimentos"},
        "G2":  {"ciclo": "Educação Infantil", "codigos": "EI02", "campo": "Traços, sons, cores e formas"},
        "G3":  {"ciclo": "Educação Infantil", "codigos": "EI02/EI03", "campo": "Escuta, fala, pensamento e imaginação"},
        "G4":  {"ciclo": "Educação Infantil — Etapa 1", "codigos": "EI03", "campo": "Espaços, tempos, quantidades, relações e transformações"},
        "G5":  {"ciclo": "Educação Infantil — Etapa 2", "codigos": "EI03", "campo": "Todos os campos de experiência"},
        "1º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF01", "campo": "Língua Portuguesa / Matemática"},
        "2º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF02", "campo": "Língua Portuguesa / Matemática"},
        "3º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF03", "campo": "Língua Portuguesa / Matemática / Ciências"},
        "4º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF04", "campo": "Todas as disciplinas"},
        "5º ano EF": {"ciclo": "Ensino Fundamental", "codigos": "EF05", "campo": "Todas as disciplinas"},
    }

    info_faixa = MAPA_FAIXA.get(faixa_etaria, {"ciclo": faixa_etaria, "codigos": "EF/EI", "campo": ""})

    system_prompt = f"""Você é Jeane, assistente pedagógica especialista em {info_faixa['ciclo']}.

REGRAS ABSOLUTAS:
1. Use SOMENTE os trechos BNCC fornecidos para citar códigos. Se não encontrar, escreva: "Verificar manualmente na BNCC"
2. Nunca invente códigos BNCC
3. Para {faixa_etaria}: use apenas campos de experiência EI ({info_faixa['campo']})
4. Fundamentação teórica: citar autor, obra e conceito específico
5. Desenvolvimento: progressão real introdução → aprofundamento → consolidação → avaliação

CONTEXTO BNCC: {fmt(bncc)}
CONTEXTO TEÓRICO: {fmt(teoricos)}
CONTEXTO RCNEI: {fmt(rcnei)}"""

    user_prompt = f"""Crie um plano de atividade completo:
Pedido: {pedido} | Turma: {faixa_etaria}
{f'Objetivo: {objetivo}' if objetivo else ''} {f'Tema: {tema_mes}' if tema_mes else ''}
{f'Recursos: {recursos}' if recursos else ''} {f'Turma: {tamanho} alunos' if tamanho else ''}

Estruture com: Cabeçalho, Objetivo Pedagógico, Materiais, Tempo, Campo BNCC,
Desenvolvimento (falas reais da professora), Avaliação Formativa, Fundamentação Teórica,
Adaptações, Sugestão de Registro."""

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
    pedido                = body.get("pedido", "")
    faixa_etaria          = body.get("faixa_etaria", "")
    linhas_pontilhadas    = body.get("linhas_pontilhadas", False)
    espaco_desenho        = body.get("espaco_desenho", True)
    tema                  = body.get("tema", "")
    quantidade_exercicios = int(body.get("quantidade_exercicios", 4))

    if not pedido or not faixa_etaria:
        raise HTTPException(status_code=400, detail="pedido e faixa_etaria são obrigatórios")

    quantidade_exercicios = max(2, min(6, quantidade_exercicios))

    # Detecta intenção antes de chamar o GPT
    intencao = detectar_intencao(pedido)

    emb      = gerar_embedding(f"atividade educação infantil {faixa_etaria} {pedido}")
    contexto = buscar_atividades(emb, match_count=4)
    ctx_txt  = "\n\n".join([c["conteudo"] for c in contexto]) if contexto else ""

    FIGURAS_DISPONIVEIS = [
        "sol", "nuvem", "estrela", "lua", "casa", "arvore", "flor", "folha",
        "cachorro", "gato", "peixe", "passaro", "borboleta",
        "maca", "banana", "morango", "coracao", "circulo", "triangulo", "quadrado"
    ]

    # ── PROMPT ESPECÍFICO PARA CAÇA-PALAVRAS ──
    if intencao["caca_palavras"]:
        # Calcula quantos exercícios complementares além do caça-palavras
        n_complementares = quantidade_exercicios - 1
        complementares_instrucao = ""
        if n_complementares > 0:
            complementares_instrucao = f"""
Além do caça-palavras, gere mais {n_complementares} exercício(s) complementar(es) relacionados ao tema.
Use tipos como: circule, completar, desenho, figura, tracejado.
"""
        system_prompt = f"""Você é uma especialista em Educação Infantil criando folhas de atividade.
Retorne SOMENTE JSON válido, sem texto antes/depois, sem markdown.

O pedido é um CAÇA-PALAVRAS. Sua tarefa:
1. Escolher de 4 a 6 palavras simples relacionadas ao tema (sem acento, máx 8 letras cada)
2. Retornar um exercício do tipo "cacapalavras" com essas palavras
3. O sistema monta a grade automaticamente — você só fornece a lista de palavras

⚠️ QUANTIDADE CRÍTICA: O array "exercicios" deve ter EXATAMENTE {quantidade_exercicios} item(ns).
O PRIMEIRO exercício DEVE ser do tipo "cacapalavras".
{complementares_instrucao}

TIPOS DISPONÍVEIS PARA EXERCÍCIOS COMPLEMENTARES:
- "circule": {{"enunciado": "...", "opcoes": ["op1","op2","op3"]}}
- "completar": {{"enunciado": "...", "itens": ["Bo___", "Ca___"]}}
- "desenho": {{"enunciado": "..."}}
- "figura": {{"enunciado": "...", "imagem": "gato", "quantidade": 2}}
- "tracejado": {{"enunciado": "...", "caracteres": ["A","B","C"]}}

FORMATO OBRIGATÓRIO:
{{
  "titulo": "Título divertido",
  "exercicios": [
    {{"tipo": "cacapalavras", "enunciado": "Encontre as palavras escondidas!", "palavras": ["GATO","CASA","SOL"]}},
    ... mais {n_complementares} exercício(s) se solicitado
  ]
}}"""

        user_prompt = f"""Turma: {faixa_etaria}
Tema do caça-palavras: {pedido}
{f'Tema do mês: {tema}' if tema else ''}
Quantidade TOTAL de exercícios: {quantidade_exercicios} (sendo 1 o caça-palavras + {n_complementares} complementar(es))

Retorne JSON com exatamente {quantidade_exercicios} exercício(s), o primeiro sendo cacapalavras."""

    # ── PROMPT GERAL ──
    else:
        # Instrução extra se pedido menciona tracejado/pontilhado
        instrucao_extra = ""
        if intencao["tracejado"]:
            instrucao_extra = "\n⚠️ O pedido menciona escrita pontilhada: use o tipo 'tracejado' com os números ou letras relevantes."
        if intencao["contagem"]:
            instrucao_extra += "\n⚠️ O pedido menciona contagem: use o tipo 'contagem' com grupos de figuras."

        system_prompt = f"""Você é uma especialista em Educação Infantil criando folhas de atividade para alunos.
Retorne SOMENTE JSON válido, sem texto antes/depois, sem markdown.

ARQUITETURA: O frontend gera os visuais a partir dos parâmetros JSON. Você fornece DADOS, não descrições visuais.

⚠️ QUANTIDADE CRÍTICA: O array "exercicios" deve ter EXATAMENTE {quantidade_exercicios} item(ns). Conte antes de retornar.{instrucao_extra}

TIPOS DISPONÍVEIS:

"tracejado" — números/letras pontilhados para traçar
  {{"tipo":"tracejado","enunciado":"...","caracteres":["1","2","3"]}}
  Apenas dígitos 0-9 ou letras A-Z maiúsculas. Máx 6 por exercício.
  → USE quando o pedido mencionar números, letras ou nome para traçar/copiar

"figura" — figura SVG para colorir ou identificar
  {{"tipo":"figura","enunciado":"...","imagem":"gato","quantidade":2}}
  Figuras: sol, nuvem, estrela, lua, casa, arvore, flor, folha, cachorro, gato, peixe, passaro, borboleta, maca, banana, morango, coracao, circulo, triangulo, quadrado

"contagem" — grupos de figuras para contar
  {{"tipo":"contagem","enunciado":"...","grupos":[{{"imagem":"estrela","quantidade":3}},{{"imagem":"flor","quantidade":1}}]}}

"cacapalavras" — grade com palavras escondidas (G4/G5)
  {{"tipo":"cacapalavras","enunciado":"...","palavras":["GATO","CASA","SOL"]}}

"circule" — opções para circular
  {{"tipo":"circule","enunciado":"...","opcoes":["op1","op2","op3"]}}

"completar" — lacunas para preencher
  {{"tipo":"completar","enunciado":"...","itens":["Bo___","Ca___"]}}

"escrever" — linhas para escrever
  {{"tipo":"escrever","enunciado":"...","linhas":2}}

"desenho" — espaço para desenhar
  {{"tipo":"desenho","enunciado":"..."}}

REGRAS POR TURMA:
- Berçário/G1/G2: apenas figura, circule, desenho
- G3: figura, circule, desenho, contagem
- G4: todos exceto cacapalavras
- G5: todos os tipos
- Varie os tipos — não repita o mesmo consecutivamente

FORMATO: {{"titulo":"Título divertido","exercicios":[/* exatamente {quantidade_exercicios} itens */]}}"""

        user_prompt = f"""Turma: {faixa_etaria}
Tema: {pedido}
{f'Tema do mês: {tema}' if tema else ''}
{'Linhas pontilhadas para escrita.' if linhas_pontilhadas else ''}
Quantidade OBRIGATÓRIA: {quantidade_exercicios} exercício(s). Não gere mais nem menos.

Referência pedagógica:
{ctx_txt if ctx_txt else 'Use seu conhecimento pedagógico.'}

Retorne JSON com exatamente {quantidade_exercicios} exercício(s)."""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        texto = resp.choices[0].message.content.strip()
        texto = texto.replace("```json", "").replace("```", "").strip()
        dados = json.loads(texto)

        if "exercicios" not in dados or not isinstance(dados["exercicios"], list):
            raise ValueError("JSON inválido: sem array exercicios")

        # ── ENFORCE DURO DE QUANTIDADE ──
        dados["exercicios"] = dados["exercicios"][:quantidade_exercicios]

        # ── SE ERA CAÇA-PALAVRAS E O MODELO NÃO GEROU, FORÇA ──
        if intencao["caca_palavras"]:
            tem_caca = any(ex.get("tipo") == "cacapalavras" for ex in dados["exercicios"])
            if not tem_caca:
                # Extrai palavras do tema e injeta forçado
                palavras_fallback = [normalizar_palavra(w) for w in re.findall(r'\b[a-zA-ZÀ-ú]{3,8}\b', pedido)]
                palavras_fallback = [p for p in palavras_fallback if len(p) >= 3][:6]
                if not palavras_fallback:
                    palavras_fallback = ["GATO", "CASA", "SOL", "MAR"]
                caca_exercicio = {
                    "tipo": "cacapalavras",
                    "enunciado": "Encontre as palavras escondidas na grade!",
                    "palavras": palavras_fallback
                }
                dados["exercicios"] = [caca_exercicio] + dados["exercicios"][:(quantidade_exercicios - 1)]

        # ── SANITIZA CADA EXERCÍCIO ──
        for ex in dados["exercicios"]:
            tipo = ex.get("tipo", "")

            if tipo == "tracejado":
                chars_validos = []
                for c in (ex.get("caracteres") or []):
                    c = str(c).strip().upper()
                    if len(c) == 1 and (c.isdigit() or c.isalpha()):
                        chars_validos.append(c)
                # Fallback: tenta extrair números do enunciado
                if not chars_validos:
                    m = re.search(r'(\d)\s*[aà]\s*(\d)', ex.get("enunciado", ""))
                    if m:
                        chars_validos = [str(n) for n in range(int(m[1]), int(m[2])+1)]
                ex["caracteres"] = chars_validos[:6]

            elif tipo == "figura":
                if ex.get("imagem") not in FIGURAS_DISPONIVEIS:
                    ex["imagem"] = "estrela"
                ex["quantidade"] = max(1, min(5, int(ex.get("quantidade", 1))))

            elif tipo == "contagem":
                grupos = ex.get("grupos") or []
                grupos_validos = []
                for g in grupos[:5]:
                    img = g.get("imagem", "estrela")
                    if img not in FIGURAS_DISPONIVEIS:
                        img = "estrela"
                    qtd = max(1, min(6, int(g.get("quantidade", 1))))
                    grupos_validos.append({"imagem": img, "quantidade": qtd})
                ex["grupos"] = grupos_validos

            elif tipo == "cacapalavras":
                palavras = ex.get("palavras") or []
                palavras_ok = [normalizar_palavra(p) for p in palavras if p]
                palavras_ok = [p for p in palavras_ok if 2 <= len(p) <= 8][:8]
                if not palavras_ok:
                    palavras_ok = ["GATO", "CASA", "SOL"]
                ex["palavras"] = palavras_ok

        return dados

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Erro JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
