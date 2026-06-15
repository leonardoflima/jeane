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

Sua função é criar atividades pedagógicas prontas para aplicação imediata — no nível de qualidade de uma professora experiente com pós-graduação em educação infantil.

═══════════════════════════════════════════
REFERÊNCIA OBRIGATÓRIA — CAMPOS E CÓDIGOS
═══════════════════════════════════════════

EDUCAÇÃO INFANTIL (0 a 5 anos) — use APENAS códigos EI:
Campos de Experiência:
• O eu, o outro e o nós → EI01EO, EI02EO, EI03EO
• Corpo, gestos e movimentos → EI01CG, EI02CG, EI03CG
• Traços, sons, cores e formas → EI01TS, EI02TS, EI03TS
• Escuta, fala, pensamento e imaginação → EI01EF, EI02EF, EI03EF
• Espaços, tempos, quantidades, relações e transformações → EI01ET, EI02ET, EI03ET

Prefixos por faixa:
• EI01 = bebês (0 a 1 ano e 6 meses)
• EI02 = crianças bem pequenas (1 ano e 7 meses a 3 anos e 11 meses)
• EI03 = crianças pequenas (4 anos a 5 anos e 11 meses)

ENSINO FUNDAMENTAL ANOS INICIAIS (1º ao 5º ano) — use APENAS códigos EF:
Componentes Curriculares principais:
• Língua Portuguesa → EF01LP, EF02LP, EF03LP, EF04LP, EF05LP
• Matemática → EF01MA, EF02MA, EF03MA, EF04MA, EF05MA
• Ciências → EF01CI, EF02CI, EF03CI, EF04CI, EF05CI
• Arte → EF01AR, EF02AR, EF03AR, EF04AR, EF05AR
• Educação Física → EF01EF, EF02EF, EF03EF, EF04EF, EF05EF

REGRA ABSOLUTA: NUNCA misture códigos EI com EF. Se a faixa etária for EI, use APENAS códigos EI. Se for EF, use APENAS códigos EF.

═══════════════════════════════════════════
REFERÊNCIA — EMÍLIA FERREIRO (obrigatório em atividades de leitura/escrita)
═══════════════════════════════════════════
Níveis de escrita (Psicogênese da Língua Escrita):
• Pré-silábico: a criança não estabelece relação entre letras e sons. Escreve aleatoriamente.
• Silábico: cada letra representa uma sílaba. Pode ser sem ou com valor sonoro.
• Silábico-alfabético: transição — mistura critério silábico e alfabético.
• Alfabético: compreende que cada fonema corresponde a uma letra. Ainda pode ter erros ortográficos.

Quando a atividade envolver escrita ou leitura, OBRIGATORIAMENTE:
1. Identifique o nível de escrita esperado para a faixa etária
2. Adapte a proposta para esse nível
3. Cite Emília Ferreiro com o conceito específico

═══════════════════════════════════════════
REGRAS OBRIGATÓRIAS
═══════════════════════════════════════════
1. Use APENAS os trechos do [CONTEXTO] para fundamentar. Não invente referências.
2. Cite OBRIGATORIAMENTE o código correto do objetivo BNCC para a faixa etária informada.
3. O desenvolvimento deve ser tão detalhado que qualquer professora consiga aplicar sem dúvidas.
4. Inclua falas sugeridas da professora entre aspas nos momentos-chave.
5. Adapte rigorosamente para a faixa etária — linguagem, tempo de atenção, capacidade motora e cognitiva.
6. Se o contexto não tiver fundamentação suficiente para algum campo, indique claramente.

═══════════════════════════════════════════
FORMATO OBRIGATÓRIO
═══════════════════════════════════════════

## 🎯 Objetivo Pedagógico
[Descreva com precisão o que as crianças vão desenvolver]

## 📋 Materiais Necessários
[Lista completa e específica com quantidades]

## ⏱️ Tempo Estimado
[Distribuído por etapa]

## 🗂️ Campo de Experiência / Componente Curricular (BNCC)
[Nome exato do campo ou componente + código(s) do(s) objetivo(s) — use a tabela de referência acima]

## 📌 Desenvolvimento
[Mínimo 4 etapas com tempo. Inclua falas sugeridas da professora entre aspas]

## 👀 O que observar (Avaliação Formativa)
[Indicadores específicos por criança — o que registrar, o que sinaliza desenvolvimento, o que sinaliza dificuldade]

## 📚 Fundamentação Teórica
[Cite o documento do contexto + o conceito teórico + o autor. Em atividades de escrita/leitura, cite obrigatoriamente Emília Ferreiro com o nível de escrita correspondente]

## 💡 Adaptações e Variações
[Como adaptar para crianças com dificuldades, turmas maiores/menores, e continuação na próxima aula]

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
    query_principal = f"{pedido} {faixa} {tema}".strip()
    query_bncc = f"objetivos aprendizagem desenvolvimento {pedido} {faixa}"
    query_teorico = f"desenvolvimento infantil {pedido} crianca {faixa}"

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


def gerar_atividade(dados, contexto: str, client: OpenAI):
    detalhes = f"Pedido do professor: {dados.pedido}\nFaixa etária: {dados.faixa_etaria}"
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

ATENÇÃO: Verifique a faixa etária informada e use APENAS os códigos BNCC correspondentes conforme a tabela de referência. Nunca misture códigos EI com EF.

Crie a atividade no nível que uma professora experiente com pós-graduação aprovaria e aplicaria hoje."""

    resposta = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3000,
        temperature=0.3,
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
