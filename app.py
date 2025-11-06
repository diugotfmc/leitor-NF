# -*- coding: utf-8 -*-
import io
import math
import unicodedata
from typing import List, Tuple, Dict, Optional

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF


# ===============================
# Utils
# ===============================
def no_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return s.upper()


def words_from_pdf(file_bytes: bytes) -> List[List[Tuple[float, float, float, float, str]]]:
    """
    Retorna lista por página. Cada item é (x0, y0, x1, y1, texto).
    """
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            ws = page.get_text("words")  # (x0, y0, x1, y1, "text", block, line, word_no)
            ws = [(w[0], w[1], w[2], w[3], w[4]) for w in ws]
            ws.sort(key=lambda t: (round(t[1], 1), t[0]))
            pages.append(ws)
    return pages


def group_lines(words: List[Tuple[float, float, float, float, str]], tol_y: float = 2.0):
    """
    Agrupa palavras por linha (y). Retorna lista de linhas; cada linha é lista de palavras ordenadas por x.
    """
    linhas = []
    cur_y = None
    cur = []
    for (x0, y0, x1, y1, t) in words:
        if cur_y is None:
            cur_y = y0
            cur = [(x0, y0, x1, y1, t)]
            continue
        if abs(y0 - cur_y) <= tol_y:
            cur.append((x0, y0, x1, y1, t))
        else:
            linhas.append(sorted(cur, key=lambda z: z[0]))
            cur = [(x0, y0, x1, y1, t)]
            cur_y = y0
    if cur:
        linhas.append(sorted(cur, key=lambda z: z[0]))
    return linhas


# ===============================
# Detecção de cabeçalho e colunas
# ===============================
HEADER_TARGETS = [
    ("COD", ["CÓD.", "COD.", "COD", "CÓD"]),
    ("DESCRICAO", ["DESCRIÇÃO", "DESCRICAO"]),
    ("NCM/SH", ["NCM/SH"]),
    ("CFOP", ["CFOP"]),
    ("UN", ["UN"]),
    ("QTD", ["QTD"]),
    ("V_UNITARIO", ["V.", "V", "VALOR"], "UNITARIO"),
    ("V_TOTAL", ["V.", "V", "VALOR"], "TOTAL"),
]
# Observação: Para "V. UNITÁRIO" e "V. TOTAL" a DANFE traz separado "V." + "UNITÁRIO/TOTAL".


def find_header_positions(lines: List[List[Tuple[float, float, float, float, str]]]) -> Optional[Dict[str, float]]:
    """
    Tenta localizar, numa única linha, os títulos das colunas e retorna o x inicial de cada uma.
    Retorna também a y (linha do cabeçalho) via chave especial "__y__".
    """
    for ln in lines:
        tokens = [no_accents_upper(w[4]) for w in ln]
        # Linha candidata tem que conter ao menos 5 dos rótulos
        joined = " ".join(tokens)
        score = 0
        for lbls in ["DADOS DO PRODUTO", "DADOS DO PRODUTO / SERVICO", "DADOS DO PRODUTO / SERVIÇO"]:
            if lbls in joined:
                score += 1
                break

        # Fermamente, busque por presença de NCM/SH e CFOP na mesma linha
        if not any("NCM/SH" in t for t in tokens):
            continue
        if not any("CFOP" in t for t in tokens):
            continue

        # Mapa {coluna: x0}
        col_x = {}
        for i, (x0, y0, x1, y1, text) in enumerate(ln):
            t = no_accents_upper(text)

            # COD PROD: geralmente aparece "CÓD." seguido logo de "PROD."
            if t in {"COD.", "COD", "CÓD.", "CÓD"}:
                # checa se próximo é "PROD."
                nxt = no_accents_upper(ln[i + 1][4]) if i + 1 < len(ln) else ""
                if nxt.startswith("PROD"):
                    col_x["COD"] = x0

            # DESCRIÇÃO
            if t.startswith("DESCRICAO") or t.startswith("DESCRIÇÃO"):
                col_x["DESCRICAO"] = x0

            # NCM/SH
            if "NCM/SH" in t:
                col_x["NCM/SH"] = x0

            # CFOP
            if "CFOP" == t:
                col_x["CFOP"] = x0

            # UN
            if t == "UN":
                col_x["UN"] = x0

            # QTD
            if t == "QTD":
                col_x["QTD"] = x0

            # V. UNITARIO (V.  +  UNITARIO)
            if t in {"V.", "V"} and i + 1 < len(ln):
                next_t = no_accents_upper(ln[i + 1][4])
                if next_t.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if next_t.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0

            # Algumas DANFEs trazem "VALOR UNITÁRIO"/"VALOR TOTAL"
            if t == "VALOR" and i + 1 < len(ln):
                next_t = no_accents_upper(ln[i + 1][4])
                if next_t.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if next_t.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0

        # Critério mínimo: precisamos de pelo menos estas 6 colunas
        needed = {"COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"}
        have = set(col_x.keys())
        if len(needed.intersection(have)) >= 6:
            col_x["__y__"] = ln[0][1]  # y do cabeçalho
            return col_x

    return None


def build_column_edges(col_x: Dict[str, float], page_width: float) -> List[Tuple[str, float, float]]:
    """
    Com os x dos títulos, gera intervalos [x_ini, x_fim) para cada coluna.
    """
    # pega as colunas base
    keys = [k for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"] if k in col_x]
    # ordena por x
    keys.sort(key=lambda k: col_x[k])
    xs = [col_x[k] for k in keys]
    # limites: meio do caminho entre vizinhos
    edges = []
    for i, k in enumerate(keys):
        left = (xs[i - 1] + xs[i]) / 2 if i > 0 else max(0, xs[i] - 5)
        right = (xs[i] + xs[i + 1]) / 2 if i + 1 < len(xs) else page_width
        edges.append((k, left, right))
    return edges


# ===============================
# Extração das linhas da tabela
# ===============================
END_MARKERS = [
    "DADOS ADICIONAIS",
    "INFORMACOES COMPLEMENTARES",
    "INFORMAÇÕES COMPLEMENTARES",
    "DADOS ADICIONAIS (COMPLEMENTO)",
    "INFORMACOES ADICIONAIS",
    "INFORMAÇÕES ADICIONAIS",
]


def extract_table_page(page, unite_wrapped_description: bool = True) -> List[Dict[str, str]]:
    """
    Extrai as linhas de uma página, respeitando o cabeçalho encontrado naquela página.
    Retorna lista de dicts (uma linha por item "como no PDF").
    """
    words = page.get_text("words")
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in words]
    words.sort(key=lambda t: (round(t[1], 1), t[0]))

    lines = group_lines(words, tol_y=2.0)
    if not lines:
        return []

    # Detecta header nesta página
    col_x = find_header_positions(lines)
    if not col_x:
        return []

    header_y = col_x["__y__"]
    page_width = page.rect.width
    col_edges = build_column_edges(col_x, page_width)

    # Varre linhas abaixo do cabeçalho até o marcador de fim
    rows = []
    past_header = False
    for ln in lines:
        y = ln[0][1]
        row_text_all = " ".join(no_accents_upper(w[4]) for w in ln)

        if not past_header:
            # pula linhas antes do cabeçalho
            if y <= header_y + 0.5:
                continue
            past_header = True

        # Se chega nos marcadores de término, para
        if any(m in row_text_all for m in END_MARKERS):
            break

        # Monta células por coluna
        row = {k: "" for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]}
        for (x0, y0, x1, y1, t) in ln:
            xc = (x0 + x1) / 2
            placed = False
            for (k, left, right) in col_edges:
                if left <= xc < right:
                    # adiciona com espaço
                    if row.get(k):
                        row[k] += " " + t
                    else:
                        row[k] = t
                    placed = True
                    break
            if not placed:
                # se ficou fora, joga na descrição (mais seguro para manter "tudo o que está na tabela")
                row["DESCRICAO"] = (row.get("DESCRICAO", "") + " " + t).strip()

        rows.append(row)

    # Unir linhas quebradas de descrição (opcional)
    if unite_wrapped_description and rows:
        merged = []
        for r in rows:
            is_continuation = (
                r["COD"].strip() == "" and
                r["NCM/SH"].strip() == "" and
                r["CFOP"].strip() == "" and
                r["UN"].strip() == "" and
                r["QTD"].strip() == "" and
                r["V_UNITARIO"].strip() == "" and
                r["V_TOTAL"].strip() == "" and
                r["DESCRICAO"].strip() != ""
            )
            if is_continuation and merged:
                merged[-1]["DESCRICAO"] = (merged[-1]["DESCRICAO"] + " " + r["DESCRICAO"]).strip()
            else:
                merged.append(r)
        rows = merged

    return rows


def extract_table_full(file_bytes: bytes, unite_wrapped_description: bool = True) -> pd.DataFrame:
    """
    Percorre todas as páginas; em cada página, detecta cabeçalho e extrai linhas.
    Concatena mantendo a ordem.
    """
    out_rows = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            page_rows = extract_table_page(p, unite_wrapped_description=unite_wrapped_description)
            out_rows.extend(page_rows)

    if not out_rows:
        return pd.DataFrame(columns=["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"])
    df = pd.DataFrame(out_rows)

    # Limpeza mínima (sem normalizar valores!): strip espaços
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="Leitura exata da Tabela de Itens (DANFE)", layout="wide")
st.title("🧾 Leitura EXATA da Tabela de Itens da NF (DANFE) — sem heurísticas")

st.markdown(
    """
Este app **lê a tabela como está no PDF**, usando as **posições de coluna do cabeçalho** da própria DANFE.  
Nada de interpretar ou recalcular valores: primeiro ele **captura a tabela crua**.
"""
)

file = st.file_uploader("Selecione o PDF da DANFE", type=["pdf"])
unir_quebras = st.checkbox("Unir linhas quebradas de descrição (recomendado)", value=True)
mostrar_preview = st.checkbox("Mostrar algumas linhas brutas para conferência", value=True)

colA, colB = st.columns(2)
with colA:
    btn = st.button("📤 Extrair tabela")
with colB:
    baixar_normal = st.checkbox("Baixar CSV/XLSX após extrair", value=True)

if btn and file is not None:
    raw = file.read()
    with st.spinner("Lendo a tabela diretamente do PDF..."):
        df = extract_table_full(raw, unite_wrapped_description=unir_quebras)

    if df.empty:
        st.error("Não consegui localizar o cabeçalho da tabela nesta DANFE ou não havia linhas sob o cabeçalho.")
        st.info("Se quiser, me envie esse PDF para eu calibrar o detector de cabeçalho.")
    else:
        st.success(f"Tabela capturada. Linhas: {len(df)}")
        st.dataframe(df, use_container_width=True, height=420)

        if mostrar_preview:
            st.subheader("Prévia (primeiras 10 linhas cruas)")
            st.write(df.head(10))

        if baixar_normal:
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Baixar CSV (tabela crua)", data=csv_bytes, file_name="itens_danfe_cru.csv", mime="text/csv")

            xls = io.BytesIO()
            with pd.ExcelWriter(xls, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="Itens (cru)")
            xls.seek(0)
            st.download_button(
                "📥 Baixar Excel (tabela crua)",
                data=xls,
                file_name="itens_danfe_cru.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.markdown("---")
st.caption("Observação: este leitor depende do cabeçalho da DANFE para demarcar as colunas. Em layouts muito customizados, posso ajustar rapidamente.")
