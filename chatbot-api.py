import os
import json
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import anthropic
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ── ENV ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set")

GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEET_ID         = os.environ.get("GOOGLE_SHEET_ID")

# ── APP ──────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ChatBot API", docs_url=None, redoc_url=None)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://imobiliarianoporto.pt", "http://localhost", "http://localhost:8080", "http://127.0.0.1"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── COMPANY CONTEXT ───────────────────────────────────────────────────────────
COMPANY_CONTEXT = """
És o assistente virtual da ATJF Imobiliária, agência imobiliária sediada no Porto, Portugal.

SOBRE A EMPRESA:
- Nome: ATJF Imobiliária
- Morada: R. de Pinto Bessa n.º 522, 4300-428 Porto
- Zona de actuação: Porto, Vila Nova de Gaia e todo o território nacional
- Telefone: +351 937 649 190
- Website: imobiliarianoporto.pt
- Serviços: Compra, Venda, Arrendamento, Gestão de Condomínios, Avaliação de Imóveis
- Anos de experiência: Vários anos de reputação sólida no mercado
- Atendimento: Português, Francês e Inglês

SERVIÇOS DETALHADOS:
1. COMPRA E VENDA — Acompanhamento completo: avaliação, promoção, visitas, negociação e escritura
2. ARRENDAMENTO — Selecção de inquilinos, contratos, gestão mensal e acompanhamento
3. GESTÃO DE CONDOMÍNIOS — Administração diária, manutenção, contabilidade e comunicação
4. AVALIAÇÃO DE IMÓVEIS — Estimativa de mercado para venda, arrendamento ou investimento

ZONAS DE ACTUAÇÃO NO PORTO:
Campanhã, Bonfim, Cedofeita, Massarelos, Paranhos, Sé, São Nicolau, Vila Nova de Gaia

PROCESSO TÍPICO:
1. Cliente descreve o que procura ou o imóvel que tem
2. Assistente recolhe informações relevantes (tipo, zona, orçamento, objectivo)
3. Equipa contacta em breve para acompanhamento personalizado

REGRAS DE COMPORTAMENTO:
- Responde no idioma do utilizador: se escrever em português, responde em português europeu; se escrever em francês, responde em francês; se escrever em inglês, responde em inglês
- Sê simpático, profissional e conciso
- Foca sempre em perceber o objectivo do cliente (comprar, vender, arrendar, gerir)
- Pede sempre o nome, email e telefone antes de terminar a conversa
- NUNCA inventes preços exactos de imóveis — diz que a equipa fará uma avaliação personalizada
- Se a pergunta não for sobre imobiliária, redireciona educadamente para os serviços da ATJF

DETECÇÃO DE LEAD:
Quando o utilizador fornecer nome + email ou telefone, inclui no teu JSON o campo "lead" com os dados.
"""

SYSTEM_PROMPT = COMPANY_CONTEXT + """

Responde SEMPRE em JSON com este formato exacto:
{
  "message": "a tua resposta ao utilizador",
  "lead": {
    "nome": "nome se fornecido ou null",
    "email": "email se fornecido ou null",
    "telefone": "telefone se fornecido ou null",
    "projecto": "resumo do interesse do cliente (ex: comprar T2 no Bonfim, orçamento 200k) ou null"
  } ou null se não houver dados de contacto ainda
}
"""

# ── GOOGLE SHEETS ─────────────────────────────────────────────────────────────
def save_lead_to_sheets(lead: dict):
    if not GOOGLE_CREDENTIALS_JSON or not GOOGLE_SHEET_ID:
        return
    try:
        creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GOOGLE_SHEET_ID)
        ws = sh.sheet1
        ws.append_row([
            datetime.now().strftime("%d/%m/%Y %H:%M"),
            lead.get("nome") or "",
            lead.get("email") or "",
            lead.get("telefone") or "",
            lead.get("projecto") or "",
        ])
    except Exception as e:
        print(f"Sheets error: {e}")

# ── ENDPOINT ──────────────────────────────────────────────────────────────────
@app.post("/chat")
@limiter.limit("30/hour")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Limita histórico a últimas 10 mensagens
    messages = messages[-10:]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI service unavailable")

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"message": raw, "lead": None}

    # Guarda lead no Google Sheets se tiver dados suficientes
    lead = parsed.get("lead")
    if lead and (lead.get("email") or lead.get("telefone")):
        save_lead_to_sheets(lead)

    return JSONResponse({
        "message": parsed.get("message", ""),
        "lead_captured": bool(lead and (lead.get("email") or lead.get("telefone")))
    })

@app.get("/health")
def health():
    return {"status": "ok", "service": "ChatBot API - ATJF Imobiliária"}
