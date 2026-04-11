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
    allow_origins=["*"],  # em produção muda para o domínio do cliente
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── COMPANY CONTEXT ───────────────────────────────────────────────────────────
# Substitui isto para cada cliente
COMPANY_CONTEXT = """

És o assistente virtual da ATJF Imobiliária, agência imobiliária sediada no Porto.

SOBRE A EMPRESA:
- Nome: ATJF Imobiliária
- Morada: Rua das Mestras, 818, Loja J, 4415-389, Porto
- Zona de actuação: Porto, Vila Nova de Gaia e todo o território nacional
- Contacto: contacto.atjf@gmail.com | 937 649 190
- Website: imobiliarianoporto.pt
- Atendimento em: Português, Inglês e Francês

SERVIÇOS:
- Compra e venda de imóveis (residencial e comercial)
- Arrendamento de imóveis (estúdios, T1, T2, moradias)
- Gestão de condomínios (administração financeira, manutenção, assembleias)
- Gestão locativa (seleção de inquilinos, contratos, cobranças, reparações)
- Avaliação de imóveis (estimativa de mercado para venda, arrendamento ou investimento)
- Apoio a clientes estrangeiros e não lusófonos

PROCESSO DE COMPRA/VENDA:
1. Cliente descreve o imóvel ou o que procura
2. Assistente recolhe informação e agenda contacto com a equipa
3. Equipa faz avaliação, promoção, gestão de visitas e acompanhamento até escritura

PROCESSO DE ARRENDAMENTO:
1. Cliente descreve o que procura (zona, tipologia, orçamento)
2. Assistente apresenta opções disponíveis e agenda visita
3. Equipa trata de contrato, caução e acompanhamento

REGRAS:
- Responde SEMPRE em português europeu (ou no idioma do cliente se falar inglês ou francês)
- Sê simpático, profissional e conciso
- Nunca inventes valores exactos de imóveis — diz sempre que o valor depende de avaliação
- Para pedidos de avaliação, recolhe: zona, tipologia, área aproximada e estado do imóvel
- No final de cada conversa de interesse, pede: nome, email e telefone
- Se a pergunta não for sobre imobiliário, redireciona educadamente

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
    "projecto": "resumo do projecto se discutido ou null"
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
    return {"status": "ok", "service": "ChatBot API"}
