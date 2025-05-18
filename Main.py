import sys
# Patch para contornar a ausência do módulo 'distutils'
try:
    import distutils
except ImportError:
    try:
        import setuptools._distutils as distutils
        sys.modules['distutils'] = distutils
    except ImportError:
        pass

import os
import time
import re
import logging
import requests
import io
import psutil
import threading
import concurrent.futures
import importlib.util
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
from PIL import Image, ExifTags
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import nltk
import socket
import gradio as gr

# opcional: import pyshark se instalado
try:
    import pyshark
except ImportError:
    pyshark = None
    logging.warning("pyshark não está instalado. Análise de rede indisponível.")

# ==================== CONFIGURAÇÕES GERAIS ====================
nltk.download('punkt')
nltk.download('vader_lexicon')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sentiment_analyzer = SentimentIntensityAnalyzer()

LANGUAGE_MAP = {
    'Português': {'code': 'pt-BR', 'instruction': 'Responda em português brasileiro'},
    'English':   {'code': 'en-US', 'instruction': 'Respond in English'},
    'Español':   {'code': 'es-ES', 'instruction': 'Responde en español'},
    'Français':  {'code': 'fr-FR', 'instruction': 'Réponds en français'},
    'Deutsch':   {'code': 'de-DE', 'instruction': 'Antworte auf Deutsch'}
}

DEFAULT_MODEL_NAME      = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
DEFAULT_MODEL_FILE      = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DEFAULT_LOCAL_MODEL_DIR = "models"

# ==================== SITES DE INVESTIGAÇÃO ====================
INVESTIGATION_SITES = {
    "e-sus Território":    "https://territorio.datasus.gov.br/?q={query}",
    "Wigle Net":           "https://wigle.net/search?query={query}",
    "FotoForensics":       "https://fotoforensics.com/search.php?url={query}",
    "PeekYou":             "https://www.peekyou.com/?q={query}",
    "TruePeopleSearch":    "https://www.truepeoplesearch.com/results?name={query}",
    "That'sThem":          "https://thatsthem.com/search?term={query}",
    "Webmii":              "https://webmii.com/people?q={query}",
    "SocialSearch":        "https://social-searcher.com/social-profile?keyword={query}",
    "Lullar":              "https://lullar.com/search/{query}",
    "Whois (DomainTools)": "https://whois.domaintools.com/{query}",
    "ViewDNS.info":        "https://viewdns.info/reversewhois/?q={query}",
    "DNSDumpster":         "https://dnsdumpster.com/static/map?search={query}",
    "crt.sh":              "https://crt.sh/?q={query}",
    "Talos Intelligence":  "https://talosintelligence.com/reputation_center/lookup?search={query}",
    "Namecheckr":          "https://namecheckr.com/{query}",
    "WhatsMyName":         "https://whatsmyname.app/profile/{query}",
    "Instagram Viewer":    "https://www.picuki.com/profile/{query}",
    "Twitter Advanced":    "https://twitter.com/search?q={query}",
    "RedditSearch.io":     "https://redditsearch.io/?q={query}",
    "Snap Map":            "https://snapmap.snapchat.com/search?query={query}",
    "TikTok Search":       "https://www.tiktok.com/search?q={query}",
    "IPlocation.net":      "https://www.iplocation.net/search?query={query}",
    "GeoIPTool":           "https://geoiptool.com/?ip={query}",
    "BGPlay (RIPE NCC)":   "https://bgplay.nic.ad.jp/cgi-bin/bgplay?prefix={query}",
    "LeakCheck":           "https://leakcheck.net/search?query={query}",
    "Ahmia":               "https://ahmia.fi/search/?q={query}",
    "Onion.live":          "https://onion.live/{query}",
    "Google Dorks":        "https://www.google.com/search?q={query}",
    "Metadata2Go":         "https://www.metadata2go.com/?url={query}",
    "FOIA.gov":            "https://www.foia.gov/search?query={query}",
    "Internet Archive":    "https://archive.org/search.php?query={query}",
    "PDF Examiner":        "https://www.pdfexaminer.com/examine?url={query}",
    "Google Reverse Image":"https://images.google.com/searchbyimage?image_url={query}",
    "Yandex Images":       "https://yandex.com/images/search?rpt=imageview&url={query}",
    "TinEye":              "https://tineye.com/search?url={query}",
    "Exif.tools":          "https://exif.tools/?url={query}",
    "InVID Verification":  "https://www.invid-project.eu/tools-and-services/invid-verification-plugin/?url={query}",
    "TorLinks":            "https://torlinkbgs6aabns.onion.link/search?q={query}"
}

def build_site_links(query: str) -> str:
    html  = "<h3>Sites Específicos</h3>"
    html += "<table border='1' style='width:100%; border-collapse: collapse; text-align: left;'>"
    html += "<thead><tr><th>Site</th><th>Link</th></tr></thead><tbody>"
    for name, tpl in INVESTIGATION_SITES.items():
        url = tpl.format(query=query)
        html += f"<tr><td>{name}</td><td><a href='{url}' target='_blank'>{url}</a></td></tr>"
    html += "</tbody></table><br>"
    return html

# ==================== FLASK APP ====================
app = Flask(__name__)

# Métricas Prometheus
REQUEST_COUNT   = Counter('flask_request_count', 'Total de requisições', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('flask_request_latency_seconds', 'Tempo de resposta', ['endpoint'])

@app.before_request
def before_request():
    request.start_time = time.time()
    REQUEST_COUNT.labels(request.path, request.method).inc()

@app.after_request
def after_request(response):
    REQUEST_LATENCY.labels(request.path).observe(time.time() - request.start_time)
    return response

@app.route('/')
def index():
    # agora busca em templates/index.html
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

@app.route('/ask', methods=['POST'])
def ask():
    mode = request.form.get('mode', 'Chat')
    user_input = request.form.get('user_input', '').strip()

    if mode == "Investigação":
        # 1) Autocorreção e pré-processamento (omitido aqui)
        query = user_input

        # 2) Gera links dos sites estáticos
        links_sites = build_site_links(query)

        # 3) Pesquisa web básica
        sites_meta = int(request.form.get('sites_meta', 5))
        results_web = perform_search(query, 'web', sites_meta)
        _, links_web, _ = format_search_results(results_web, "Resultados Web")

        # 4) Geração de relatório via modelo
        report, _ = process_investigation(
            target=query,
            sites_meta=sites_meta,
            investigation_focus=request.form.get('investigation_focus',''),
            search_news=request.form.get('search_news','false')=='true',
            search_leaked_data=request.form.get('search_leaked_data','false')=='true',
            custom_temperature=None,
            lang='Português',
            fast_mode=False
        )

        # 5) Combina tudo e retorna
        final_links = links_sites + links_web
        return jsonify({'response': report, 'links': final_links})

    # modos Chat e Metadados seguem seu fluxo normal…
    return jsonify({'response': 'Modo não suportado.'})

# ==================== FUNÇÕES AUXILIARES ====================
def perform_search(query: str, search_type: str, max_results: int) -> list:
    with DDGS() as ddgs:
        if search_type == 'web':
            return list(ddgs.text(keywords=query, max_results=max_results))
        elif search_type == 'news':
            return list(ddgs.news(keywords=query, max_results=max_results))
        elif search_type == 'leaked':
            return list(ddgs.text(keywords=f"{query} leaked", max_results=max_results))
    return []

def format_search_results(results: list, title: str) -> tuple[str, str, str]:
    links_table = f"<h3>{title}</h3><table border='1' style='width:100%; border-collapse: collapse;'>"
    links_table += "<thead><tr><th>#</th><th>Título</th><th>Link</th></tr></thead><tbody>"
    for i, r in enumerate(results, 1):
        href = r.get('href','')
        title = r.get('title','')
        links_table += f"<tr><td>{i}</td><td>{title}</td><td><a href='{href}' target='_blank'>{href}</a></td></tr>"
    links_table += "</tbody></table>"
    return "", links_table, ""

def process_investigation(target: str, sites_meta: int, investigation_focus: str,
                          search_news: bool, search_leaked_data: bool,
                          custom_temperature, lang, fast_mode) -> tuple[str, str]:
    """
    Aqui entra todo o seu fluxo de autocorreção, buscas paralelas (web/news/leaked),
    análise forense (regex), montagem do prompt e chamada ao modelo Llama.
    Para simplificar, retorna um relatório dummy:
    """
    report = f"Relatório detalhado para '{target}' com foco em '{investigation_focus}'."
    return report, ""

# ==================== MODELO LLM ====================
model_lock = threading.Lock()

def load_model(custom_gpu_layers=None, custom_n_batch=None) -> Llama:
    model_path = os.path.join(DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_FILE)
    if not os.path.exists(model_path):
        hf_hub_download(repo_id=DEFAULT_MODEL_NAME, filename=DEFAULT_MODEL_FILE, local_dir=DEFAULT_LOCAL_MODEL_DIR, resume_download=True)
    # detecção de GPU e configuração de n_gpu_layers, n_batch...
    # (mesmo código que você já tem)
    return Llama(model_path=model_path, n_ctx=4096, n_threads=psutil.cpu_count(), n_gpu_layers=-1, n_batch=1024)

model = load_model()

# ==================== GRADIO (opcional) ====================
def gradio_interface(...):
    # sua função de interface Gradio, sem alterações
    ...

# ==================== EXECUÇÃO ====================
if __name__ == '__main__':
    # iniciar Gradio em thread, se desejar...
    app.run(host='0.0.0.0', port=5000, debug=True)
