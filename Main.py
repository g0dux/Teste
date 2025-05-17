import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import requests
from bs4 import BeautifulSoup
from PIL import Image, ExifTags
import os
import json
import urllib.parse # Para construir URLs de busca

# --- 1. Configurações e Gerenciamento do Modelo LLM ---
MODEL_DIR = "./llm_models"
MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
LLM_INSTANCE = None

def ensure_model_downloaded():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"Modelo {MODEL_NAME} não encontrado. Baixando...")
        try:
            hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_NAME, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
            print("Download completo.")
        except Exception as e:
            print(f"Erro ao baixar o modelo: {e}")
            return None
    return model_path

def load_llm():
    global LLM_INSTANCE
    if LLM_INSTANCE is None:
        model_path = ensure_model_downloaded()
        if model_path:
            try:
                print("Carregando Mistral 7B...")
                LLM_INSTANCE = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
                print("Mistral 7B carregado.")
            except Exception as e:
                print(f"Erro ao carregar o LLM: {e}")
                LLM_INSTANCE = None
        else:
            LLM_INSTANCE = None

# --- 2. Módulos de Automação OSINT ---
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def osint_search_github(username):
    results = {}
    try:
        r = requests.get(f"https://github.com/{username}", headers=HEADERS, timeout=7)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            bio_element = soup.find('div', class_=['p-note', 'user-profile-bio'])
            name_element = soup.find('span', class_='p-name')
            results["GitHub"] = {
                "status": "Perfil encontrado",
                "url": f"https://github.com/{username}",
                "name": name_element.get_text(strip=True) if name_element else "N/A",
                "bio": bio_element.get_text(strip=True) if bio_element else "N/A"
            }
        else:
            results["GitHub"] = {"status": f"Não encontrado ou erro (Status: {r.status_code})", "url": f"https://github.com/{username}"}
    except Exception as e:
        results["GitHub"] = {"status": f"Erro ao buscar: {e}", "url": f"https://github.com/{username}"}
    return results

def osint_search_picuki_instagram(username):
    results = {}
    url = f"https://www.picuki.com/profile/{username}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=7)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            # Essas classes são exemplos, podem precisar de ajuste para Picuki
            profile_info = soup.find('div', class_='profile-info') # Classe hipotética
            error_page_indicator = soup.find('h1', string=lambda t: t and "404" in t.lower() or "not found" in t.lower())

            if error_page_indicator or "Page not found" in r.text or not profile_info:
                 results["Picuki_Instagram"] = {"status": "Perfil não encontrado em Picuki", "url": url}
            else:
                results["Picuki_Instagram"] = {"status": "Potencial perfil encontrado (verificar manualmente)", "url": url}
        else:
            results["Picuki_Instagram"] = {"status": f"Não encontrado ou erro (Status: {r.status_code})", "url": url}
    except Exception as e:
        results["Picuki_Instagram"] = {"status": f"Erro ao buscar: {e}", "url": url}
    return results
    
def osint_search_viewdns_whois(domain):
    results = {}
    url = f"https://viewdns.info/whois/?domain={domain}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            whois_table = soup.find('table', attrs={'border': '1'})
            data = whois_table.prettify() if whois_table else "Dados WHOIS não puderam ser parseados."
            results["ViewDNS_WHOIS"] = {"status": "Informação WHOIS obtida", "domain": domain, "raw_data_preview": data[:1500]}
        else:
            results["ViewDNS_WHOIS"] = {"status": f"Erro ou não encontrado (Status: {r.status_code})", "domain": domain}
    except Exception as e:
        results["ViewDNS_WHOIS"] = {"status": f"Erro ao buscar: {e}", "domain": domain}
    return results

def osint_search_crt_sh(domain):
    results = {}
    url = f"https://crt.sh/?q=%.{domain}&output=json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            certs = r.json()
            if certs:
                summary = [f"{c.get('common_name', 'N/A')} (Issuer: {c.get('issuer_name', 'N/A')}, Valid from: {c.get('not_before', 'N/A')})" for c in certs[:5]]
                results["crt.sh"] = {"status": f"{len(certs)} certificados encontrados", "domain": domain, "summary": summary, "count": len(certs)}
            else:
                results["crt.sh"] = {"status": "Nenhum certificado encontrado", "domain": domain}
        else:
            results["crt.sh"] = {"status": f"Erro (Status: {r.status_code})", "domain": domain}
    except Exception as e:
        results["crt.sh"] = {"status": f"Erro ao buscar: {e}", "domain": domain}
    return results

def osint_search_wayback_machine(target_url_or_domain):
    results = {}
    clean_target = target_url_or_domain.replace("http://", "").replace("https://", "")
    api_url = f"http://archive.org/wayback/available?url={clean_target}"
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("archived_snapshots") and data["archived_snapshots"].get("closest"):
                snapshot = data["archived_snapshots"]["closest"]
                results["WaybackMachine"] = {
                    "status": "Snapshot encontrado",
                    "target": target_url_or_domain,
                    "snapshot_url": snapshot.get("url"),
                    "timestamp": snapshot.get("timestamp"),
                    "available": snapshot.get("available")
                }
            else:
                results["WaybackMachine"] = {"status": "Nenhum snapshot encontrado", "target": target_url_or_domain}
        else:
            results["WaybackMachine"] = {"status": f"Erro (Status: {r.status_code})", "target": target_url_or_domain}
    except Exception as e:
        results["WaybackMachine"] = {"status": f"Erro ao buscar: {e}", "target": target_url_or_domain}
    return results

def osint_search_ahmia(query):
    results = {}
    search_url = f"https://ahmia.fi/search/?q={urllib.parse.quote_plus(query)}"
    results["Ahmia_Search"] = {
        "status": "Link de busca gerado (verificar manualmente)",
        "query": query,
        "search_url": search_url,
        "note": "A automação direta do Ahmia é complexa. Use o link."
    }
    return results
    
def generate_google_dork_links(query, dorks=None):
    if dorks is None:
        dorks = [
            "site:{target} intitle:\"index of\"", "site:{target} inurl:login", "site:{target} filetype:pdf",
            "site:{target} \"{query}\"", "intext:\"{query}\" confidential"
        ]
    links = {}
    target_domain = query if '.' in query and ' ' not in query else None
    for dork_template in dorks:
        dork_query = dork_template
        if target_domain: dork_query = dork_query.replace("{target}", target_domain)
        dork_query = dork_query.replace("{query}", query)
        dork_query = dork_query.replace("{target}", "").replace("{query}", "").strip()
        if not dork_query: continue
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(dork_query)}"
        links[f"GoogleDork: {dork_template.split(':')[0] if ':' in dork_template else dork_template[:20]}"] = url
    return {"Google_Dorks": {"status": "Links de Google Dorks gerados (verificar manualmente)", "query": query, "links": links}}

def analyze_image_locally(image_filepath): # Nome do parâmetro alterado
    results = {}
    try:
        img = Image.open(image_filepath) # Usar o filepath diretamente
        exif_data = {}
        if hasattr(img, '_getexif') and img._getexif() is not None:
            for k, v in img._getexif().items():
                if k in ExifTags.TAGS:
                    exif_data[ExifTags.TAGS[k]] = str(v)
            results["EXIF_Metadata"] = exif_data if exif_data else "Nenhum metadado EXIF encontrado."
        else:
            results["EXIF_Metadata"] = "Nenhum metadado EXIF principal encontrado."
        results["ReverseImageSearch_Links"] = (
            f"Tente buscar esta imagem em:\n"
            f"- Google Images: https://images.google.com/ \n"
            f"- Yandex Images: https://yandex.com/images/ \n"
            f"- TinEye: https://tineye.com/"
        )
    except Exception as e:
        results["Image_Analysis_Error"] = f"Erro ao analisar imagem: {e}"
    return results # Retorna apenas o dicionário de dados

# --- 3. Módulo de Processamento com IA (Mistral 7B) ---
def get_llm_analysis(collected_data_json_str):
    if LLM_INSTANCE is None:
        print("LLM não está carregado. Tentando carregar agora...")
        load_llm()
        if LLM_INSTANCE is None:
            return "Erro: Modelo LLM não está carregado. Verifique o console."

    prompt_template = f"""
Você é um assistente de OSINT profissional e direto. Sua tarefa é analisar os resultados de uma investigação de forma concisa.
Para cada fonte de dados fornecida abaixo, resuma os achados principais.
Destaque as informações que parecem mais relevantes.
Se alguma busca falhou, não retornou resultados significativos ou apenas forneceu um link para verificação manual, mencione isso claramente.
Evite jargões desnecessários. Seja objetivo.

Resultados da Investigação:
{collected_data_json_str}

Sua Análise Profissional:
"""
    try:
        print("Enviando dados para análise do Mistral 7B...")
        output = LLM_INSTANCE(prompt_template, max_tokens=1500, temperature=0.3, stop=["\n\nHuman:", "\n\nResultados da Investigação:"])
        analysis = output['choices'][0]['text'].strip()
        print("Análise recebida do Mistral 7B.")
        return analysis
    except Exception as e:
        print(f"Erro durante a inferência do LLM: {e}")
        return f"Erro ao processar com LLM: {e}"

# --- 4. Lógica da Interface Gradio e Chat ---
def process_chat_message(message, history, current_osint_results_state):
    initial_msg_llm = ""
    if LLM_INSTANCE is None and any(kw in message.lower() for kw in ["buscar", "analisar"]):
        initial_msg_llm = "Iniciando o modelo de linguagem (Mistral 7B)...\n"
        load_llm()
        if LLM_INSTANCE is None: initial_msg_llm += "Falha ao carregar Mistral 7B. Análise da IA indisponível.\n"
        else: initial_msg_llm += "Mistral 7B carregado.\n"
    
    response_text = initial_msg_llm
    new_results_this_turn = {}

    cmd = message.lower().strip()
    parts = cmd.split(maxsplit=2)

    if len(parts) >= 3 and parts[0] == "buscar":
        search_type = parts[1]
        target = parts[2]
        response_text += f"Buscando '{target}' como {search_type}...\n"
        if search_type == "username":
            new_results_this_turn.update(osint_search_github(target))
            new_results_this_turn.update(osint_search_picuki_instagram(target))
        elif search_type == "dominio":
            new_results_this_turn.update(osint_search_viewdns_whois(target))
            new_results_this_turn.update(osint_search_crt_sh(target))
            new_results_this_turn.update(osint_search_wayback_machine(target))
        elif search_type == "onion":
            new_results_this_turn.update(osint_search_ahmia(target))
        elif search_type == "dorks":
            new_results_this_turn.update(generate_google_dork_links(target))
        else:
            response_text += f"Tipo de busca '{search_type}' não reconhecido.\n"
    elif cmd in ["analisar", "analisar tudo", "resumir"]:
        full_data_to_analyze = {**current_osint_results_state, **new_results_this_turn} # Inclui resultados do turno atual
        if not full_data_to_analyze:
            response_text += "Não há dados coletados para analisar.\n"
        else:
            response_text += "Solicitando análise completa dos dados ao Mistral 7B...\n"
            json_str_data = json.dumps(full_data_to_analyze, indent=2, ensure_ascii=False)
            ia_analysis = get_llm_analysis(json_str_data)
            response_text += f"\n**Análise do Assistente IA (Mistral 7B):**\n{ia_analysis}\n"
    elif not cmd: pass
    else:
        response_text += ("Comando não reconhecido. Tente:\n"
                          "- `buscar username [nome]`\n"
                          "- `buscar dominio [dominio.com]`\n"
                          "- `buscar onion [termo]`\n"
                          "- `buscar dorks [termo ou dominio.com]`\n"
                          "- `analisar` (após coletar dados)\n")

    if new_results_this_turn and "analisar" not in cmd:
        response_text += "\n**Resultados Coletados Neste Turno:**\n"
        for source, data_item in new_results_this_turn.items():
            response_text += f"- **{source}**:\n"
            if isinstance(data_item, dict):
                for k, v in data_item.items():
                    v_str = str(v); limit = 200
                    if isinstance(v, list) or isinstance(v,dict) : limit = 500 # Mais espaço para listas/dicionários
                    if len(v_str) > limit and k not in ["url", "search_url", "snapshot_url"]: v_str = v_str[:limit] + "..."
                    response_text += f"  - {k}: {v_str}\n"
            else: response_text += f"  - {str(data_item)[:200]}...\n"
        response_text += "\nDigite 'analisar' para obter um resumo da IA sobre todos os dados coletados.\n"
    
    current_osint_results_state.update(new_results_this_turn)
    history.append((message, response_text if response_text.strip() else "Ok.")) # Evita resposta vazia se só houve msg de carregamento LLM
    return history, current_osint_results_state

def process_image_upload_and_analyze(image_filepath, current_osint_results_state): # Parâmetro é o filepath
    if not image_filepath:
        return "Nenhuma imagem enviada.", current_osint_results_state
    
    initial_msg_llm = ""
    if LLM_INSTANCE is None:
        initial_msg_llm = "Iniciando o modelo de linguagem (Mistral 7B)...\n"
        load_llm()
        if LLM_INSTANCE is None: initial_msg_llm += "Falha ao carregar Mistral 7B. Análise da IA indisponível.\n"
        else: initial_msg_llm += "Mistral 7B carregado.\n"
    
    image_analysis_data = analyze_image_locally(image_filepath) # Retorna só o dicionário de dados
    
    image_file_basename = os.path.basename(image_filepath)
    current_osint_results_state[f"ImageAnalysis_{image_file_basename}"] = image_analysis_data
    
    formatted_results = f"{initial_msg_llm}**Análise da Imagem ({image_file_basename}):**\n"
    for k_main, v_main in image_analysis_data.items():
        formatted_results += f"- **{k_main}**:\n"
        if isinstance(v_main, dict):
            for sub_k, sub_v in v_main.items():
                formatted_results += f"  - {sub_k}: {str(sub_v)}\n"
        else:
            formatted_results += f"  - {str(v_main)}\n"
    formatted_results += "\n"

    response_text = formatted_results + "Solicitando análise da imagem ao Mistral 7B...\n"
    json_str_data = json.dumps(image_analysis_data, indent=2, ensure_ascii=False)
    ia_analysis = get_llm_analysis(json_str_data)
    response_text += f"\n**Análise do Assistente IA (Mistral 7B) sobre a Imagem:**\n{ia_analysis}\n"
    
    return response_text, current_osint_results_state

with gr.Blocks(title="OSINT IA Investigator v2.1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️ OSINT IA Investigator v2.1 (Mistral 7B Local)")
    gr.Markdown("Use o chat para comandos OSINT (ex: `buscar username nome`) ou envie uma imagem para análise EXIF. "
                "Após coletar dados, digite `analisar` no chat para um resumo da IA.")

    session_state_osint_results = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            # CORREÇÃO AQUI: Adicionado type="messages" e removido bubble_full_width
            chatbot_component = gr.Chatbot(height=600, label="Chat OSINT", type="messages", avatar_images=("user.png", "bot.png"))
            chat_input_msg = gr.Textbox(label="Digite seu comando:", placeholder="Ex: buscar username nomedeusuario")
            
            chat_input_msg.submit(
                fn=process_chat_message,
                inputs=[chat_input_msg, chatbot_component, session_state_osint_results],
                outputs=[chatbot_component, session_state_osint_results]
            ).then(lambda: "", inputs=[], outputs=[chat_input_msg])

        with gr.Column(scale=1):
            gr.Markdown("### Análise de Imagem (EXIF Local)")
            # CORREÇÃO AQUI: type="filepath"
            image_upload_input = gr.Image(type="filepath", label="Envie uma imagem")
            analyze_image_btn = gr.Button("Analisar Imagem Enviada e Resumir com IA", variant="secondary")
            image_analysis_output_markdown = gr.Markdown(label="Resultado da Análise da Imagem")

            analyze_image_btn.click(
                fn=process_image_upload_and_analyze, # Nome da função corrigido
                inputs=[image_upload_input, session_state_osint_results],
                outputs=[image_analysis_output_markdown, session_state_osint_results]
            )
    
    gr.Markdown("---")
    gr.Markdown(f"Modelo LLM: {MODEL_REPO}/{MODEL_NAME}. O download/carregamento inicial pode demorar. Verifique o console.")

if __name__ == "__main__":
    print("Bem-vindo ao OSINT IA Investigator v2.1. Preparando...")
    print("Interface Gradio pronta. Acesse no seu navegador.")
    demo.launch(share=False, server_port=7860)
