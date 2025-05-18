#!/usr/bin/env python3
# chatcyber_sherlock.py

import os
import sys
import shutil
import stat
import zipfile
import platform
import subprocess
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.align import Align
from rich import box
from pyfiglet import Figlet
from stem.process import launch_tor_with_config

console = Console()
TIMEOUT = 7
TOR_PORT = 9050
TOR_CONTROL_PORT = 9051

# --- TODAS AS FERRAMENTAS SOLICITADAS ---
SITES = {
    "e-SUS Território":         "https://www.e-sus-territorio.gov.br/usuarios/{}",
    "Wiggle.net":               "https://wiggle.net/{}",
    "FotoForensics":            "https://fotoforensics.com/analysis.php?image={}",
    "PeekYou":                  "https://www.peekyou.com/{}",
    "TruePeopleSearch":         "https://www.truepeoplesearch.com/results?name={}",
    "That'sThem":               "https://thatsthem.com/name/{}",
    "Webmii":                   "https://webmii.com/{}",
    "SocialSearch":             "https://www.social-searcher.com/social/{}",
    "Lullar":                   "https://www.lullar.com/#!/search/{}",
    "Hunter.io":                "https://hunter.io/search/{}",
    "HaveIBeenPwned":           "https://haveibeenpwned.com/unifiedsearch/{}",
    "DomainTools Whois":        "https://whois.domaintools.com/{}",
    "ViewDNS.info":             "https://viewdns.info/reversewhois/?q={}",
    "DNSDumpster":              "https://dnsdumpster.com/static/map.html?domain={}",
    "crt.sh":                   "https://crt.sh/?q={}",
    "Shodan":                   "https://www.shodan.io/host/{}",
    "Censys":                   "https://censys.io/ipv4/{}",
    "VirusTotal":               "https://www.virustotal.com/gui/ip-address/{}/detection",
    "Talos Intelligence":       "https://talosintelligence.com/reputation_center/lookup?search={}",
    "Namecheckr":               "https://namechk.com/{}",
    "WhatsMyName":              "https://whatsmyname.app/{}",
    "Instagram (Picuki)":       "https://api.picuki.com/v1/profile/{}",
    "Twitter Advanced Search":  "https://twitter.com/search?q={}&f=live",
    "Facebook Graph Search":    "https://www.facebook.com/public/{}",
    "RedditSearch.io":          "https://redditsearch.io/?q={}",
    "LinkedIn Lookup":          "https://www.linkedin.com/in/{}",
    "Snap Map":                 "https://www.snapmap.io/{}",
    "TikTok Search":            "https://www.tiktok.com/@{}",
    "IPinfo.io":                "https://ipinfo.io/{}",
    "IPlocation.net":           "https://www.iplocation.net/ip-lookup/{}",
    "AbuseIPDB":                "https://www.abuseipdb.com/check/{}",
    "GeoIPTool":                "https://geoiptool.com/en/?ip={}",
    "BGPlay (RIPE NCC)":        "https://stat.ripe.net/data/bg-play/data.json?resource={}",
    "DeHashed":                 "https://dehashed.com/search?query={}",
    "IntelligenceX":            "https://intelx.io/?order=d&q={}",
    "LeakCheck":                "https://leakcheck.net/search?q={}",
    "Ahmia":                    "https://ahmia.fi/search/?q={}",
    "Onion.live":               "https://onion.live/search?q={}",
    "Exploit-DB (Google Dorks)": "https://www.exploit-db.com/google-hacking-database?q={}",
    "Metadata2Go":              "https://www.metadata2go.com/api/analyze?url={}",
    "FOIA.gov":                 "https://www.foia.gov/search/site/{}",
    "Wayback Machine":          "https://web.archive.org/web/*/{}",
    "PDF Examiner":             "https://www.pdfexaminer.com/?file={}",
    "Google Reverse Image":     "https://www.google.com/searchbyimage?image_url={}",
    "Yandex Images":            "https://yandex.com/images/search?url={}",
    "TinEye":                   "https://tineye.com/search?url={}",
    "Exif.tools":               "https://exif.tools/analyze?url={}",
    "InVID Verification":       "https://verify.invid-project.eu/?url={}",
    "TorLinks (.onion)":        "http://torlinkbgs6aabns.onion/search?query={}"
}

def ensure_tor():
    """Baixa e extrai o Tor Expert Bundle se não estiver no PATH."""
    exe = "tor.exe" if platform.system()=="Windows" else "tor"
    path = shutil.which(exe)
    if path:
        return path

    console.log("[yellow]Tor não encontrado. Baixando Expert Bundle…[/]")
    # URLs oficiais (atualize versões se necessário)
    if platform.system()=="Windows":
        url = "https://www.torproject.org/dist/torbrowser/12.0.6/tor-win64-0.4.9.14.zip"
        ext = ".zip"
    else:
        url = "https://www.torproject.org/dist/tor/0.4.9.14/tor-0.4.9.14-linux-x86_64-en-US.tar.gz"
        ext = ".tar.gz"

    dl = os.path.join(tempfile.gettempdir(), "tor"+ext)
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length",0))
    with open(dl, "wb") as f, Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
        task = prog.add_task("Baixando Tor...", total=total)
        for chunk in r.iter_content(1024*64):
            f.write(chunk); prog.advance(task, len(chunk))

    console.log("[green]Download concluído. Extraindo…[/]")
    extract_dir = os.path.join(tempfile.gettempdir(), "tor_expert")
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)

    if ext==".zip":
        with zipfile.ZipFile(dl,'r') as z: z.extractall(extract_dir)
    else:
        import tarfile
        with tarfile.open(dl,'r:gz') as t: t.extractall(extract_dir)

    for root,_,files in os.walk(extract_dir):
        if exe in files:
            bin_path = os.path.join(root,exe)
            os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IEXEC)
            console.log(f"[green]Tor extraído em {bin_path}[/]")
            return bin_path

    console.log("[red]Erro: não localizei o binário tor[/]")
    sys.exit(1)

def launch_tor():
    """Inicia o Tor em background via Stem."""
    tor_path = ensure_tor()
    console.log("[cyan]Iniciando Tor na porta 9050…[/]")
    return launch_tor_with_config(
        config={
            "SocksPort": str(TOR_PORT),
            "ControlPort": str(TOR_CONTROL_PORT)
        },
        tor_cmd=tor_path,
        init_msg_handler=lambda line: console.log(f"[blue]{line}[/]") if "Bootstrapped" in line else None
    )

def get_session_for(url):
    sess = requests.Session()
    if ".onion" in url:
        sess.proxies.update({
            "http":  f"socks5h://127.0.0.1:{TOR_PORT}",
            "https": f"socks5h://127.0.0.1:{TOR_PORT}"
        })
    return sess

def check_site(name, pattern, target):
    url = pattern.format(target)
    sess = get_session_for(url)
    try:
        r = sess.get(url, timeout=TIMEOUT, headers={"User-Agent":"Mozilla/5.0"})
        found = (r.status_code==200 and len(r.text)>500)
        return name, url, found
    except Exception:
        return name, url, False

def print_header():
    fig = Figlet(font="slant")
    art = fig.renderText("CHATCYBER CSI")
    console.print(Panel.fit(art, style="bold magenta", subtitle="OSINT Sherlocked"))

def main():
    if len(sys.argv)!=2:
        console.print("Uso: python chatcyber_sherlock.py <alvo>", style="red")
        sys.exit(1)
    target = sys.argv[1]
    print_header()
    tor_proc = launch_tor()
    console.print(f"[bold green]Verificando “[yellow]{target}[/]” em {len(SITES)} sites[/]\n")

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Status", style="bold")
    table.add_column("Site")
    table.add_column("URL", overflow="fold")

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = { pool.submit(check_site, n, p, target): n for n,p in SITES.items() }
        for fut in as_completed(futures):
            name, url, ok = fut.result()
            status = "[green]✔[/]" if ok else "[red]✖[/]"
            table.add_row(status, name, url)
            console.clear()
            console.print(Panel.fit("Resultados em tempo real", style="bold blue"))
            console.print(table)

    console.print("\n[bold blue]✓ Busca concluída![/]")
    console.log("[cyan]Parando Tor…[/]")
    tor_proc.kill()

if __name__=="__main__":
    main()