"""Downloads bypass environment HTTP proxies unless H3_USE_PROXY=1."""
import os
import subprocess
import urllib.request


def curl_download(arguments):
    proxy_flags = [] if os.environ.get('H3_USE_PROXY') == '1' else ['--noproxy', '*']
    subprocess.run(['curl', *proxy_flags, *arguments], check=True)


def direct_urlopen(url, timeout=30):
    handler = (urllib.request.ProxyHandler() if os.environ.get('H3_USE_PROXY') == '1'
               else urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(handler).open(url, timeout=timeout)
