"""Fetch verified official PDD weights and the matching pruned time basis."""
import hashlib
import json
from pathlib import Path
import shutil
from infer.download_utils import curl_download, direct_urlopen
import urllib.request

NAME='MiniMax-H3-FL2VA-Acc-8Step.safetensors'
REVISION='335001fb9e5455d68a0caa18ec2e319072150328'
SIZE=1372450680
SHA='0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196'
BASIS_SHA='941891e20f36c5f7c244901c5792424e24023431371773def8d49ea6501e4889'


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(16*1024**2),b''):
            digest.update(part)
    return digest.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    (root/'weights').mkdir(parents=True, exist_ok=True)
    (root/'metadata').mkdir(parents=True, exist_ok=True)
    target=root/'weights/pdd'/NAME
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path('/tmp/minimax-h3-downloads')/NAME
    stage.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size != SIZE:
        raise ValueError('Installed checkpoint has wrong size; refusing to overwrite')
    path=target if target.exists() else stage
    url=f'https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/resolve/{REVISION}/{NAME}'
    if not path.exists() or path.stat().st_size!=SIZE:
        print(json.dumps({'stage':'pdd_download','expected_bytes':SIZE}),flush=True)
        curl_download(['--fail','--location','--silent','--show-error','--retry','5',
                        '--continue-at','-','--output',str(path),url])
    if path.stat().st_size!=SIZE or sha256(path)!=SHA:
        raise ValueError('PDD checkpoint failed size/SHA256 verification')
    if path!=target:
        shutil.move(str(path),str(target))
    basis_source='https://raw.githubusercontent.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc/main/adaln_basis/basis_fl2va.safetensors'
    basis=root/'weights/pdd/basis_fl2va.safetensors'
    temp_basis=Path('/tmp/basis_fl2va.safetensors')
    if not basis.exists():
        if not temp_basis.exists() or sha256(temp_basis)!=BASIS_SHA:
            temp_basis.write_bytes(direct_urlopen(basis_source,timeout=30).read())
        if sha256(temp_basis)!=BASIS_SHA:
            raise ValueError('PDD basis SHA256 mismatch')
        shutil.copyfile(temp_basis,basis)
    if sha256(basis)!=BASIS_SHA:
        raise ValueError('Installed PDD basis SHA256 mismatch')
    report={'checkpoint':str(target),'repository':'alibaba-pai/MiniMax-H3-Acc-LoRAs','revision':REVISION,
            'checkpoint_sha256':SHA,'size':SIZE,'basis':str(basis),'basis_sha256':BASIS_SHA,
            'basis_source':basis_source,'verified':True}
    (root/'metadata/pdd_weights.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage':'pdd_weights_verified',**report}),flush=True)


if __name__=='__main__':
    main()
