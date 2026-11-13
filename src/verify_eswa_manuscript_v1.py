"""Compile and render complete article remotely; verify tables/claims and artifacts."""
import os, re, csv, json, subprocess, hashlib, datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
assert os.name!='nt'
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'paper/eswa_selective_rescoring_v1';G=P/'generated'
R=ROOT/'tmp/pdfs/eswa_selective_rescoring_v1';R.mkdir(parents=True,exist_ok=True)
tex=(P/'manuscript.tex').read_text();bib=(P/'references.bib').read_text()
ledger=list(csv.DictReader((G/'manuscript_evidence.csv').open()));values={r['key']:r['display'] for r in ledger}
keys=set(re.findall(r'\\stat\{([^}]+)\}',tex));assert keys<=values.keys(),sorted(keys-values.keys())
cites={k.strip() for c in re.findall(r'\\cite[pt]?\{([^}]+)\}',tex) for k in c.split(',')}
bkeys=set(re.findall(r'@\w+\{([^,]+),',bib));assert cites==bkeys,(cites-bkeys,bkeys-cites)
resolved=re.sub(r'\\stat\{([^}]+)\}',lambda m:values[m[1]],tex)
abstract=re.search(r'\\begin\{abstract\}([\s\S]*?)\\end\{abstract\}',resolved)[1]
abstract_words=len(re.findall(r"[\w]+(?:[-'][\w]+)*",abstract))
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
base=read(ROOT/'results/eswa_gate_stage2_v1/analysis/main_results.csv')
ext=read(ROOT/'results/eswa_extension_20261111_v1/analysis/main_results.csv')
def serial(r):return next(x for x in base if x['model']==r['model'] and x['dataset']==r['dataset'] and x['method']=='serial')
assert all(int(r['context_differences'])==0 and int(r['epsilon_exceeded_candidates'])==0 for r in ext if r['method']=='band')
def quality_equal(r):return all(abs(float(r[f])-float(serial(r)[f]))<1e-12 for f in ['gold_recall','answer_f1'])
assert all(quality_equal(r) for r in ext if r['method']=='band')
gate_unequal=[{k:r[k] for k in ['model','dataset','batch_size','answer_f1','gold_recall']} for r in ext if r['method']=='gate_v2' and not quality_equal(r)]
assert len(gate_unequal)==1 and gate_unequal[0]['model']=='qwen' and gate_unequal[0]['dataset']=='hotpot' and gate_unequal[0]['batch_size']=='32',gate_unequal
assert all(int(r['repeat_context_changes'])==0 and int(r['repeat_score_changes'])==0 for r in ext)
for r in ext:
    if r['method'] not in ['top15','band']:continue
    other=next((x for x in ext if x['model']==r['model'] and x['dataset']==r['dataset'] and x['batch_size']==r['batch_size'] and x['method']=='gate_v2'),None)
    if other:assert float(other['mean_rescored_fraction'])<float(r['mean_rescored_fraction'])
run=subprocess.run([str(ROOT/'tmp/ipm_writing_tools_v1/tectonic'),'manuscript.tex','--keep-logs','--keep-intermediates'],cwd=P,capture_output=True,text=True,timeout=240)
(R/'compile_stdout.txt').write_text(run.stdout+'\n'+run.stderr)
assert run.returncode==0,run.stdout+run.stderr
log=(P/'manuscript.log').read_text();assert not re.search(r'undefined|Missing character',log,re.I),log[-3000:]
pdf=P/'manuscript.pdf'
for cmd in [['pdftotext','-layout',str(pdf),str(R/'text.txt')],['pdftotext','-bbox-layout',str(pdf),str(R/'bbox.html')],['pdftoppm','-scale-to','1400','-png',str(pdf),str(R/'page')]]:subprocess.run(cmd,check=True,capture_output=True)
pages=ET.parse(R/'bbox.html').findall('.//{http://www.w3.org/1999/xhtml}page');outside=[]
for i,page in enumerate(pages,1):
    w,h=float(page.attrib['width']),float(page.attrib['height'])
    for word in page.iter('{http://www.w3.org/1999/xhtml}word'):
        a=word.attrib
        if float(a['xMin'])<0 or float(a['yMin'])<0 or float(a['xMax'])>w+.5 or float(a['yMax'])>h+.5:outside.append({'page':i,'text':word.text})
assert not outside,outside
images=sorted(R.glob('page-*.png'));assert len(images)==len(pages)
for offset in range(0,len(images),4):
    sheet=Image.new('RGB',(1200,1760),'#d8dce0');draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(images[offset:offset+4]):
        im=Image.open(path).convert('RGB');im.thumbnail((580,830));x=(j%2)*600+(600-im.width)//2;y=(j//2)*880+30
        sheet.paste(im,(x,y));draw.text((x,y-22),f'Page {offset+j+1}',fill='black')
    sheet.save(R/f'contact_{offset//4+1:02}.png',optimize=True)
qa={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'compiled_and_numerically_checked_visual_review_pending','pages':len(pages),'abstract_words':abstract_words,'citation_keys':len(cites),'numeric_macros_used':len(keys),'evidence_ledger_rows':len(ledger),'unresolved_citations':False,'out_of_page_words':outside,'overfull_lines':re.findall(r'Overfull[^\n]+',log),'gate_quality_exceptions':gate_unequal,'human_oversight_placeholder_intentionally_retained':True,'journal_specific_format_verification':'official guide HTTP403; not certified','pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
(P/'quality_checks.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa))
