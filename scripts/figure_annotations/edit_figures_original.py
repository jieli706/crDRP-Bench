from pathlib import Path
import json, sys, hashlib, math, subprocess, re
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject, DictionaryObject, ArrayObject, ByteStringObject, TextStringObject, FloatObject
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
work = Path.cwd() / '_revision_20261012_REV7_work'
cfg = json.loads((work / 'config.json').read_text(encoding='utf-8'))
out = Path(cfg['output']) / '04_内部核对参考_不上传' / '更新图件_矢量PDF与PNG'
out.mkdir(parents=True, exist_ok=True)
poppler = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
main_backup = Path(cfg['prepared']) / '04_内部核对参考_不上传' / '图件PDF备份'
old_supp = Path(cfg['journal']) / '2610.1返修3' / 'submission_review_20261009_REV5' / 'supp_figures'
jobs = [
    ('main',1,main_backup/'Figure1_REV6.pdf', [('1477','1475')]),
    ('main',3,main_backup/'Figure3_REV6.pdf', [('Spearman ρ with independent reference (95% bootstrap CI)','Spearman r_s with independent reference (95% bootstrap CI)'),('f  Ours (blue) vs established tools (grey), TCGA-BRCA','f  Ours (blue) vs established tools (gray), TCGA-BRCA')]),
    ('main',6,Path(cfg['package'])/'figures_main'/'Figure6.pdf', [('architecture / featurisation','architecture / featurization')]),
    ('si',3,Path(cfg['package'])/'figures_supp'/'FigureS3.pdf', [('fit on 1,477-compound library','fit on 1,475-compound library')]),
    ('si',5,old_supp/'FigureS5.pdf', [('a  Signature-TME pseudo-bulks: RMSE','a  Signature-TME RMSE'),('b  Signature-TME pseudo-bulks: bias','b  Signature-TME bias')]),
    ('si',14,Path(cfg['package'])/'figures_supp'/'FigureS14.pdf', [('GlobalXGB, TME-residualised input','GlobalXGB, TME-residualized input')]),
]
reports=[]

def raw_string(v):
    return ByteStringObject(v.original_bytes) if isinstance(v,TextStringObject) else v

def optext(args, op):
    if op==b'TJ': return ''.join(str(v) for v in args[0] if isinstance(v,TextStringObject))
    if op==b'Tj': return str(args[0])
    return ''

for document, number, source, replacements in jobs:
    reader=PdfReader(source);writer=PdfWriter(clone_from=reader);page=writer.pages[0]
    stream=ContentStream(page.get_contents(),writer);ops=stream.operations
    replacements=dict(replacements);counts={k:0 for k in replacements};edits=[];font=None;font_op=None
    fonts=page['/Resources']['/Font'].get_object()
    fonts[NameObject('/R7Helvetica')]=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica'),NameObject('/Encoding'):NameObject('/WinAnsiEncoding')})
    fonts[NameObject('/R7HelveticaBold')]=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica-Bold'),NameObject('/Encoding'):NameObject('/WinAnsiEncoding')})
    for i,(args,op) in enumerate(ops):
        if op==b'Tf':font=args[0];font_op=i
        if op not in [b'TJ',b'Tj']:continue
        old=optext(args,op);new=replacements.get(old)
        if document=='si' and number==5 and ('grey' in old or 'tumour' in old):
            new=old.replace('grey','gray').replace('tumour','tumor')
        if document=='si' and number==14 and 'Spearman ρ' in old:
            new=old.replace('Spearman ρ','Spearman r_s')
        if new is not None:
            if old in counts:counts[old]+=1
            assert font_op is not None
            base=fonts[font].get_object().get('/BaseFont','')
            try:
                encoded=new.encode('cp1252')
                ops[font_op][0][0]=NameObject('/R7HelveticaBold' if 'Bold' in base else '/R7Helvetica')
            except UnicodeEncodeError:
                # Preserve the original embedded font for labels containing Greek symbols.
                encoded=new.encode('utf-16-be')
            if op==b'TJ':args[0]=ArrayObject([ByteStringObject(encoded)])
            else:args[0]=ByteStringObject(encoded)
            edits.append({'operator':i,'before':old,'after':new})
        elif op==b'TJ':
            args[0]=ArrayObject([raw_string(v) for v in args[0]])
        else:args[0]=raw_string(args[0])
    assert all(n==1 for n in counts.values()), (source,counts)
    geometry=[]
    if document=='main' and number==1:
        # Figure 1f is log-scaled. Correct the final bar as well as its label.
        assert abs(float(ops[3443][0][1])-77.414958)<1e-5
        scale=(71.638495-55.463991)/math.log10(949/275)
        delta=scale*math.log10(1475/1477)
        for ix in [3443,3444]:ops[ix][0][1]=FloatObject(float(ops[ix][0][1])+delta)
        ops[3816][0][5]=FloatObject(float(ops[3816][0][5])+delta)
        geometry.append({'kind':'log-scale bar and annotation','old_count':1477,'new_count':1475,'vertical_delta_pt':delta})
    if document=='main' and number==6:
        # Isolate each complete legend entry in q/cm/Q; curves and uncertainty bands remain byte-equivalent.
        moves=[
          (3217,3238,323-358.30125,336-234.117195,'e, pooled'),
          (3238,3259,323-358.30125,329.8-227.92032,'e, stroma-rich'),
          (3259,3280,323-358.30125,323.6-221.70782,'e, myeloid-rich'),
          (3280,3302,410-358.30125,336-215.52657,'e, bulk'),
          (3302,3327,410-358.30125,329.8-209.31407,'e, BayesPrism'),
          (3327,3348,410-358.30125,323.6-203.117195,'e, oracle'),
          (4040,4064,0,169-54.985625,'f, total'),
          (4064,4096,194-63.01625,169-48.116875,'f, architecture'),
          (4096,4120,350-63.01625,169-36.035625,'f, TME estimation'),
          (4120,4144,63-156.18125,156-54.985625,'f, malignant estimation'),
          (4144,4168,350-156.18125,156-48.03875,'f, bulk loss'),
        ]
        starts={a:(b,x,y,label) for a,b,x,y,label in moves};result=[];i=0
        while i<len(ops):
            if i in starts:
                b,x,y,label=starts[i]
                result.extend([([],b'q'),([FloatObject(1),FloatObject(0),FloatObject(0),FloatObject(1),FloatObject(x),FloatObject(y)],b'cm')])
                result.extend(ops[i:b]);result.append(([],b'Q'))
                geometry.append({'kind':'legend translation','entry':label,'dx_pt':x,'dy_pt':y})
                i=b
            else:result.append(ops[i]);i+=1
        ops=result
    stream.operations=ops;page.replace_contents(stream)
    stem=('Figure' if document=='main' else 'FigureS')+str(number)+'_REV7'
    pdf=out/(stem+'.pdf')
    with pdf.open('wb') as f:writer.write(f)
    raster=out/stem
    subprocess.run([str(poppler),'-singlefile','-r','300','-png',str(pdf),str(raster)],check=True,capture_output=True)
    png=out/(stem+'.png')
    with Image.open(png) as im:im.convert('RGB').save(png,dpi=(300,300))
    verify=PdfReader(pdf).pages[0].extract_text()
    for _,new in replacements.items():assert new in verify,(stem,new)
    report={'document':document,'figure_number':number,'source_pdf':str(source),'pdf':str(pdf),'png':str(png),'text_edits':edits,'geometry_edits':geometry,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'method':'native vector PDF content edit; no raw model analyses rerun'}
    reports.append(report)
    print(stem+': '+str(len(edits))+' text edits; '+str(len(geometry))+' geometry edits')
(work/'figure_edits.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
