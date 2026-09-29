# -*- coding: utf-8 -*-
"""Non-destructive continuation intake. Does not redownload any existing asset."""
import csv,hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs'/('continuation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));OUT.mkdir()
needs=[('DepMap Public 26Q1','Model.csv','depmap'),('DepMap Public 26Q1','OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv','depmap'),('Harmonized GDSC 25Q2','GDSC2AUCMatrix.csv','gdsc2'),('Harmonized GDSC 25Q2','GDSC2Log2IC50Matrix.csv','gdsc2'),('Harmonized GDSC 25Q2','GDSC2ResponseCurves.csv','gdsc2'),('Sanger GDSC1 and GDSC2','sanger-dose-response.csv','gdsc2')]
rows=[]
for release,name,folder in needs:
    dest=ROOT/'data/raw'/folder/name;dest.parent.mkdir(parents=True,exist_ok=True)
    matches=list(ROOT.rglob(name))
    if name=='OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv':matches+=list(ROOT.rglob('OmicsExpressionProteinCodingGenesTPMLogp1.csv'))
    rows.append({'release':release,'filename':name,'save_to':str(dest),'local_matches':[str(p) for p in matches],'status':'present_requires_validation' if matches else 'missing'})
(OUT/'core_input_status.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
probes=[]
for url in ['https://depmap.org/portal/download/all/','https://depmap.org/portal/api/download/files']:
    row={'url':url,'utc':datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(url,timeout=25) as r:
            b=r.read(65536);row.update(http_status=r.status,content_type=r.headers.get('Content-Type'),prefix_sha256=hashlib.sha256(b).hexdigest(),status='verification_required' if b'Verification' in b or b'reCAPTCHA' in b else 'inspect_response')
    except Exception as e:row.update(status='request_failed',error=type(e).__name__)
    probes.append(row)
(OUT/'official_download_probes.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
old=[str(p.relative_to(ROOT)) for p in (ROOT/'runs').glob('*/COMPLETE.json')]
(OUT/'prior_completed_runs.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
instructions=['# 核心数据手动下载清单（继续分析）','','官方入口：https://depmap.org/portal/download/all/ 。请在正常浏览器中完成网站人机验证；程序不会代为绕过验证。若入口改版，可使用官方数据页 https://depmap.org/portal/data_page/?tab=allData 查找同一发布集；没有对应版本时请告知，不要选择最新版本替代。','','左侧分类选 DepMap Portal Harmonized Drug Screens，再在右侧 Version 选 Harmonized GDSC 25Q2。sanger-dose-response.csv 则在左侧 Sanger GDSC1 and GDSC2 分类。','','下表仅列本次仍缺失的文件；已存在的文件不会重复下载。','','| 应选择的发布集 | 文件名 | 完整保存位置 |','|---|---|---|']
instructions += [f"| {r['release']} | `{r['filename']}` | `{r['save_to']}` |" for r in rows if r['status']=='missing']
instructions += ['','下载完毕后请回复“已下载”。会先核对 CSV 结构、来源版本、校验和和 ID，再运行真实数据测试。若只能下载其中一部分，请说明已取得哪些文件；不会按零字节或 HTML 错误页认定下载成功。','','后续 PRISM 文件与版本见先前的缺口清单；此阶段优先取得上述六项。']
(OUT/'手动下载核心数据.md').write_text('\n'.join(instructions),encoding='utf-8')
print(json.dumps({'continuation_dir':str(OUT),'missing':sum(r['status']=='missing' for r in rows),'probes':probes},ensure_ascii=False))
