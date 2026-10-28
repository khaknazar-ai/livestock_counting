'use strict';
var $=function(id){return document.getElementById(id)};
var dz=$('drop-zone'),fi=$('file-input'),pc=$('preview-container'),pi=$('preview-img'),pn=$('preview-name'),bu=$('btn-upload');
var sf=null;
if(dz){dz.onclick=function(){fi.click()};fi.onchange=function(e){if(e.target.files.length)hf(e.target.files[0])};
dz.ondragover=function(e){e.preventDefault();dz.classList.add('drag-over')};dz.ondragleave=function(){dz.classList.remove('drag-over')};
dz.ondrop=function(e){e.preventDefault();dz.classList.remove('drag-over');if(e.dataTransfer.files[0])hf(e.dataTransfer.files[0])}}
function hf(f){if(['image/jpeg','image/png'].indexOf(f.type)<0){toast('请上传JPG或PNG','error');return}
sf=f;var r=new FileReader();r.onload=function(e){pi.src=e.target.result;pn.textContent=f.name;pc.classList.remove('hidden')};r.readAsDataURL(f);bu.disabled=false;tg('placeholder')}
if(bu)bu.onclick=function(){if(!sf)return;var fd=new FormData();fd.append('image',sf);var tn=$('task-name-input');if(tn&&tn.value)fd.append('task_name',tn.value);var ms=$('model-select');if(ms)fd.append('model',ms.value);tg('loading');
fetch('/api/upload',{method:'POST',body:fd}).then(function(r){return r.json()}).then(hr).catch(function(e){se(e.message)})};
function runExample(n){tg('loading');fetch('/api/example/'+encodeURIComponent(n)).then(function(r){return r.json()}).then(hr).catch(function(e){se(e.message)})}
function hr(d){if(!d.success){se(d.error||'失败');return}
// Always show count
$('count-value').textContent=d.display_count>=0?d.display_count:'—';
$('count-value').style.color='#2563eb';
var re=$('raw-count');if(re)re.textContent='Raw: '+d.raw_count;
$('meta-time').textContent=d.timestamp||'';$('meta-ms').textContent=(d.inference_ms||0)+' ms';$('meta-file').textContent=d.filename||'';
var mLabel=$('meta-model');if(mLabel){var mNames={'csrnet':'CSRNet','msa_csrnet':'MSA-CSRNet','yolo_finetune':'YOLOv8n-FT','yolo_zeroshot':'YOLOv8n-ZS','yolo':'YOLOv8n'};mLabel.textContent=mNames[d.model_used]||d.model_used||'CSRNet'};
// Confidence
var cb=$('confidence-badge'),ch=$('confidence-hint'),wl=$('warning-list'),si=$('status-indicator');
if(d.confidence){
  cb.textContent=d.confidence.label;cb.style.color=d.confidence.color;cb.classList.remove('hidden');
  ch.textContent=d.confidence.hint;ch.classList.remove('hidden');
  ch.style.background=d.confidence.level==='high'?'#ecfdf5':d.confidence.level==='medium'?'#fef3c7':'#fef2f2';
  ch.style.color=d.confidence.level==='high'?'#059669':d.confidence.level==='medium'?'#92400e':'#dc2626';
  ch.style.borderLeft='4px solid '+d.confidence.color;
  if(wl&&d.confidence.warnings&&d.confidence.warnings.length){wl.innerHTML=d.confidence.warnings.map(function(w){return'<li>'+w+'</li>'}).join('');wl.classList.remove('hidden')}
  else if(wl){wl.classList.add('hidden')}
}else{cb.classList.add('hidden');ch.classList.add('hidden');if(wl)wl.classList.add('hidden')}
// Status
if(si){if(d.status==='needs_review'){si.textContent='⚠ 建议人工复核';si.style.background='#fef3c7';si.style.color='#92400e';si.classList.remove('hidden')}
else{si.classList.add('hidden')}}
// Detail link
var db=$('btn-view-detail');if(d.task_id){db.href='/task/'+d.task_id;db.style.display='inline-flex';db.textContent='详情/校正'}
// Always show heatmap
if(d.heatmap_b64){$('heatmap-img').src='data:image/png;base64,'+d.heatmap_b64;$('heatmap-img').classList.remove('hidden');$('heatmap-placeholder').classList.add('hidden')}
if(d.overlay_b64){$('overlay-img').src='data:image/png;base64,'+d.overlay_b64;$('overlay-img').classList.remove('hidden');$('overlay-placeholder').classList.add('hidden')}
tg('result');toast('清点完成','success')}
function switchTab(n,b){document.querySelectorAll('.tab-btn').forEach(function(x){x.classList.remove('active')});document.querySelectorAll('.tab-content').forEach(function(x){x.classList.remove('active');x.classList.add('hidden')});b.classList.add('active');var t=$('tab-'+n);t.classList.add('active');t.classList.remove('hidden')}
function tg(s){['placeholder','result','loading','error'].forEach(function(n){var e=$(n+'-state');if(e)e.classList.toggle('hidden',n!==s)})}
function se(m){tg('error');$('error-msg').textContent=m;toast(m,'error')}
