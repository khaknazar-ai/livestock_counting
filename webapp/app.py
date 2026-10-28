"""webapp/app.py — Livestock Counting System v5"""
import os, sys, uuid, csv, base64, argparse, yaml, logging, functools, traceback
from datetime import datetime
from io import BytesIO, StringIO
import cv2
from flask import (Flask, request, jsonify, render_template, session,
                   send_from_directory, abort, Response, redirect, url_for, send_file)
from werkzeug.utils import secure_filename
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import predict, load_model, preprocess
from webapp.database import *
from webapp.validation import (DomainGate, run_tta, check_input_quality,
                                compute_confidence, format_count)

def login_required(f):
    @functools.wraps(f)
    def w(*a,**kw):
        if 'user' not in session: return redirect(url_for('login_page'))
        return f(*a,**kw)
    return w

def role_required(*roles):
    def d(f):
        @functools.wraps(f)
        def w(*a,**kw):
            if 'user' not in session: return redirect(url_for('login_page'))
            if session['user']['role'] not in roles: abort(403)
            return f(*a,**kw)
        return w
    return d

def create_app(cfg, checkpoint_path):
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = os.environ.get('LIVESTOCK_SECRET_KEY') or cfg.get('webapp',{}).get('secret_key') or os.urandom(32)
    wc = cfg['webapp']
    app.config['MAX_CONTENT_LENGTH'] = wc['max_upload_mb']*1024*1024
    app.config['UPLOAD_FOLDER'] = os.path.abspath(cfg['paths']['webapp_upload'])
    app.config['RESULTS_FOLDER'] = os.path.abspath(os.path.join(cfg['paths']['webapp_upload'],'results'))
    app.config['ALLOWED_EXT'] = set(wc['allowed_ext'])
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    inp_size = tuple(cfg['train']['input_size'])

    # ── Load CSRNet ──
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path, device, model_name='csrnet'); ml = True
    else: model = None; ml = False

    # ── Load MSA-CSRNet ──
    msa_model = None; msa_loaded = False
    log_dir = cfg['paths']['log_dir']
    msa_path = os.path.join(log_dir, 'checkpoint_msa_csrnet_best.pth')
    if os.path.exists(msa_path):
        try:
            msa_model = load_model(msa_path, device, model_name='msa_csrnet')
            msa_loaded = True
            print(f"  [MSA-CSRNet] Loaded: {msa_path}")
        except Exception as e:
            print(f"  [MSA-CSRNet] Failed: {e}")
    else:
        print(f"  [MSA-CSRNet] Not found: {msa_path}")

    # ── Load YOLOv8n fine-tuned ──
    yolo_model = None; yolo_loaded = False
    yolo_path = cfg.get('baseline', {}).get('finetune_weights', 'baseline/yolo_finetune/train/weights/best.pt')
    for yp in [yolo_path,
               'baseline/yolo_finetune/train/weights/best.pt',
               'baseline/yolo_finetune/train2/weights/best.pt',
               'baseline/yolo_finetune/train3/weights/best.pt',
               'baseline/yolo_finetune/train4/weights/best.pt',
               'baseline/yolo_finetune/train5/weights/best.pt',
               'baseline/yolo_finetune/train6/weights/best.pt']:
        if os.path.exists(yp):
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(yp)
                yolo_loaded = True
                print(f"  [YOLO-FT] Loaded: {yp}")
            except Exception as e:
                print(f"  [YOLO-FT] Failed to load: {e}")
            break
    if not yolo_loaded:
        print("  [YOLO-FT] No fine-tuned weights found")

    # ── Load YOLOv8n zero-shot (COCO) ──
    yolo_zs_model = None; yolo_zs_loaded = False
    try:
        from ultralytics import YOLO
        yolo_zs_model = YOLO('yolov8n.pt')
        yolo_zs_loaded = True
        print("  [YOLO-ZS] Loaded yolov8n.pt (COCO zero-shot)")
    except Exception as e:
        print(f"  [YOLO-ZS] Not available: {e}")

    # Model registry for the web app
    model_registry = {
        'csrnet':       {'model': model,        'loaded': ml,            'label': 'CSRNet', 'type': 'density'},
        'msa_csrnet':   {'model': msa_model,    'loaded': msa_loaded,    'label': 'MSA-CSRNet', 'type': 'density'},
        'yolo_finetune':{'model': yolo_model,   'loaded': yolo_loaded,   'label': 'YOLOv8n Fine-tuned', 'type': 'yolo'},
        'yolo_zeroshot':{'model': yolo_zs_model,'loaded': yolo_zs_loaded,'label': 'YOLOv8n Zero-shot', 'type': 'yolo'},
    }

    app.config.update(MODEL=model, MSA_MODEL=msa_model, YOLO_MODEL=yolo_model,
                      YOLO_ZS_MODEL=yolo_zs_model, DEVICE=device,
                      INPUT_SIZE=inp_size, MODEL_LOADED=ml,
                      MSA_LOADED=msa_loaded, YOLO_LOADED=yolo_loaded,
                      YOLO_ZS_LOADED=yolo_zs_loaded, CFG=cfg,
                      MODEL_REGISTRY=model_registry)
    db = get_db_path(cfg); init_db(db); app.config['DB'] = db

    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models','pretrained','domain_reference.npz')
    gate = DomainGate(ref_path)

    def ok_file(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in app.config['ALLOWED_EXT']
    def cu(): return session.get('user',{})
    def is_r(*r): return cu().get('role') in r

    def can_access(task):
        """Ownership check: operator sees only own tasks, admin/manager sees all."""
        if not task: return False
        if is_r('admin', 'manager'): return True
        return task.get('created_by', '') == cu().get('username', '')

    @app.context_processor
    def inj():
        return dict(current_user=cu(), model_loaded=ml,
                    msa_loaded=msa_loaded, yolo_loaded=yolo_loaded,
                    yolo_zs_loaded=yolo_zs_loaded, model_registry=model_registry)

    @app.errorhandler(403)
    def e403(e): return render_template('error.html', code=403, msg='权限不足'), 403
    @app.errorhandler(404)
    def e404(e): return render_template('error.html', code=404, msg='页面未找到'), 404
    @app.errorhandler(500)
    def e500(e): return render_template('error.html', code=500, msg='服务器内部错误'), 500

    # ══ Auth ══
    @app.route('/login', methods=['GET','POST'])
    def login_page():
        if request.method == 'GET': return render_template('login.html')
        u = get_user(db, request.form.get('username',''))
        if not u or not check_pw(request.form.get('password',''), u['password']):
            return render_template('login.html', error='用户名或密码错误')
        if not u.get('is_active',1):
            return render_template('login.html', error='账号已停用，请联系管理员')
        session['user'] = {k:u[k] for k in ['id','username','display_name','role','organization']}
        audit(db, session['user'], 'login', 'user', u['username'])
        return redirect('/')

    @app.route('/logout')
    def logout():
        if 'user' in session: audit(db, session['user'], 'logout')
        session.pop('user',None); return redirect('/login')

    # ══ Pages ══
    @app.route('/')
    @login_required
    def pg_index():
        ex = _list_examples(wc.get('example_images','webapp/static/examples'))
        return render_template('index.html', examples=ex, active_page='count')

    @app.route('/batch')
    @login_required
    def pg_batch(): return render_template('batch.html', active_page='batch')

    @app.route('/history')
    @login_required
    def pg_history():
        s = request.args.get('search','')
        st = request.args.get('status','')
        conf = request.args.get('confidence','')
        uf_q = request.args.get('user','')
        df = request.args.get('date_from','')
        dt = request.args.get('date_to','')
        pg, pp = int(request.args.get('page',1)), 20
        uf = '' if is_r('admin','manager') else cu().get('username','')
        if uf_q and is_r('admin','manager'): uf = uf_q
        tasks = get_all_tasks(db, pp, (pg-1)*pp, s, st, uf, conf, df, dt)
        total = get_task_count(db, s, st, uf, conf, df, dt)
        users = get_all_users(db) if is_r('admin','manager') else []
        return render_template('history.html', tasks=tasks, page=pg,
            total_pages=max(1,(total+pp-1)//pp), total=total,
            search=s, status=st, confidence=conf, user_filter=uf_q,
            date_from=df, date_to=dt, all_users=users, active_page='history')

    @app.route('/task/<tid>')
    @login_required
    def pg_detail(tid):
        t = get_task(db, tid)
        if not t or not can_access(t): abort(404)
        logs = get_audit_for_task(db, tid)
        return render_template('detail.html', task=t,
            confidence=_conf_from_db(t.get('confidence_level','')),
            audit_logs=logs, active_page='history')

    @app.route('/dashboard')
    @role_required('admin','manager')
    def pg_dash():
        return render_template('dashboard.html', stats=get_dashboard_stats(db), active_page='dashboard')

    @app.route('/help')
    @login_required
    def pg_help(): return render_template('help.html', active_page='help')

    @app.route('/profile', methods=['GET','POST'])
    @login_required
    def pg_profile():
        msg = ''
        if request.method == 'POST':
            act = request.form.get('action')
            if act == 'update_profile':
                update_profile(db, cu()['id'], request.form.get('display_name',''),
                               request.form.get('organization',''))
                session['user']['display_name'] = request.form.get('display_name','')
                session['user']['organization'] = request.form.get('organization','')
                msg = '个人信息已更新'
                audit(db, cu(), 'update_profile', 'user', cu()['username'])
            elif act == 'change_password':
                ok = change_password(db, cu()['id'], request.form.get('old_password',''),
                                     request.form.get('new_password',''))
                msg = '密码已修改' if ok else '旧密码错误'
                if ok: audit(db, cu(), 'change_password', 'user', cu()['username'])
        return render_template('profile.html', msg=msg, active_page='profile')

    @app.route('/admin')
    @role_required('admin')
    def pg_admin():
        users = get_all_users(db)
        deleted = get_deleted_tasks(db, 20)
        logs = get_audit_log(db, 50)
        return render_template('admin.html', users=users, deleted_tasks=deleted,
                               audit_logs=logs, active_page='admin')

    # ══ API: Upload ══
    @app.route('/api/upload', methods=['POST'])
    @login_required
    def api_upload():
        try:
            if 'image' not in request.files:
                return jsonify(success=False, error='请选择图片文件'), 400
            f = request.files['image']
            if not f.filename:
                return jsonify(success=False, error='未选择文件'), 400
            if not ok_file(f.filename):
                return jsonify(success=False, error=f'不支持的文件类型。请上传 {", ".join(app.config["ALLOWED_EXT"])}'), 400
            tid = str(uuid.uuid4()); ext = f.filename.rsplit('.',1)[1].lower()
            sn = f"{tid}.{ext}"; save_p = os.path.join(app.config['UPLOAD_FOLDER'], sn)
            f.save(save_p)
            # Verify file readable
            test = cv2.imread(save_p)
            if test is None:
                os.remove(save_p)
                return jsonify(success=False, error='文件已损坏或格式不正确'), 400

            # Model selection (non-admin forced to csrnet)
            model_choice = request.form.get('model', 'csrnet')
            if cu().get('role') not in ('admin',):
                model_choice = 'csrnet'
            if model_choice == 'yolo_finetune' and yolo_model:
                r = _infer_yolo(save_p, yolo_model, label='YOLOv8n-FT')
            elif model_choice == 'yolo_zeroshot' and yolo_zs_model:
                r = _infer_yolo(save_p, yolo_zs_model, label='YOLOv8n-ZS')
            elif model_choice == 'msa_csrnet' and msa_model:
                r = _infer(save_p, msa_model, device, inp_size, gate)
            else:
                r = _infer(save_p, model, device, inp_size, gate)
                model_choice = 'csrnet'

            _save_imgs(tid, r)
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conf_level = r['confidence'].get('level','') if r.get('confidence') else ''
            status = 'needs_review' if r.get('confidence',{}).get('needs_review') else 'completed'

            insert_task(db, dict(task_id=tid, task_name=request.form.get('task_name',''),
                filename=secure_filename(f.filename), input_type='image', timestamp=ts,
                predicted_count=r.get('count',0), inference_ms=r.get('inference_ms',0),
                status=status, confidence_level=conf_level,
                image_path=sn, heatmap_path=f'{tid}_heatmap.png', overlay_path=f'{tid}_overlay.png',
                source='upload', created_by=cu().get('username',''), model_used=model_choice))
            audit(db, cu(), 'upload', 'task', tid, f"model={model_choice}, file={f.filename}, count={r.get('display_count',0)}")
            return jsonify(success=True, task_id=tid,
                display_count=r.get('display_count',0), raw_count=r.get('raw_count',0),
                inference_ms=round(r.get('inference_ms',0),1), timestamp=ts,
                heatmap_b64=r.get('hb64',''), overlay_b64=r.get('ob64',''),
                filename=secure_filename(f.filename), confidence=r.get('confidence',{}),
                status=status, model_used=model_choice)
        except Exception as e:
            logging.error(f"Upload error: {traceback.format_exc()}")
            return jsonify(success=False, error=f'推理失败: {str(e)}'), 500

    # ══ API: Batch ══
    @app.route('/api/batch_upload', methods=['POST'])
    @login_required
    def api_batch():
        files = request.files.getlist('images')
        if not files: return jsonify(success=False, error='未选择文件'), 400
        bn = request.form.get('batch_name',''); results = []; total_c = 0
        model_choice = request.form.get('model', 'csrnet')
        if cu().get('role') not in ('admin',):
            model_choice = 'csrnet'
        for f in files:
            if not f.filename or not ok_file(f.filename):
                results.append(dict(filename=f.filename or '?', success=False, error='不支持的格式')); continue
            try:
                tid = str(uuid.uuid4()); ext = f.filename.rsplit('.',1)[1].lower()
                sn = f"{tid}.{ext}"; f.save(os.path.join(app.config['UPLOAD_FOLDER'], sn))
                save_p = os.path.join(app.config['UPLOAD_FOLDER'], sn)
                test = cv2.imread(save_p)
                if test is None:
                    os.remove(save_p)
                    results.append(dict(filename=f.filename, success=False, error='文件已损坏或格式不正确'))
                    continue
                if model_choice == 'yolo_finetune' and yolo_model:
                    r = _infer_yolo(save_p, yolo_model, label='YOLOv8n-FT')
                elif model_choice == 'yolo_zeroshot' and yolo_zs_model:
                    r = _infer_yolo(save_p, yolo_zs_model, label='YOLOv8n-ZS')
                elif model_choice == 'msa_csrnet' and msa_model:
                    r = _infer(save_p, msa_model, device, inp_size, gate)
                else:
                    r = _infer(save_p, model, device, inp_size, gate)
                    model_choice = 'csrnet'
                _save_imgs(tid, r)
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                nm = f"{bn} - {f.filename}" if bn else ''
                conf_level = r.get('confidence',{}).get('level','')
                status = 'needs_review' if r.get('confidence',{}).get('needs_review') else 'completed'
                insert_task(db, dict(task_id=tid, task_name=nm, filename=secure_filename(f.filename),
                    input_type='batch', timestamp=ts, predicted_count=r.get('count',0),
                    inference_ms=r.get('inference_ms',0), status=status, confidence_level=conf_level,
                    image_path=sn, heatmap_path=f'{tid}_heatmap.png', overlay_path=f'{tid}_overlay.png',
                    source='batch', created_by=cu().get('username',''), model_used=model_choice))
                dc = r.get('display_count',0); total_c += dc
                results.append(dict(filename=f.filename, success=True, task_id=tid,
                    display_count=dc, raw_count=r.get('raw_count',0),
                    inference_ms=round(r.get('inference_ms',0),1),
                    confidence=r.get('confidence',{}), status=status))
            except Exception as e:
                results.append(dict(filename=f.filename, success=False, error=str(e)))
        n_ok = sum(1 for x in results if x.get('success'))
        audit(db, cu(), 'batch_upload', 'task', '', f"images={len(files)}, success={n_ok}")
        # Batch summary
        counts = [x.get('display_count',0) for x in results if x.get('success')]
        avg_c = round(sum(counts)/len(counts),1) if counts else 0
        conf_dist = {}
        for x in results:
            if x.get('success'):
                cl = x.get('confidence',{}).get('level','unknown')
                conf_dist[cl] = conf_dist.get(cl,0)+1
        return jsonify(success=True, results=results, total_images=len(results),
            successful=n_ok, total_count=total_c, avg_count=avg_c, confidence_distribution=conf_dist)

    # ══ API: Example ══
    @app.route('/api/example/<name>', methods=['GET'])
    @login_required
    def api_example(name):
        d = wc.get('example_images','webapp/static/examples')
        img = next((os.path.join(d,f"{name}.{e}") for e in ['jpg','jpeg','png']
                    if os.path.exists(os.path.join(d,f"{name}.{e}"))), None)
        if not img: return jsonify(success=False, error='示例未找到'), 404
        r = _infer(img, model, device, inp_size, gate)
        tid = str(uuid.uuid4()); ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _save_imgs(tid, r)
        conf_level = r.get('confidence',{}).get('level','')
        status = 'needs_review' if r.get('confidence',{}).get('needs_review') else 'completed'
        insert_task(db, dict(task_id=tid, task_name=f'Example: {name}',
            filename=os.path.basename(img), input_type='example', timestamp=ts,
            predicted_count=r.get('count',0), inference_ms=r.get('inference_ms',0),
            status=status, confidence_level=conf_level,
            image_path='', heatmap_path=f'{tid}_heatmap.png', overlay_path=f'{tid}_overlay.png',
            source=f'example:{name}', created_by=cu().get('username','')))
        return jsonify(success=True, task_id=tid, display_count=r.get('display_count',0),
            raw_count=r.get('raw_count',0), inference_ms=round(r.get('inference_ms',0),1),
            timestamp=ts, heatmap_b64=r.get('hb64',''), overlay_b64=r.get('ob64',''),
            filename=os.path.basename(img), confidence=r.get('confidence',{}), status=status)

    # ══ API: Task CRUD ══
    @app.route('/api/task/<tid>/correct', methods=['POST'])
    @role_required('admin','manager')
    def api_correct(tid):
        d = request.get_json()
        if d.get('corrected_count') is None: return jsonify(success=False, error='请输入校正数量'), 400
        val = round(float(d['corrected_count']))  # Store as integer
        update_correction(db, tid, val, d.get('correction_note',''))
        audit(db, cu(), 'correct', 'task', tid, f"corrected={val}")
        return jsonify(success=True, msg='校正已保存')

    @app.route('/api/task/<tid>/name', methods=['POST'])
    @login_required
    def api_name(tid):
        t = get_task(db, tid)
        if not t or not can_access(t): return jsonify(success=False, error='无权限'), 403
        update_task_name(db, tid, request.get_json().get('task_name',''))
        return jsonify(success=True)

    @app.route('/api/task/<tid>', methods=['DELETE'])
    @role_required('admin','manager')
    def api_del(tid):
        soft_delete_task(db, tid, cu().get('username',''))
        audit(db, cu(), 'delete', 'task', tid)
        return jsonify(success=True, msg='记录已移至回收站')

    @app.route('/api/task/<tid>/restore', methods=['POST'])
    @role_required('admin')
    def api_restore(tid):
        restore_task(db, tid)
        audit(db, cu(), 'restore', 'task', tid)
        return jsonify(success=True)

    @app.route('/api/tasks/batch_delete', methods=['POST'])
    @role_required('admin')
    def api_bdel():
        ids = request.get_json().get('task_ids',[])
        if ids:
            batch_soft_delete(db, ids, cu().get('username',''))
            audit(db, cu(), 'batch_delete', 'task', '', f"count={len(ids)}")
        return jsonify(success=True, deleted=len(ids))

    # ══ API: Export ══
    @app.route('/api/export/csv')
    @role_required('admin','manager')
    def api_csv():
        tasks = _exp(request, db)
        out = StringIO(); out.write('\ufeff')
        w = csv.writer(out)
        w.writerow(['TaskID','Name','File','Type','Time','Predicted','Corrected','Note','ms','Status','Confidence','User'])
        for t in tasks:
            w.writerow([t['task_id'][:12],t['task_name'],t['filename'],t['input_type'],t['timestamp'],
                round(t['predicted_count']),
                round(t['corrected_count']) if t['corrected_count'] is not None else '',
                t['correction_note'],round(t['inference_ms'],1),t['status'],t.get('confidence_level',''),t.get('created_by','')])
        audit(db, cu(), 'export_csv', 'report', '', f"records={len(tasks)}")
        return Response(out.getvalue().encode('utf-8-sig'), mimetype='text/csv; charset=utf-8-sig',
            headers={'Content-Disposition':f'attachment; filename=report_{datetime.now():%Y%m%d_%H%M%S}.csv'})

    @app.route('/api/export/excel')
    @role_required('admin','manager')
    def api_excel():
        try: import openpyxl; from openpyxl.styles import Font, PatternFill
        except: return jsonify(success=False, error='pip install openpyxl'), 500
        tasks = _exp(request, db); wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Report"
        hdrs = ['TaskID','Name','File','Type','Time','Predicted','Corrected','Note','ms','Status','Confidence','User']
        hf, hfill = Font(bold=True,color="FFFFFF"), PatternFill(start_color="2563EB",fill_type="solid")
        for c,h in enumerate(hdrs,1):
            cell = ws.cell(row=1,column=c,value=h); cell.font=hf; cell.fill=hfill
        for r,t in enumerate(tasks,2):
            for c,v in enumerate([t['task_id'][:12],t['task_name'],t['filename'],t['input_type'],
                t['timestamp'],round(t['predicted_count']),
                round(t['corrected_count']) if t['corrected_count'] is not None else '',
                t['correction_note'],round(t['inference_ms'],1),t['status'],t.get('confidence_level',''),t.get('created_by','')],1):
                ws.cell(row=r,column=c,value=v)
        for c in range(1,len(hdrs)+1): ws.column_dimensions[openpyxl.utils.get_column_letter(c)].width=16
        buf = BytesIO(); wb.save(buf); buf.seek(0)
        audit(db, cu(), 'export_excel', 'report', '', f"records={len(tasks)}")
        return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True, download_name=f'report_{datetime.now():%Y%m%d_%H%M%S}.xlsx')

    @app.route('/api/export/report')
    @role_required('admin','manager')
    def api_report():
        tasks = _exp(request, db); stats = get_dashboard_stats(db)
        rows = ''.join(f"<tr><td>{t['timestamp']}</td><td>{t['filename'][:30]}</td>"
               f"<td>{round(t['predicted_count'])}</td>"
               f"<td>{round(t['corrected_count']) if t['corrected_count'] is not None else '-'}</td>"
               f"<td>{t['status']}</td><td>{t.get('created_by','')}</td></tr>" for t in tasks[:200])
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
        body{{font-family:sans-serif;padding:20px;font-size:12px}}h1{{font-size:18px}}
        table{{width:100%;border-collapse:collapse;margin-top:10px}}
        th{{background:#2563eb;color:#fff;padding:6px 8px;text-align:left}}
        td{{padding:5px 8px;border-bottom:1px solid #eee}}</style></head><body>
        <h1>Livestock Counting Report</h1>
        <p>Generated: {datetime.now():%Y-%m-%d %H:%M:%S} | Total: {stats['total_tasks']} | Avg: {stats['avg_predicted']} | Correction: {stats['correction_rate']}%</p>
        <table><tr><th>Time</th><th>File</th><th>Predicted</th><th>Corrected</th><th>Status</th><th>User</th></tr>{rows}</table></body></html>"""
        buf = BytesIO(); buf.write(html.encode('utf-8')); buf.seek(0)
        audit(db, cu(), 'export_report', 'report', '', f"records={len(tasks)}")
        return send_file(buf, mimetype='text/html', as_attachment=True,
            download_name=f'report_{datetime.now():%Y%m%d_%H%M%S}.html')

    @app.route('/api/export/task/<tid>')
    @login_required
    def api_export_task(tid):
        t = get_task(db, tid)
        if not t or not can_access(t): return jsonify(success=False, error='无权限'), 403
        out = StringIO(); out.write('\ufeff'); w = csv.writer(out)
        w.writerow(['Field','Value'])
        for k,v in [('TaskID',t['task_id']),('Name',t['task_name']),('File',t['filename']),
            ('Time',t['timestamp']),('Predicted',round(t['predicted_count'])),
            ('Corrected',round(t['corrected_count']) if t['corrected_count'] is not None else ''),
            ('Note',t['correction_note']),('Status',t['status']),
            ('Confidence',t.get('confidence_level','')),('User',t.get('created_by','')),
            ('Inference_ms',round(t['inference_ms'],1))]:
            w.writerow([k,v])
        return Response(out.getvalue().encode('utf-8-sig'), mimetype='text/csv',
            headers={'Content-Disposition':f'attachment; filename=task_{tid[:8]}.csv'})

    # ══ API: Admin ══
    @app.route('/api/admin/user', methods=['POST'])
    @role_required('admin')
    def api_add_user():
        d = request.get_json()
        if not d.get('username') or not d.get('password'):
            return jsonify(success=False, error='用户名和密码不能为空')
        ok = create_user(db, d['username'], d['password'], d.get('display_name',''),
                         d.get('role','operator'), d.get('organization',''))
        if ok: audit(db, cu(), 'create_user', 'user', d['username'])
        return jsonify(success=ok, error='' if ok else '用户名已存在')

    @app.route('/api/admin/user/<int:uid>', methods=['DELETE'])
    @role_required('admin')
    def api_del_user(uid):
        delete_user(db, uid)
        audit(db, cu(), 'delete_user', 'user', str(uid))
        return jsonify(success=True)

    @app.route('/api/admin/user/<int:uid>/toggle', methods=['POST'])
    @role_required('admin')
    def api_toggle_user(uid):
        d = request.get_json()
        toggle_user_active(db, uid, d.get('active', True))
        audit(db, cu(), 'toggle_user', 'user', str(uid), f"active={d.get('active')}")
        return jsonify(success=True)

    @app.route('/api/admin/user/<int:uid>/reset', methods=['POST'])
    @role_required('admin')
    def api_reset_pw(uid):
        d = request.get_json() or {}
        new_password = d.get('password', '')
        if not new_password:
            return jsonify(success=False, error='请输入新密码'), 400
        reset_password(db, uid, new_password)
        audit(db, cu(), 'reset_password', 'user', str(uid))
        return jsonify(success=True)

    # ══ Static ══
    @app.route('/api/result_image/<fn>')
    @login_required
    def serve_r(fn): return send_from_directory(app.config['RESULTS_FOLDER'], secure_filename(fn))
    @app.route('/api/upload_image/<fn>')
    @login_required
    def serve_u(fn): return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(fn))
    @app.route('/api/health')
    def health(): return jsonify(status='ok', model_loaded=ml, domain_gate=gate.enabled)

    def _save_imgs(tid, r):
        if r.get('hbgr') is not None:
            cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], f'{tid}_heatmap.png'), r['hbgr'])
            cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], f'{tid}_overlay.png'), r['obgr'])

    return app

def _infer(path, model, device, size, gate=None):
    """
    Simple pipeline: quality check → count → TTA → confidence.
    ALWAYS shows results. No rejection. Human correction is the gate.
    """
    if model is None:
        return dict(count=0, raw_count=0, display_count=0, inference_ms=0,
                    hb64='', ob64='', hbgr=None, obgr=None,
                    confidence=dict(level='low', label='模型未加载', color='#dc2626',
                                    hint='模型文件未加载。', needs_review=True, warnings=['模型未加载']))

    # 1. Input quality
    iq = check_input_quality(path)
    input_quality = iq.get('quality', 'acceptable')

    # 2. Count (always runs)
    r = predict(path, model, device, size)
    raw = r['count']
    dc, rr = format_count(raw)

    # 3. TTA stability
    tta = None
    try:
        tensor, _ = preprocess(path, size)
        tta = run_tta(model, tensor, device, 3)
    except:
        pass

    # 4. Domain similarity (optional, for experiment analysis)
    similarity = None
    if gate and gate.enabled:
        try:
            tensor, _ = preprocess(path, size)
            features = gate.extract_features(model, tensor, device)
            similarity = gate.compute_similarity(features)
        except:
            pass

    # 5. Confidence (TTA stability + quality, no rejection)
    conf = compute_confidence(raw, input_quality, tta, similarity)
    conf['warnings'] = iq.get('warnings', []) + conf.get('warnings', [])

    # Always show heatmap — user judges with their eyes
    return dict(
        count=raw, raw_count=rr, display_count=dc,
        inference_ms=r['inference_ms'],
        hb64=_b64(r['heatmap_bgr']), ob64=_b64(r['overlay_bgr']),
        hbgr=r['heatmap_bgr'], obgr=r['overlay_bgr'],
        confidence=conf,
    )

def _infer_yolo(path, yolo_model, label='YOLOv8n'):
    """
    YOLOv8n inference: detect + count + draw boxes.
    Returns same dict format as _infer for compatibility.
    """
    import time as _time
    img = cv2.imread(path)
    if img is None:
        return dict(count=0, raw_count=0, display_count=0, inference_ms=0,
                    hb64='', ob64='', hbgr=None, obgr=None,
                    confidence=dict(level='low', label='检测失败', color='#dc2626',
                                    hint='无法读取图片', needs_review=True, warnings=[]))

    t0 = _time.time()
    results = yolo_model.predict(path, conf=0.25, iou=0.45, imgsz=640, verbose=False)
    inference_ms = (_time.time() - t0) * 1000

    # Count detections
    boxes = results[0].boxes
    count = len(boxes)
    dc, rr = format_count(float(count))

    # Draw bounding boxes on image (overlay)
    overlay = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf_val = float(box.conf[0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f'{conf_val:.2f}', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # No heatmap for YOLO — use detection visualization as both
    confidence = dict(
        level='high' if count > 0 else 'low',
        label=f'{label}检测' if count > 0 else '未检测到',
        color='#2563eb' if count > 0 else '#dc2626',
        hint=f'{label}检测到 {count} 个目标（bbox）',
        needs_review=False, warnings=[],
    )

    return dict(
        count=float(count), raw_count=rr, display_count=dc,
        inference_ms=inference_ms,
        hb64='', ob64=_b64(overlay),
        hbgr=None, obgr=overlay,
        confidence=confidence,
    )

def _b64(img):
    if img is None: return ''
    ok, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ''
def _list_examples(d):
    if not os.path.exists(d): return []
    return [os.path.splitext(f)[0] for f in sorted(os.listdir(d)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
def _conf_from_db(level):
    """Reconstruct confidence display from stored level string."""
    mapping = {
        'high':   dict(level='high', label='高置信度', color='#059669', hint='模型预测稳定，结果可信。', needs_review=False, warnings=[]),
        'medium': dict(level='medium', label='中置信度', color='#d97706', hint='结果基本可靠，建议参考热力图。', needs_review=False, warnings=[]),
        'low':    dict(level='low', label='低置信度', color='#dc2626', hint='建议人工核查并校正。', needs_review=True, warnings=[]),
    }
    return mapping.get(level, dict(level='unknown', label='—', color='#94a3b8', hint='', needs_review=False, warnings=[]))
def _exp(req, db):
    ids = req.args.get('task_ids','')
    ids = [t.strip() for t in ids.split(',') if t.strip()] if ids else None
    return get_tasks_for_export(db, ids)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--checkpoint', default='experiments/logs/checkpoint_best.pth')
    a = p.parse_args()
    with open(a.config) as f: cfg = yaml.safe_load(f)
    logging.basicConfig(level=logging.INFO)
    app = create_app(cfg, a.checkpoint)
    print(f"\n  Livestock Counting System v5\n  http://127.0.0.1:{cfg['webapp']['port']}")
    print(f"  Accounts: admin/admin123  manager/manager123  operator/operator123\n")
    app.run(host=cfg['webapp']['host'], port=cfg['webapp']['port'], debug=cfg['webapp']['debug'])
