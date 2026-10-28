"""webapp/database.py — SQLite with users + tasks + audit log.

Passwords are hashed with bcrypt (cost factor 12).
"""
import os, sqlite3
from datetime import datetime

import bcrypt  # requires bcrypt>=4.0 (see requirements.txt)

DB_NAME = 'counting_system.db'

def get_db_path(cfg): return os.path.join(cfg['paths']['log_dir'], DB_NAME)

def _conn(db_path):
    c = sqlite3.connect(db_path); c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL"); c.execute("PRAGMA foreign_keys=ON")
    return c


# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

_BCRYPT_ROUNDS = 12  # ~200 ms on a modern CPU — the intended slowness.


def _hash(pw):
    """Hash a password with bcrypt."""
    if pw is None:
        pw = ''
    return bcrypt.hashpw(pw.encode('utf-8'),
                         bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode('utf-8')


def check_pw(pw, stored):
    """Verify `pw` against a stored bcrypt hash."""
    if not stored:
        return False
    try:
        return bcrypt.checkpw((pw or '').encode('utf-8'),
                              stored.encode('utf-8'))
    except Exception:
        return False


def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    c = _conn(db_path)
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            organization TEXT DEFAULT '',
            role TEXT DEFAULT 'operator',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS counting_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            task_name TEXT DEFAULT '',
            filename TEXT NOT NULL,
            input_type TEXT DEFAULT 'image',
            timestamp TEXT NOT NULL,
            predicted_count REAL NOT NULL,
            corrected_count REAL,
            correction_note TEXT DEFAULT '',
            inference_ms REAL NOT NULL,
            status TEXT DEFAULT 'completed',
            confidence_level TEXT DEFAULT '',
            image_path TEXT DEFAULT '',
            heatmap_path TEXT DEFAULT '',
            overlay_path TEXT DEFAULT '',
            source TEXT DEFAULT 'upload',
            created_by TEXT DEFAULT '',
            model_used TEXT DEFAULT 'csrnet',
            is_deleted INTEGER DEFAULT 0,
            deleted_at TEXT DEFAULT '',
            deleted_by TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT DEFAULT '',
            action TEXT NOT NULL,
            target_type TEXT DEFAULT '',
            target_id TEXT DEFAULT '',
            detail TEXT DEFAULT '',
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_ts ON counting_tasks(timestamp);
        CREATE INDEX IF NOT EXISTS idx_task_st ON counting_tasks(status);
        CREATE INDEX IF NOT EXISTS idx_task_del ON counting_tasks(is_deleted);
        CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
    ''')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if c.execute('SELECT COUNT(*) as n FROM users').fetchone()['n'] == 0:
        for u, p, name, role in [
            ('admin','admin123','System Admin','admin'),
            ('manager','manager123','Farm Manager','manager'),
            ('operator','operator123','Farm Operator','operator')]:
            c.execute('INSERT INTO users (username,password,display_name,role,created_at) VALUES (?,?,?,?,?)',
                      (u, _hash(p), name, role, now))
    c.commit(); c.close()

# ── Audit ──
def audit(db_path, user, action, target_type='', target_id='', detail=''):
    c = _conn(db_path)
    uid = user.get('id','') if isinstance(user,dict) else ''
    uname = user.get('username','') if isinstance(user,dict) else str(user)
    c.execute('INSERT INTO audit_log (user_id,username,action,target_type,target_id,detail,timestamp) VALUES (?,?,?,?,?,?,?)',
              (uid, uname, action, target_type, target_id, detail, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    c.commit(); c.close()

def get_audit_log(db_path, limit=50, offset=0, action_filter='', user_filter=''):
    c = _conn(db_path)
    q, p = 'SELECT * FROM audit_log WHERE 1=1', []
    if action_filter: q += ' AND action=?'; p.append(action_filter)
    if user_filter: q += ' AND username=?'; p.append(user_filter)
    q += ' ORDER BY id DESC LIMIT ? OFFSET ?'; p += [limit, offset]
    rows = c.execute(q, p).fetchall(); c.close()
    return [dict(r) for r in rows]

def get_audit_for_task(db_path, task_id):
    c = _conn(db_path)
    rows = c.execute('SELECT * FROM audit_log WHERE target_id=? ORDER BY id DESC', (task_id,)).fetchall()
    c.close(); return [dict(r) for r in rows]

# ── Users ──
def get_user(db_path, username):
    c = _conn(db_path)
    r = c.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()
    c.close(); return dict(r) if r else None

def get_all_users(db_path):
    c = _conn(db_path)
    rows = c.execute('SELECT id,username,display_name,organization,role,is_active,created_at FROM users ORDER BY id').fetchall()
    c.close(); return [dict(r) for r in rows]

def create_user(db_path, username, password, display_name, role, organization=''):
    c = _conn(db_path)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        c.execute('INSERT INTO users (username,password,display_name,organization,role,created_at) VALUES (?,?,?,?,?,?)',
                  (username, _hash(password), display_name, organization, role, now))
        c.commit(); c.close(); return True
    except sqlite3.IntegrityError: c.close(); return False

def delete_user(db_path, uid):
    c = _conn(db_path)
    c.execute('DELETE FROM users WHERE id=? AND username!="admin"',(uid,))
    c.commit(); c.close()

def toggle_user_active(db_path, uid, active):
    c = _conn(db_path)
    c.execute('UPDATE users SET is_active=? WHERE id=?',(1 if active else 0, uid))
    c.commit(); c.close()

def reset_password(db_path, uid, new_pw):
    c = _conn(db_path)
    c.execute('UPDATE users SET password=? WHERE id=?',(_hash(new_pw), uid))
    c.commit(); c.close()

def change_password(db_path, uid, old_pw, new_pw):
    c = _conn(db_path)
    u = c.execute('SELECT password FROM users WHERE id=?',(uid,)).fetchone()
    if not u or not check_pw(old_pw, u['password']): c.close(); return False
    c.execute('UPDATE users SET password=? WHERE id=?',(_hash(new_pw), uid))
    c.commit(); c.close(); return True

def update_profile(db_path, uid, display_name, organization):
    c = _conn(db_path)
    c.execute('UPDATE users SET display_name=?, organization=? WHERE id=?',
              (display_name, organization, uid))
    c.commit(); c.close()

# ── Tasks ──
def insert_task(db_path, d):
    c = _conn(db_path)
    c.execute('''INSERT INTO counting_tasks
        (task_id,task_name,filename,input_type,timestamp,predicted_count,inference_ms,
         status,confidence_level,image_path,heatmap_path,overlay_path,source,created_by,model_used)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (d['task_id'],d.get('task_name',''),d['filename'],d.get('input_type','image'),
         d['timestamp'],d['predicted_count'],d['inference_ms'],
         d.get('status','completed'),d.get('confidence_level',''),
         d.get('image_path',''),d.get('heatmap_path',''),d.get('overlay_path',''),
         d.get('source','upload'),d.get('created_by',''),d.get('model_used','csrnet')))
    c.commit(); c.close()

def get_task(db_path, task_id):
    c = _conn(db_path)
    r = c.execute('SELECT * FROM counting_tasks WHERE task_id=? AND is_deleted=0',(task_id,)).fetchone()
    c.close(); return dict(r) if r else None

def get_all_tasks(db_path, limit=100, offset=0, search='', status_filter='',
                  user_filter='', confidence_filter='', date_from='', date_to=''):
    c = _conn(db_path)
    q, p = 'SELECT * FROM counting_tasks WHERE is_deleted=0', []
    if search:
        q += ' AND (filename LIKE ? OR task_name LIKE ? OR task_id LIKE ?)'
        s = f'%{search}%'; p += [s,s,s]
    if status_filter: q += ' AND status=?'; p.append(status_filter)
    if user_filter: q += ' AND created_by=?'; p.append(user_filter)
    if confidence_filter: q += ' AND confidence_level=?'; p.append(confidence_filter)
    if date_from: q += ' AND timestamp>=?'; p.append(date_from)
    if date_to: q += ' AND timestamp<=?'; p.append(date_to + ' 23:59:59')
    q += ' ORDER BY id DESC LIMIT ? OFFSET ?'; p += [limit, offset]
    rows = c.execute(q, p).fetchall(); c.close()
    return [dict(r) for r in rows]

def get_task_count(db_path, search='', status_filter='', user_filter='',
                   confidence_filter='', date_from='', date_to=''):
    c = _conn(db_path)
    q, p = 'SELECT COUNT(*) as n FROM counting_tasks WHERE is_deleted=0', []
    if search:
        q += ' AND (filename LIKE ? OR task_name LIKE ? OR task_id LIKE ?)'
        s = f'%{search}%'; p += [s,s,s]
    if status_filter: q += ' AND status=?'; p.append(status_filter)
    if user_filter: q += ' AND created_by=?'; p.append(user_filter)
    if confidence_filter: q += ' AND confidence_level=?'; p.append(confidence_filter)
    if date_from: q += ' AND timestamp>=?'; p.append(date_from)
    if date_to: q += ' AND timestamp<=?'; p.append(date_to + ' 23:59:59')
    r = c.execute(q, p).fetchone(); c.close(); return r['n']

def update_correction(db_path, task_id, val, note=''):
    c = _conn(db_path)
    c.execute("UPDATE counting_tasks SET corrected_count=?,correction_note=?,status='corrected' WHERE task_id=?",
              (val, note, task_id))
    c.commit(); c.close()

def update_task_name(db_path, task_id, name):
    c = _conn(db_path)
    c.execute('UPDATE counting_tasks SET task_name=? WHERE task_id=?',(name,task_id))
    c.commit(); c.close()

def soft_delete_task(db_path, task_id, username=''):
    c = _conn(db_path)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('UPDATE counting_tasks SET is_deleted=1, deleted_at=?, deleted_by=? WHERE task_id=?',
              (now, username, task_id))
    c.commit(); c.close()

def restore_task(db_path, task_id):
    c = _conn(db_path)
    c.execute('UPDATE counting_tasks SET is_deleted=0, deleted_at="", deleted_by="" WHERE task_id=?',(task_id,))
    c.commit(); c.close()

def batch_soft_delete(db_path, ids, username=''):
    c = _conn(db_path)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for tid in ids:
        c.execute('UPDATE counting_tasks SET is_deleted=1, deleted_at=?, deleted_by=? WHERE task_id=?',
                  (now, username, tid))
    c.commit(); c.close()

def get_deleted_tasks(db_path, limit=50):
    c = _conn(db_path)
    rows = c.execute('SELECT * FROM counting_tasks WHERE is_deleted=1 ORDER BY deleted_at DESC LIMIT ?',(limit,)).fetchall()
    c.close(); return [dict(r) for r in rows]

def get_tasks_for_export(db_path, ids=None):
    c = _conn(db_path)
    if ids:
        rows = c.execute(f'SELECT * FROM counting_tasks WHERE task_id IN ({",".join("?"*len(ids))}) AND is_deleted=0 ORDER BY id DESC', ids).fetchall()
    else:
        rows = c.execute('SELECT * FROM counting_tasks WHERE is_deleted=0 ORDER BY id DESC').fetchall()
    c.close(); return [dict(r) for r in rows]

def get_dashboard_stats(db_path):
    c = _conn(db_path)
    total = c.execute('SELECT COUNT(*) as n FROM counting_tasks WHERE is_deleted=0').fetchone()['n']
    if total == 0:
        c.close()
        return dict(total_tasks=0,avg_predicted=0,avg_corrected=0,correction_rate=0,
                    avg_inference_ms=0,recent_tasks=[],status_counts={},daily_counts=[],confidence_counts={})
    avg_p = c.execute('SELECT AVG(predicted_count) as v FROM counting_tasks WHERE is_deleted=0').fetchone()['v'] or 0
    cr = c.execute('SELECT AVG(corrected_count) as v, COUNT(*) as n FROM counting_tasks WHERE corrected_count IS NOT NULL AND is_deleted=0').fetchone()
    avg_ms = c.execute('SELECT AVG(inference_ms) as v FROM counting_tasks WHERE is_deleted=0').fetchone()['v'] or 0
    recent = c.execute('SELECT * FROM counting_tasks WHERE is_deleted=0 ORDER BY id DESC LIMIT 5').fetchall()
    sc = c.execute('SELECT status, COUNT(*) as c FROM counting_tasks WHERE is_deleted=0 GROUP BY status').fetchall()
    cc = c.execute('SELECT confidence_level, COUNT(*) as c FROM counting_tasks WHERE is_deleted=0 AND confidence_level!="" GROUP BY confidence_level').fetchall()
    daily = c.execute("SELECT SUBSTR(timestamp,1,10) as day, COUNT(*) as c, AVG(predicted_count) as avg_count FROM counting_tasks WHERE is_deleted=0 GROUP BY day ORDER BY day DESC LIMIT 14").fetchall()
    c.close()
    return dict(total_tasks=total, avg_predicted=round(avg_p,1),
                avg_corrected=round(cr['v'] or 0,1),
                correction_rate=round(cr['n']/total*100,1) if total else 0,
                avg_inference_ms=round(avg_ms,1),
                recent_tasks=[dict(r) for r in recent],
                status_counts={r['status']:r['c'] for r in sc},
                confidence_counts={r['confidence_level']:r['c'] for r in cc},
                daily_counts=[dict(r) for r in daily])
