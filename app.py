"""
SMART COLLEGE ATTENDANCE SYSTEM  (Flask + face_recognition + SQLite)
====================================================================
Features
  * Admin login
  * Dashboard (today's stats, 7-day trend, department-wise attendance, recent entries)
  * Student management (add with photo upload or webcam, delete)
  * LIVE GROUP ATTENDANCE: one webcam sees the whole class, every recognised face is
    marked present at the same time (once per student / subject / day)
  * Reports by date / subject / department / section + CSV export

Install
    pip install flask face_recognition numpy pillow
    (face_recognition needs dlib. On Windows, if install fails, try: pip install dlib-bin)

Run
    python app.py
    Open  http://127.0.0.1:5000   (use Chrome/Edge; camera works on localhost)

Default login:  admin / admin123
Change it with environment variables ADMIN_USER, ADMIN_PASS, SECRET_KEY.
"""
import base64
import csv
import io
import os
import sqlite3
import threading
from contextlib import closing
from datetime import date, datetime, timedelta
from functools import wraps

import face_recognition
import numpy as np
from flask import (Flask, Response, flash, g, jsonify, redirect,
                   render_template, request, send_from_directory, session, url_for)
from jinja2 import DictLoader
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------ config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "college_attendance.db")
PHOTO_DIR = os.path.join(BASE_DIR, "student_photos")
TOLERANCE = 0.5            # lower = stricter face matching
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")

os.makedirs(PHOTO_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ---------------------------------------------------------------- database
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with closing(sqlite3.connect(DB_FILE)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                section TEXT NOT NULL,
                photo TEXT,
                encoding BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                subject TEXT NOT NULL,
                UNIQUE(student_id, date, subject)
            );
            """
        )
        conn.commit()


# ------------------------------------------------- in-memory face encodings
_cache = {"students": []}
_lock = threading.Lock()


def reload_cache():
    with closing(sqlite3.connect(DB_FILE)) as conn:
        rows = conn.execute(
            "SELECT id, roll_no, name, department, section, encoding FROM students"
        ).fetchall()
    students = [
        {
            "id": r[0], "roll_no": r[1], "name": r[2],
            "department": r[3], "section": r[4],
            "encoding": np.frombuffer(r[5], dtype=np.float64),
        }
        for r in rows
    ]
    with _lock:
        _cache["students"] = students


def get_candidates(dept, section):
    with _lock:
        return [
            s for s in _cache["students"]
            if (not dept or s["department"] == dept)
            and (not section or s["section"] == section)
        ]


# ------------------------------------------------------------------ helpers
def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify(error="login required"), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapper


def decode_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data_url)))
    return ImageOps.exif_transpose(img).convert("RGB")


def single_face_encoding(img):
    """Return (encoding, number_of_faces). encoding is None unless exactly 1 face."""
    arr = np.array(img)
    locations = face_recognition.face_locations(arr)
    if len(locations) != 1:
        return None, len(locations)
    return face_recognition.face_encodings(arr, locations)[0], 1


def get_filter_options(db):
    departments = [r[0] for r in db.execute("SELECT DISTINCT department FROM students ORDER BY 1")]
    sections = [r[0] for r in db.execute("SELECT DISTINCT section FROM students ORDER BY 1")]
    subjects = [r[0] for r in db.execute("SELECT DISTINCT subject FROM attendance ORDER BY 1")]
    return departments, sections, subjects


def build_summary(db, candidates, subject, day):
    ids = {s["id"] for s in candidates}
    rows = db.execute(
        """SELECT s.id, s.name, s.roll_no, a.time
           FROM attendance a JOIN students s ON s.id = a.student_id
           WHERE a.date = ? AND a.subject = ? ORDER BY a.time""",
        (day, subject),
    ).fetchall()
    present = [
        {"name": r["name"], "roll_no": r["roll_no"], "time": r["time"]}
        for r in rows if r["id"] in ids
    ]
    return {"present": present, "present_count": len(present), "total": len(candidates)}


def build_report(args):
    db = get_db()
    day = args.get("date") or date.today().isoformat()
    subject = (args.get("subject") or "").strip()
    dept = args.get("department") or ""
    section = args.get("section") or ""

    query, params = "SELECT * FROM students WHERE 1=1", []
    if dept:
        query += " AND department = ?"
        params.append(dept)
    if section:
        query += " AND section = ?"
        params.append(section)
    query += " ORDER BY department, section, roll_no"
    students = db.execute(query, params).fetchall()

    aq = ("SELECT student_id, MIN(time) AS time, GROUP_CONCAT(subject, ', ') AS subjects "
          "FROM attendance WHERE date = ?")
    ap = [day]
    if subject:
        aq += " AND subject = ?"
        ap.append(subject)
    aq += " GROUP BY student_id"
    att = {r["student_id"]: r for r in db.execute(aq, ap)}

    rows = []
    for s in students:
        a = att.get(s["id"])
        rows.append({
            "roll_no": s["roll_no"], "name": s["name"],
            "department": s["department"], "section": s["section"],
            "status": "Present" if a else "Absent",
            "time": a["time"] if a else "",
            "subjects": a["subjects"] if a else "",
        })
    present = sum(1 for r in rows if r["status"] == "Present")
    return {
        "rows": rows, "day": day, "subject": subject,
        "department": dept, "section": section,
        "present": present, "absent": len(rows) - present, "total": len(rows),
    }


# ------------------------------------------------------------ auth routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (request.form.get("username") == ADMIN_USER
                and request.form.get("password") == ADMIN_PASS):
            session["user"] = ADMIN_USER
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------------------------------------------------- dashboard
@app.route("/")
@login_required
def dashboard():
    db = get_db()
    today = date.today().isoformat()
    total = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    present_today = db.execute(
        "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today,)
    ).fetchone()[0]

    trend = []
    for i in range(6, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        c = db.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (d,)
        ).fetchone()[0]
        trend.append({"label": d[5:], "count": c,
                      "pct": round(c * 100 / total) if total else 0})

    dept_rows = db.execute(
        """SELECT s.department AS department, COUNT(DISTINCT s.id) AS total,
                  COUNT(DISTINCT a.student_id) AS present
           FROM students s LEFT JOIN attendance a
                ON a.student_id = s.id AND a.date = ?
           GROUP BY s.department ORDER BY s.department""",
        (today,),
    ).fetchall()

    recent = db.execute(
        """SELECT s.name, s.roll_no, s.department, s.section, a.subject, a.time
           FROM attendance a JOIN students s ON s.id = a.student_id
           WHERE a.date = ? ORDER BY a.time DESC LIMIT 15""",
        (today,),
    ).fetchall()

    return render_template(
        "dashboard.html", total=total, present=present_today,
        absent=max(total - present_today, 0),
        percent=round(present_today * 100 / total) if total else 0,
        trend=trend, dept_rows=dept_rows, recent=recent, today=today,
    )


# --------------------------------------------------------------- students
@app.route("/students")
@login_required
def students():
    db = get_db()
    rows = db.execute(
        "SELECT id, roll_no, name, department, section, photo FROM students "
        "ORDER BY department, section, roll_no"
    ).fetchall()
    departments, _, _ = get_filter_options(db)
    return render_template("students.html", students=rows, departments=departments)


@app.route("/students/add", methods=["POST"])
@login_required
def add_student():
    roll = request.form.get("roll_no", "").strip()
    name = request.form.get("name", "").strip()
    dept = request.form.get("department", "").strip().upper()
    section = request.form.get("section", "").strip().upper()
    if not (roll and name and dept and section):
        flash("All fields are required.", "error")
        return redirect(url_for("students"))

    file = request.files.get("photo_file")
    photo_data = request.form.get("photo_data", "")
    try:
        if file and file.filename:
            img = ImageOps.exif_transpose(Image.open(file.stream)).convert("RGB")
        elif photo_data:
            img = decode_image(photo_data)
        else:
            flash("Please upload a photo or capture one with the webcam.", "error")
            return redirect(url_for("students"))
    except Exception:
        flash("Could not read that image.", "error")
        return redirect(url_for("students"))

    img.thumbnail((1000, 1000))
    encoding, count = single_face_encoding(img)
    if encoding is None:
        flash(f"Found {count} faces in the photo. Use a clear photo with exactly one face.", "error")
        return redirect(url_for("students"))

    db = get_db()
    if db.execute("SELECT 1 FROM students WHERE roll_no = ?", (roll,)).fetchone():
        flash(f"Roll number {roll} already exists.", "error")
        return redirect(url_for("students"))

    filename = (secure_filename(roll) or "student") + ".jpg"
    img.save(os.path.join(PHOTO_DIR, filename), "JPEG")
    db.execute(
        "INSERT INTO students (roll_no, name, department, section, photo, encoding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (roll, name, dept, section, filename, encoding.astype(np.float64).tobytes()),
    )
    db.commit()
    reload_cache()
    flash(f"Student {name} ({roll}) added.", "success")
    return redirect(url_for("students"))


@app.route("/students/<int:sid>/delete", methods=["POST"])
@login_required
def delete_student(sid):
    db = get_db()
    row = db.execute("SELECT photo FROM students WHERE id = ?", (sid,)).fetchone()
    if row:
        path = os.path.join(PHOTO_DIR, row["photo"] or "")
        if row["photo"] and os.path.exists(path):
            os.remove(path)
        db.execute("DELETE FROM attendance WHERE student_id = ?", (sid,))
        db.execute("DELETE FROM students WHERE id = ?", (sid,))
        db.commit()
        reload_cache()
        flash("Student deleted.", "success")
    return redirect(url_for("students"))


@app.route("/photos/<path:filename>")
@login_required
def photo(filename):
    return send_from_directory(PHOTO_DIR, filename)


# ------------------------------------------------------- live attendance
@app.route("/live")
@login_required
def live():
    db = get_db()
    departments, sections, _ = get_filter_options(db)
    students = db.execute(
        "SELECT id, roll_no, name, department, section FROM students "
        "ORDER BY department, section, roll_no"
    ).fetchall()
    return render_template(
        "live.html", departments=departments, sections=sections,
        students=students, today=date.today().isoformat(),
    )


@app.route("/attendance/add", methods=["POST"])
@login_required
def add_attendance():
    try:
        student_id = int(request.form.get("student_id", ""))
    except ValueError:
        flash("Choose a valid student.", "error")
        return redirect(url_for("live"))

    attendance_day = request.form.get("date", "")
    try:
        attendance_day = date.fromisoformat(attendance_day).isoformat()
    except ValueError:
        flash("Choose a valid attendance date.", "error")
        return redirect(url_for("live"))

    subject = (request.form.get("subject") or "").strip() or "General"
    db = get_db()
    student = db.execute(
        "SELECT id, name FROM students WHERE id = ?", (student_id,)
    ).fetchone()
    if student is None:
        flash("That student does not exist. Refresh and choose a student.", "error")
        return redirect(url_for("live"))

    now = datetime.now()
    cur = db.execute(
        "INSERT OR IGNORE INTO attendance (student_id, date, time, subject) "
        "VALUES (?, ?, ?, ?)",
        (student_id, attendance_day, now.strftime("%H:%M:%S"), subject),
    )
    db.commit()
    if cur.rowcount == 1:
        flash(f"Attendance added for {student['name']} ({subject}) on {attendance_day}.",
              "success")
    else:
        flash(f"{student['name']} already has attendance for {subject} on {attendance_day}.",
              "error")
    return redirect(url_for("live"))


@app.route("/api/summary")
@login_required
def api_summary():
    subject = (request.args.get("subject") or "General").strip() or "General"
    candidates = get_candidates(request.args.get("department", ""),
                                request.args.get("section", ""))
    return jsonify(build_summary(get_db(), candidates, subject, date.today().isoformat()))


@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    payload = request.get_json(force=True)
    subject = (payload.get("subject") or "General").strip() or "General"
    candidates = get_candidates(payload.get("department", ""), payload.get("section", ""))

    img = decode_image(payload["image"])
    arr = np.array(img)
    locations = face_recognition.face_locations(arr)
    encodings = face_recognition.face_encodings(arr, locations)
    known = np.array([s["encoding"] for s in candidates]) if candidates else None

    db = get_db()
    now = datetime.now()
    today = now.date().isoformat()
    seen, faces = set(), []

    for loc, enc in zip(locations, encodings):
        face = {"box": [int(v) for v in loc], "name": "Unknown", "roll_no": "", "status": "unknown"}
        if known is not None:
            dist = face_recognition.face_distance(known, enc)
            best = int(np.argmin(dist))
            if dist[best] <= TOLERANCE:
                s = candidates[best]
                face.update(name=s["name"], roll_no=s["roll_no"], status="present")
                if s["id"] not in seen:
                    seen.add(s["id"])
                    cur = db.execute(
                        "INSERT OR IGNORE INTO attendance (student_id, date, time, subject) "
                        "VALUES (?, ?, ?, ?)",
                        (s["id"], today, now.strftime("%H:%M:%S"), subject),
                    )
                    if cur.rowcount == 1:
                        face["status"] = "new"
        faces.append(face)
    db.commit()

    return jsonify(faces=faces, summary=build_summary(db, candidates, subject, today))


# ----------------------------------------------------------------- reports
@app.route("/report")
@login_required
def report():
    db = get_db()
    departments, sections, subjects = get_filter_options(db)
    data = build_report(request.args)
    return render_template("report.html", departments=departments,
                           sections=sections, subjects=subjects, **data)


@app.route("/report/export")
@login_required
def report_export():
    data = build_report(request.args)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Date", "Roll No", "Name", "Department", "Section",
                     "Status", "First Seen", "Subjects"])
    for r in data["rows"]:
        writer.writerow([data["day"], r["roll_no"], r["name"], r["department"],
                         r["section"], r["status"], r["time"], r["subjects"]])
    return Response(
        out.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{data['day']}.csv"},
    )


# --------------------------------------------------------------- templates
STYLE = """
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f1f5f9;color:#0f172a}
.layout{display:flex;min-height:100vh}
aside{width:220px;background:#0f172a;color:#cbd5e1;padding:20px 14px;position:sticky;top:0;height:100vh}
aside h1{font-size:18px;color:#fff;margin:0 0 24px}
aside a{display:block;color:#cbd5e1;text-decoration:none;padding:10px 12px;border-radius:8px;margin-bottom:4px}
aside a:hover,aside a.active{background:#1e293b;color:#fff}
main{flex:1;padding:24px;max-width:1200px}
h2{margin:0 0 16px}h3{margin:0 0 12px}
.card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:18px}
.stat{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.stat b{display:block;font-size:30px}.stat span{color:#64748b;font-size:13px}
.grid2{display:grid;grid-template-columns:2fr 1fr;gap:18px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}aside{display:none}}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:9px 8px;text-align:left;border-bottom:1px solid #e2e8f0}
th{color:#475569;font-size:12px;text-transform:uppercase}
input,select{padding:9px 10px;border:1px solid #cbd5e1;border-radius:8px;font-size:14px;width:100%}
label{display:block;font-size:13px;color:#475569;margin:10px 0 4px}
.btn{background:#2563eb;color:#fff;border:0;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px;text-decoration:none;display:inline-block}
.btn:hover{background:#1d4ed8}.btn.secondary{background:#475569}.btn.danger{background:#dc2626;padding:6px 10px}
.btn.green{background:#16a34a}.btn:disabled{opacity:.5;cursor:not-allowed}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;align-items:end;margin-bottom:12px}
.flash{padding:10px 14px;border-radius:8px;margin-bottom:14px;font-size:14px}
.flash.success{background:#dcfce7;color:#166534}.flash.error{background:#fee2e2;color:#991b1b}
.muted{color:#64748b;font-size:13px}
.badge{padding:3px 9px;border-radius:99px;font-size:12px;font-weight:600}
.badge.Present{background:#dcfce7;color:#166534}.badge.Absent{background:#fee2e2;color:#991b1b}
.bars{display:flex;gap:10px;align-items:flex-end;height:160px}
.bar{flex:1;text-align:center;font-size:12px;color:#475569}
.bar div{background:#2563eb;border-radius:6px 6px 0 0;min-height:3px}
.thumb{width:44px;height:44px;border-radius:50%;object-fit:cover}
.video-wrap{position:relative;width:100%;background:#000;border-radius:10px;overflow:hidden}
.video-wrap video{width:100%;display:block}
.video-wrap canvas{position:absolute;left:0;top:0;width:100%;height:100%}
#plist{list-style:none;padding:0;margin:0;max-height:480px;overflow:auto}
#plist li{padding:8px 4px;border-bottom:1px solid #e2e8f0;font-size:14px;display:flex;justify-content:space-between}
.cam-row{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}
.cam-row video,.cam-row img{width:240px;border-radius:8px;background:#e2e8f0}
.login-body{display:flex;align-items:center;justify-content:center;min-height:100vh;background:#0f172a}
.login-card{background:#fff;padding:30px;border-radius:14px;width:340px}
.login-card .btn{width:100%;margin-top:18px}
.progress{background:#e2e8f0;border-radius:99px;height:8px;overflow:hidden}
.progress div{background:#16a34a;height:100%}
"""

TEMPLATES = {
"base.html": """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}Smart Attendance{% endblock %}</title>
<style>{{ STYLE|safe }}</style></head>
<body><div class="layout">
<aside>
  <h1>Smart Attendance</h1>
  <a href="{{ url_for('dashboard') }}" class="{{ 'active' if request.endpoint=='dashboard' }}">Dashboard</a>
  <a href="{{ url_for('live') }}" class="{{ 'active' if request.endpoint=='live' }}">Live Attendance</a>
  <a href="{{ url_for('students') }}" class="{{ 'active' if request.endpoint=='students' }}">Students</a>
  <a href="{{ url_for('report') }}" class="{{ 'active' if request.endpoint=='report' }}">Reports</a>
  <a href="{{ url_for('logout') }}">Logout</a>
</aside>
<main>
{% for cat, msg in get_flashed_messages(with_categories=true) %}
  <div class="flash {{ cat }}">{{ msg }}</div>
{% endfor %}
{% block content %}{% endblock %}
</main></div></body></html>""",

"login.html": """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login</title><style>{{ STYLE|safe }}</style></head>
<body class="login-body">
<form method="post" class="login-card">
  <h2>Smart Attendance</h2>
  <p class="muted">College admin login</p>
  {% for cat, msg in get_flashed_messages(with_categories=true) %}
    <div class="flash {{ cat }}">{{ msg }}</div>
  {% endfor %}
  <label>Username</label><input name="username" required autofocus>
  <label>Password</label><input type="password" name="password" required>
  <button class="btn" type="submit">Sign in</button>
</form></body></html>""",

"dashboard.html": """{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<h2>Dashboard <span class="muted">{{ today }}</span></h2>
<div class="stats">
  <div class="stat"><b>{{ total }}</b><span>Total students</span></div>
  <div class="stat"><b style="color:#16a34a">{{ present }}</b><span>Present today</span></div>
  <div class="stat"><b style="color:#dc2626">{{ absent }}</b><span>Absent today</span></div>
  <div class="stat"><b>{{ percent }}%</b><span>Attendance today</span></div>
</div>
<div class="grid2">
  <div class="card"><h3>Last 7 days (students present)</h3>
    <div class="bars">
      {% for t in trend %}
      <div class="bar"><small>{{ t.count }}</small>
        <div style="height:{{ t.pct * 1.2 }}px"></div>{{ t.label }}</div>
      {% endfor %}
    </div>
  </div>
  <div class="card"><h3>Department-wise today</h3>
    {% for d in dept_rows %}
      <div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;font-size:14px">
        <span>{{ d.department }}</span><span>{{ d.present }} / {{ d.total }}</span></div>
        <div class="progress"><div style="width:{{ (d.present * 100 / d.total) if d.total else 0 }}%"></div></div></div>
    {% else %}<p class="muted">No students yet.</p>{% endfor %}
  </div>
</div>
<div class="card"><h3>Recent attendance today</h3>
<table><tr><th>Time</th><th>Roll No</th><th>Name</th><th>Dept</th><th>Section</th><th>Subject</th></tr>
{% for r in recent %}
<tr><td>{{ r.time }}</td><td>{{ r.roll_no }}</td><td>{{ r.name }}</td><td>{{ r.department }}</td><td>{{ r.section }}</td><td>{{ r.subject }}</td></tr>
{% else %}<tr><td colspan="6" class="muted">No attendance marked yet today.</td></tr>{% endfor %}
</table></div>
{% endblock %}""",

"students.html": """{% extends "base.html" %}
{% block title %}Students{% endblock %}
{% block content %}
<h2>Students</h2>
<div class="card"><h3>Register new student</h3>
<form method="post" action="{{ url_for('add_student') }}" enctype="multipart/form-data">
  <div class="form-grid">
    <div><label>Roll No</label><input name="roll_no" required></div>
    <div><label>Name</label><input name="name" required></div>
    <div><label>Department</label><input name="department" list="depts" placeholder="CSE" required></div>
    <div><label>Section / Year</label><input name="section" placeholder="A" required></div>
  </div>
  <datalist id="depts">{% for d in departments %}<option value="{{ d }}">{% endfor %}</datalist>
  <label>Photo (clear, front-facing, only one person)</label>
  <input type="file" name="photo_file" accept="image/*">
  <p class="muted">Or capture with the webcam:</p>
  <div class="cam-row"><video id="cam" autoplay playsinline muted></video><img id="preview" alt=""></div>
  <input type="hidden" name="photo_data" id="photo_data">
  <button type="button" class="btn secondary" id="camBtn">Start camera</button>
  <button type="button" class="btn secondary" id="snapBtn" disabled>Capture photo</button>
  <button class="btn green" type="submit">Save student</button>
</form></div>

<div class="card"><h3>All students ({{ students|length }})</h3>
<table><tr><th></th><th>Roll No</th><th>Name</th><th>Dept</th><th>Section</th><th></th></tr>
{% for s in students %}
<tr><td><img class="thumb" src="{{ url_for('photo', filename=s.photo) }}" alt=""></td>
<td>{{ s.roll_no }}</td><td>{{ s.name }}</td><td>{{ s.department }}</td><td>{{ s.section }}</td>
<td><form method="post" action="{{ url_for('delete_student', sid=s.id) }}"
     onsubmit="return confirm('Delete this student and their attendance?')">
  <button class="btn danger">Delete</button></form></td></tr>
{% else %}<tr><td colspan="6" class="muted">No students registered yet.</td></tr>{% endfor %}
</table></div>

<script>
const cam = document.getElementById("cam");
const snapBtn = document.getElementById("snapBtn");
let stream = null;
document.getElementById("camBtn").onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cam.srcObject = stream;
    snapBtn.disabled = false;
  } catch (e) { alert("Cannot access camera: " + e.message); }
};
snapBtn.onclick = () => {
  const c = document.createElement("canvas");
  c.width = cam.videoWidth; c.height = cam.videoHeight;
  c.getContext("2d").drawImage(cam, 0, 0);
  const data = c.toDataURL("image/jpeg", 0.92);
  document.getElementById("photo_data").value = data;
  document.getElementById("preview").src = data;
};
</script>
{% endblock %}""",

"live.html": """{% extends "base.html" %}
{% block title %}Live Attendance{% endblock %}
{% block content %}
<h2>Live Group Attendance</h2>
<div class="card">
  <h3>Add attendance manually</h3>
  {% if students %}
  <form method="post" action="{{ url_for('add_attendance') }}">
    <div class="form-grid">
      <div><label>Student</label>
        <select name="student_id" required>
          <option value="">Choose a student</option>
          {% for s in students %}
          <option value="{{ s.id }}">{{ s.roll_no }} - {{ s.name }} ({{ s.department }} / {{ s.section }})</option>
          {% endfor %}
        </select></div>
      <div><label>Date</label><input type="date" name="date" value="{{ today }}" required></div>
      <div><label>Subject / Period</label><input name="subject" value="General" required></div>
    </div>
    <button class="btn green" type="submit">Add attendance</button>
  </form>
  {% else %}
  <p class="muted">Register a student first, then you can add attendance manually or use the camera.</p>
  {% endif %}
</div>
<div class="grid2">
  <div class="card">
    <div class="controls">
      <div><label>Subject / Period</label><input id="subject" value="General"></div>
      <div><label>Department</label>
        <select id="department"><option value="">All</option>
        {% for d in departments %}<option>{{ d }}</option>{% endfor %}</select></div>
      <div><label>Section</label>
        <select id="section"><option value="">All</option>
        {% for s in sections %}<option>{{ s }}</option>{% endfor %}</select></div>
      <div><button class="btn green" id="startBtn">Start camera</button>
           <button class="btn danger" id="stopBtn" disabled>Stop</button></div>
    </div>
    <div class="video-wrap"><video id="video" autoplay playsinline muted></video><canvas id="overlay"></canvas></div>
    <p class="muted" id="status">Camera is off. Point it at the whole class, then press Start.</p>
  </div>
  <div class="card">
    <h3>Present: <span id="pc">0</span> / <span id="tc">0</span></h3>
    <ul id="plist"></ul>
  </div>
</div>

<script>
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const grab = document.createElement("canvas");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
let stream = null, timer = null, busy = false;

function params() {
  return {
    subject: document.getElementById("subject").value.trim() || "General",
    department: document.getElementById("department").value,
    section: document.getElementById("section").value
  };
}

function renderSummary(s) {
  document.getElementById("pc").textContent = s.present_count;
  document.getElementById("tc").textContent = s.total;
  const ul = document.getElementById("plist");
  ul.innerHTML = "";
  s.present.slice().reverse().forEach(p => {
    const li = document.createElement("li");
    const a = document.createElement("span");
    a.textContent = p.name + " (" + p.roll_no + ")";
    const b = document.createElement("span");
    b.className = "muted"; b.textContent = p.time;
    li.appendChild(a); li.appendChild(b); ul.appendChild(li);
  });
}

async function loadSummary() {
  const q = new URLSearchParams(params()).toString();
  const res = await fetch("/api/summary?" + q);
  if (res.ok) {
    const summary = await res.json();
    renderSummary(summary);
    if (summary.total === 0) {
      statusEl.textContent = "No students registered for these filters. Add a student with a clear face photo first.";
    }
  }
}

function drawFaces(faces) {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.lineWidth = 3; ctx.font = "bold 16px Arial";
  faces.forEach(f => {
    const [top, right, bottom, left] = f.box;
    const color = f.status === "new" ? "#16a34a" : (f.status === "present" ? "#2563eb" : "#dc2626");
    ctx.strokeStyle = color;
    ctx.strokeRect(left, top, right - left, bottom - top);
    const label = f.status === "unknown" ? "Unknown" : f.name;
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = color;
    ctx.fillRect(left, Math.max(top - 24, 0), w, 24);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, left + 5, Math.max(top - 7, 17));
  });
}

async function tick() {
  if (busy || !video.videoWidth) return;
  busy = true;
  const w = 640, h = Math.round(video.videoHeight * w / video.videoWidth);
  grab.width = w; grab.height = h;
  grab.getContext("2d").drawImage(video, 0, 0, w, h);
  overlay.width = w; overlay.height = h;
  try {
    const body = Object.assign({ image: grab.toDataURL("image/jpeg", 0.85) }, params());
    const res = await fetch("/api/recognize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (res.status === 401) { location.href = "/login"; return; }
    const data = await res.json();
    drawFaces(data.faces);
    renderSummary(data.summary);
    statusEl.textContent = data.summary.total === 0
      ? "No students registered for these filters. Add a student with a clear face photo first."
      : data.faces.length + " face(s) detected";
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
  } finally { busy = false; }
}

startBtn.onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia(
      { video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) { statusEl.textContent = "Cannot access camera: " + e.message; return; }
  video.srcObject = stream;
  await video.play();
  startBtn.disabled = true; stopBtn.disabled = false;
  statusEl.textContent = "Scanning...";
  timer = setInterval(tick, 1500);
};

stopBtn.onclick = () => {
  clearInterval(timer);
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false; stopBtn.disabled = true;
  statusEl.textContent = "Camera stopped.";
};

["subject", "department", "section"].forEach(id =>
  document.getElementById(id).addEventListener("change", loadSummary));
loadSummary();
</script>
{% endblock %}""",

"report.html": """{% extends "base.html" %}
{% block title %}Reports{% endblock %}
{% block content %}
<h2>Attendance Report</h2>
<div class="card">
<form method="get" class="controls">
  <div><label>Date</label><input type="date" name="date" value="{{ day }}"></div>
  <div><label>Subject</label>
    <select name="subject"><option value="">All subjects</option>
    {% for s in subjects %}<option {{ 'selected' if s==subject }}>{{ s }}</option>{% endfor %}</select></div>
  <div><label>Department</label>
    <select name="department"><option value="">All</option>
    {% for d in departments %}<option {{ 'selected' if d==department }}>{{ d }}</option>{% endfor %}</select></div>
  <div><label>Section</label>
    <select name="section"><option value="">All</option>
    {% for s in sections %}<option {{ 'selected' if s==section }}>{{ s }}</option>{% endfor %}</select></div>
  <div><button class="btn" type="submit">Filter</button>
  <a class="btn secondary" href="{{ url_for('report_export', date=day, subject=subject, department=department, section=section) }}">Export CSV</a></div>
</form></div>

<div class="stats">
  <div class="stat"><b>{{ total }}</b><span>Students</span></div>
  <div class="stat"><b style="color:#16a34a">{{ present }}</b><span>Present</span></div>
  <div class="stat"><b style="color:#dc2626">{{ absent }}</b><span>Absent</span></div>
  <div class="stat"><b>{{ (present * 100 / total)|round|int if total else 0 }}%</b><span>Attendance</span></div>
</div>

<div class="card">
<table><tr><th>Roll No</th><th>Name</th><th>Dept</th><th>Section</th><th>Status</th><th>First seen</th><th>Subjects</th></tr>
{% for r in rows %}
<tr><td>{{ r.roll_no }}</td><td>{{ r.name }}</td><td>{{ r.department }}</td><td>{{ r.section }}</td>
<td><span class="badge {{ r.status }}">{{ r.status }}</span></td><td>{{ r.time }}</td><td>{{ r.subjects }}</td></tr>
{% else %}<tr><td colspan="7" class="muted">No students match these filters.</td></tr>{% endfor %}
</table></div>
{% endblock %}""",
}

app.jinja_loader = DictLoader(TEMPLATES)


@app.context_processor
def inject_style():
    return {"STYLE": STYLE}


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    init_db()
    reload_cache()
    print("Open http://127.0.0.1:5000  (login: admin / admin123)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)