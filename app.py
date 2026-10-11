import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from models import db, User, Analysis

app = Flask(__name__)
app.config['SECRET_KEY'] = 'safevision-secret-key-2026-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///safevision.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'يجب تسجيل الدخول أولاً'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# تحميل النموذج المدرب
model = YOLO("runs/detect/runs/safevision_train_v4/weights/best.pt")

REQUIRED = {
    "hardhat": "⛑️ Hard Hat",
    "safety vest": "🦺 Safety Vest",
    "gloves": "🧤 Gloves",
    "safety shoes": "👢 Safety Shoes",
    "mask": "😷 Mask"
}
# ==================== Routes ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f'مرحباً {user.username}! 👋', 'success')
            return redirect(url_for('index'))
        else:
            flash('البريد الإلكتروني أو كلمة المرور غير صحيحة', 'danger')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        
        # التحقق من المدخلات
        if password != confirm:
            flash('كلمتا المرور غير متطابقتين', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('البريد الإلكتروني مسجل مسبقاً', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('اسم المستخدم محجوز', 'danger')
            return redirect(url_for('register'))
        
        # إنشاء المستخدم
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        
        flash('تم إنشاء الحساب بنجاح! يمكنك تسجيل الدخول الآن ✨', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('تم تسجيل الخروج بنجاح', 'info')
    return redirect(url_for('login'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/history')
@login_required
def history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    return render_template('history.html', analyses=analyses)
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
     if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return 'No file uploaded', 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # تحليل الصورة
        results = model(filepath, conf=0.1)
        
        detected = set()
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls)].lower()
            if not cls_name.startswith('no-'):
                detected.add(cls_name)
        
        found = [eq for eq in REQUIRED if eq in detected]
        score = (len(found) / len(REQUIRED)) * 100
        
        report = []
        for key, label in REQUIRED.items():
            status = '✅ Detected' if key in detected else '❌ Not Detected'
            report.append(f'{label} — {status}')
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
        results[0].save(output_path)
        
        # حفظ التحليل في قاعدة البيانات
        analysis = Analysis(
            user_id=current_user.id,
            image_path=filepath,
            result_path=output_path,
            safety_score=round(score),
            detected_items=', '.join(detected)
        )
        db.session.add(analysis)
        db.session.commit()
        
        return render_template(
    'index.html',
    report=report,
    score=round(score),
    original='uploads/' + filename,
    result=f'uploads/result_{filename}'
)
    
     return render_template('index.html')


# ==================== Main ====================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print('✅ Database initialized!')
    app.run(debug=False, host='0.0.0.0', port=7860)