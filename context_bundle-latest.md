# Project Context Bundle


- Generated: **2025-08-14 06:21:05Z UTC**
- Commit: `fa0871407ee48525428b7d053e7323eaec32fc16`
- Note: Adjust the list below to include/exclude files. You can add globs too.

## Table of Contents


1. [api_routes.py](#api_routespy)
2. [models.py](#modelspy)
3. [analysis_engine.py](#analysis_enginepy)
4. [astrological_evaluator.py](#astrological_evaluatorpy)
5. [rohini_engine.py](#rohini_enginepy)
6. [orchestration_engine.py](#orchestration_enginepy)
7. [astrological_constants.py](#astrological_constantspy)
8. [db_utils.py](#db_utilspy)
9. [multilingual_strings.py](#multilingual_stringspy)
10. [database_schema.md](#database_schemamd)
11. [akundli_report_FINAL.md](#akundli_report_finalmd)

## Files


### api_routes.py


```python
import logging
import os

from extensions import db
from models import UserChart, User
from flask import Blueprint, jsonify, request, current_app
from engine_adapter import call_generate_full_kundli

from api.schemas import parse_kundli_request, ValidationError
from api.error_codes import ErrorCode

api_bp = Blueprint('api', __name__)

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("T03e_flask")
if not logger.handlers:
    handler = logging.FileHandler("logs/T03e_flask.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'ok': True})

@api_bp.route('/user', methods=['POST'])
def create_user():
    payload = request.get_json()
    if not payload or 'user_id' not in payload:
        return jsonify({"code": "INVALID_REQUEST", "message": "Missing user_id in payload"}), 400
    user_id = payload['user_id']
    if User.query.get(user_id):
        return jsonify({"code": "CONFLICT", "message": f"User with ID {user_id} already exists"}), 409
    try:
        new_user = User(id=user_id)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": f"User {user_id} created successfully"}), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error creating user: {e}", exc_info=True)
        return jsonify({"code": "SERVER_ERROR", "message": "Could not create user"}), 500

@api_bp.route('/kundli/generate', methods=['POST'])
def generate_kundli():
    try:
        parsed = parse_kundli_request(request.get_json())
    except ValidationError as ve:
        logger.error(ve.message)
        return jsonify({"code": ve.code.code, "message": ve.message}), ve.http_status
    except Exception as e:
        logger.exception("Unexpected error while parsing request")
        return jsonify({"code": ErrorCode.SERVER_ERROR.code, "message": "Internal server error"}), ErrorCode.SERVER_ERROR.http_status

    try:
        chart_json = call_generate_full_kundli(parsed.birth_datetime, parsed.lat, parsed.lon, parsed.tz)

        user_chart = UserChart.query.filter_by(user_id=parsed.user_id, relation_type='self').first()
        if user_chart:
            user_chart.birth_data = parsed.original_payload
            user_chart.chart_json = chart_json
            user_chart.birth_datetime = parsed.birth_datetime
            user_chart.tz = parsed.tz
            user_chart.lat = parsed.lat
            user_chart.lon = parsed.lon
        else:
            user_chart = UserChart(
                user_id=parsed.user_id,
                relation_type='self',
                birth_data=parsed.original_payload,
                chart_json=chart_json,
                birth_datetime=parsed.birth_datetime,
                tz=parsed.tz,
                lat=parsed.lat,
                lon=parsed.lon,
            )
            db.session.add(user_chart)
        db.session.commit()

        latest = UserChart.query.filter_by(user_id=parsed.user_id, relation_type='self').order_by(UserChart.updated_at.desc()).first()
        return jsonify({"user_chart_id": latest.id, "chart_json": latest.chart_json}), 200
    except Exception as e:
        db.session.rollback()
        logger.exception("Error generating kundli")
        return jsonify({"code": ErrorCode.SERVER_ERROR.code, "message": "Internal server error"}), ErrorCode.SERVER_ERROR.http_status

@api_bp.route('/user/chart/<int:user_id>', methods=['GET'])
def get_latest_user_chart(user_id):
    try:
        user_chart = UserChart.query.filter_by(user_id=user_id).order_by(UserChart.created_at.desc()).first()
        if user_chart:
            return jsonify({
                "user_id": user_chart.user_id,
                "relation_type": user_chart.relation_type,
                "birth_data": user_chart.birth_data,
                "chart_json": user_chart.chart_json,
                "created_at": user_chart.created_at.isoformat()
            }), 200
        else:
            return jsonify({"code": "NOT_FOUND", "message": "no chart for user"}), 404
    except Exception as e:
        return jsonify({"code": "SERVER_ERROR", "message": str(e)}), 500
```

### models.py


```python
from extensions import db
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone
from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Add other user-related fields as needed, e.g., email, username, etc.

    def __repr__(self):
        return f"<User {self.id}>"

class SocialPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_platform = db.Column(db.String(50), nullable=False)
    unique_post_id = db.Column(db.String(255), unique=True, nullable=False)
    direct_url = db.Column(db.String(500), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    original_timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    analysis_status = db.Column(db.String(50), default='pending', nullable=False)
    sentiment = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f"<SocialPost {self.unique_post_id}>"

class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    term = db.Column(db.String(100), unique=True, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<Keyword {self.term}>"

class DataSource(db.Model):
    __tablename__ = 'data_source'
    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(db.String(50), nullable=False)
    identifier = db.Column(db.String(255), unique=True, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<DataSource {self.platform}:{self.identifier}>"

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(120), unique=True, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    channel_id = db.Column(db.String(120), nullable=False)
    published_at = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(50), default='pending', nullable=False)

import uuid

class UserChart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Foreign key to user table
    relation_type = db.Column(db.Text, nullable=False, default='self')
    birth_data = db.Column(JSONB, nullable=False)
    chart_json = db.Column(JSONB, nullable=False)
    birth_datetime = db.Column(db.TIMESTAMP(timezone=True), nullable=False)
    tz = db.Column(db.Text, nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.TIMESTAMP(timezone=True), server_default=db.func.now(), nullable=True)
    updated_at = db.Column(db.TIMESTAMP(timezone=True), server_default=db.func.now(), onupdate=db.func.now(), nullable=True)

    __table_args__ = (db.UniqueConstraint('user_id', 'relation_type', name='_user_relation_uc'),)

    def __repr__(self):
        return f"<UserChart {self.id} for User {self.user_id}>"

class AdminUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='admin')
    permissions = db.Column(db.String(255), nullable=False, default='all')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<AdminUser {self.email}>'

class AstrologerProfile(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4())) # UUID for PK
    user_chart_id = db.Column(db.Integer, db.ForeignKey('user_chart.id'), nullable=False, unique=True)
    bio = db.Column(db.Text, nullable=True)
    specializations = db.Column(db.ARRAY(db.String), nullable=True) # Text array
    consultation_fee = db.Column(db.Numeric, nullable=True)
    availability = db.Column(db.JSON, nullable=True) # JSONB
    status = db.Column(db.String(50), default='active', nullable=False)

    def __repr__(self):
        return f'<AstrologerProfile {self.user_id}>'

class AstrologerApplication(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4())) # UUID for PK
    user_chart_id = db.Column(db.Integer, db.ForeignKey('user_chart.id'), nullable=False, unique=True)
    application_data = db.Column(db.JSON, nullable=False) # JSONB
    status = db.Column(db.String(50), default='pending', nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f'<AstrologerApplication {self.user_id}>'

class ConsultationHistory(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4())) # UUID for PK
    astrologer_id = db.Column(db.String(36), db.ForeignKey('astrologer_profile.id'), nullable=False)
    client_user_chart_id = db.Column(db.Integer, db.ForeignKey('user_chart.id'), nullable=False)
    date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    duration = db.Column(db.Integer, nullable=False) # in minutes
    total_fee = db.Column(db.Numeric, nullable=False)
    platform_commission = db.Column(db.Numeric, nullable=False)
    astrologer_payout = db.Column(db.Numeric, nullable=False)

    def __repr__(self):
        return f'<ConsultationHistory {self.id}>'

class KnowledgeBaseSystems(db.Model):
    __tablename__ = 'knowledge_base_systems'
    system_name = db.Column(db.String(255), primary_key=True)
    data_en = db.Column(JSONB, nullable=False)
    data_hi = db.Column(JSONB, nullable=False)
    data_hinglish = db.Column(JSONB, nullable=False)

    def __repr__(self):
        return f'<KnowledgeBaseSystems {self.system_name}>'

class KnowledgeBaseInterpretations(db.Model):
    __tablename__ = 'knowledge_base_interpretations'
    category = db.Column(db.Text, nullable=False)
    key = db.Column(db.Text, nullable=False)
    data_en = db.Column(JSONB, nullable=False)
    data_hi = db.Column(JSONB, nullable=False)
    data_hinglish = db.Column(JSONB, nullable=False)

    __table_args__ = (db.PrimaryKeyConstraint('category', 'key', name='pk_knowledge_base_interpretations'),)

    def __repr__(self):
        return f'<KnowledgeBaseInterpretations {self.category} - {self.key}>'

class LLMPrompts(db.Model):
    __tablename__ = 'llm_prompts'
    prompt_id = db.Column(db.String(255), primary_key=True)
    trigger_type = db.Column(db.Text, nullable=False)
    template_en = db.Column(db.Text, nullable=False)
    template_hi = db.Column(db.Text, nullable=False)
    template_hinglish = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<LLMPrompts {self.prompt_id}>'
```

### analysis_engine.py


```python
# analysis_engine.py (core analysis module)

import math
from datetime import datetime
import astrological_constants as ac
import astrological_evaluator as evaluator
import db_utils  # Still needed if you fetch yoga rules from KB
import rohini_engine  # For transit positions if needed

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def _norm_angle(lon: float) -> float:
    return lon % 360.0

def _min_sep(a: float, b: float) -> float:
    """Minimal angular separation in degrees (wrap-aware)."""
    a = _norm_angle(a); b = _norm_angle(b)
    d = abs(a - b)
    return min(d, 360.0 - d)

def _arc_contains_forward(start: float, end: float, x: float, *, inclusive_edges: bool = False, tol: float = 1e-6) -> bool:
    """
    True if longitude x lies on the directed arc start -> end (forward, modulo 360).
    If inclusive_edges=True, include points equal to start/end (within tol).
    """
    start = _norm_angle(start); end = _norm_angle(end); x = _norm_angle(x)
    span = (end - start) % 360.0
    if abs(span) < tol:
        # Degenerate: nodes identical (shouldn't happen) – treat as 'contains all'
        return True
    pos = (x - start) % 360.0
    if inclusive_edges:
        return -tol <= pos <= span + tol
    return tol < pos < span - tol

# -------------------------------------------------------------------
# Foundational Analysis
# -------------------------------------------------------------------

def get_avakahada_chakra(natal_chart):
    """
    Extract and return foundational Vedic details (nakshatra, gana, nadi, etc.)
    based on Moon and Ascendant from the natal chart.
    """
    moon_data = natal_chart['planets']['moon']
    moon_nakshatra, moon_sign = moon_data['nakshatra'], moon_data['sign']
    gana = next((g for g, naks in ac.GANA_MAP.items() if moon_nakshatra in naks), "Unknown")
    nadi = next((n for n, naks in ac.NADI_MAP.items() if moon_nakshatra in naks), "Unknown")
    return {
        "Nakshatra": moon_nakshatra,
        "Pada (Charan)": moon_data.get('pada', 'N/A'),
        "Rasi (Sign)": moon_sign,
        "Lagna (Ascendant)": natal_chart['ascendant']['sign'],
        "Varna (Function)": ac.VARNA_MAP.get(moon_sign, "N/A"),
        "Vashya (Influence)": ac.VASHYA_MAP.get(moon_sign, "N/A"),
        "Yoni (Nature)": ac.YONI_MAP.get(moon_nakshatra, "N/A"),
        "Gana (Temperament)": gana,
        "Nadi (Constitution)": nadi,
        "Nakshatra Lord": ac.NAKSHATRA_LORDS.get(moon_nakshatra, "N/A"),
    }

def get_planet_avasthas(planet_name, natal_chart):
    """
    Baladi & Deeptadi Avasthas for a planet.
    """
    if planet_name in ["rahu", "ketu"]:
        return {"baladi": "n/a", "deeptadi": "n/a"}
    p = natal_chart['planets'][planet_name]
    sign_type = "odd" if (ac.SIGNS.index(p['sign']) + 1) % 2 != 0 else "even"
    degree_in_sign = p['longitude'] % 30
    baladi = next((state for state, start, end in ac.BALADI_AVASTHA_RANGES[sign_type] if start <= degree_in_sign < end), "unknown")
    deeptadi_map = {
        "Exaltation": "deepta",
        "Own Sign": "svastha",
        "Friendly Sign": "pramudita",
        "Neutral Sign": "shanta",
        "Enemy Sign": "duhkhita",
    }
    deeptadi = deeptadi_map.get(p['dignity'], "unknown")
    return {"baladi": baladi, "deeptadi": deeptadi}

def get_lajjitaadi_avasthas(planet_name, kundli):
    """
    Lajjitaadi Avasthas via conjunctions/dignity.
    """
    if planet_name in ["rahu", "ketu"]:
        return "n/a"

    p = kundli['planets'][planet_name]
    house = p['house']

    # Garvita (Proud)
    if p['dignity'] in ["Exaltation", "Moolatrikona"]:
        return "garvita"

    others = [n for n, d in kundli['planets'].items() if d['house'] == house and n != planet_name]

    # Lajjita (5th + malefic present)
    if house == 5 and any(q in ["sun", "saturn", "mars", "rahu", "ketu"] for q in others):
        return "lajjita"

    # Mudita (with Jupiter or Friendly Sign)
    if 'jupiter' in others or p['dignity'] == "Friendly Sign":
        return "mudita"

    # Kshobhita (with Sun and malefic aspect)
    if 'sun' in others:
        aspects = evaluator.get_aspects_on_house(house, kundli)
        if any(q in aspects for q in ["saturn", "mars", "rahu"]):
            return "kshobhita"

    # Kshudhita (with Saturn or Enemy Sign)
    if 'saturn' in others or p['dignity'] == "Enemy Sign":
        return "kshudhita"

    return "shanta"

# -------------------------------------------------------------------
# Planetary Data Consolidation
# -------------------------------------------------------------------

def get_all_planetary_data(user_chart):
    planetary_findings = []

    for pname, pdata in user_chart['planets'].items():
        finding = {
            "type": "planetary_position",
            "name": f"{pname.capitalize()} Position",
            "key": f"{pname}_position",
            "relevant_planet": pname,
            "relevant_house": pdata['house'],
            "details": pdata.copy(),
        }

        # Avasthas
        av = get_planet_avasthas(pname, user_chart)
        finding['details']['baladi_avastha'] = av.get('baladi')
        finding['details']['deeptadi_avastha'] = av.get('deeptadi')
        finding['details']['lajjitaadi_avastha'] = get_lajjitaadi_avasthas(pname, user_chart)

        # Positional Strength (prefer sidereal cusps)
        cusps = user_chart.get('house_cusps_sidereal') or user_chart.get('house_cusps')
        pos = evaluator.evaluate_planetary_positional_strength(pdata['longitude'], pdata['house'], cusps)
        finding['details']['bhava_positional_strength'] = pos.get('status')

        # Bhavesha Strength Summary
        finding['details']['bhavesha_strength_summary'] = evaluator.evaluate_bhavesha_strength(user_chart, pdata['house'])

        # Shadbala total
        finding['details']['shadbala_total'] = pdata.get('shadbala', {}).get('total')

        # Sarvashtakavarga bindus for the planet's sign
        sav = user_chart.get('ashtakavarga_scores', {}).get('sarvashtakavarga', {})
        finding['details']['sarvashtakavarga_score_for_house'] = sav.get(pdata['sign'], 0)

        # Vargottama
        finding['details']['vargottama_status'] = (pdata.get('d9_sign') == pdata.get('sign'))

        # Pre-written insight keys (for KB)
        finding['pre_written_insight_keys'] = [
            {"category": "graha-", "key": pname},
            {"category": "graha_in_bhava", "key": f"{pname}_in_{pdata['house']}th_house"},
            {"category": "graha_in_rashi", "key": f"{pname}_in_{pdata['sign']}"},
            {"category": "planet_in_navamsa", "key": f"{pname}_in_{pdata['d9_sign']}"},
            {"category": "planetary_avastha", "key": f"{pname}_baladi_{finding['details']['baladi_avastha']}"} if finding['details'].get('baladi_avastha') != 'n/a' else None,
            {"category": "planetary_avastha", "key": f"{pname}_deeptadi_{finding['details']['deeptadi_avastha']}"} if finding['details'].get('deeptadi_avastha') != 'n/a' else None,
            {"category": "planetary_avastha", "key": f"{pname}_lajjitaadi_{finding['details']['lajjitaadi_avastha']}"} if finding['details'].get('lajjitaadi_avastha') != 'n/a' else None,
            {"category": "planetary_dignity", "key": f"{pname}_{finding['details']['dignity'].lower().replace(' ', '_')}"},
            {"category": "planetary_status", "key": f"{pname}_{finding['details']['status'].lower().split(' ')[0]}"},
        ]
        finding['pre_written_insight_keys'] = [k for k in finding['pre_written_insight_keys'] if k]
        planetary_findings.append(finding)

    return planetary_findings

# -------------------------------------------------------------------
# Doshas
# -------------------------------------------------------------------

def check_mangal_dosha(kundli):
    """Kuja Dosha with severity and modifiers."""
    mars = kundli['planets']['mars']
    mars_house = mars['house']
    dosha_houses = [1, 4, 7, 8, 12]
    is_present = mars_house in dosha_houses

    if not is_present:
        return {"type": "dosha", "name": "Mangal Dosha", "key": "mangal_dosha_not_present", "is_present": False}

    severity = 5
    mitigators, amplifiers = [], []

    # Mitigations
    if mars['dignity'] in ["Exaltation", "Own Sign", "Moolatrikona"]:
        severity -= 3
        mitigators.append(f"Mars is strong in its {mars['dignity']}.")

    dispositor_lord = ac.SIGN_LORDS[mars['sign']]
    dispositor = kundli['planets'].get(dispositor_lord, {})
    if dispositor.get('dignity') in ["Exaltation", "Own Sign"]:
        severity -= 2
        mitigators.append(f"Mars' dispositor, {dispositor_lord}, is strong in {dispositor.get('dignity','')}.")

    aspects_on_mars_house = evaluator.get_aspects_on_house(mars_house, kundli)
    if 'jupiter' in aspects_on_mars_house:
        severity -= 4
        mitigators.append("Receives a powerful benefic aspect from Jupiter.")
    if 'venus' in aspects_on_mars_house:
        severity -= 2
        mitigators.append("Receives a benefic aspect from Venus.")

    # Amplifiers
    if mars['dignity'] in ["Debilitation", "Enemy Sign"]:
        severity += 3
        amplifiers.append(f"Mars is weak in an {mars['dignity']}.")
    for pname, pdata in kundli['planets'].items():
        if pname in ac.MALEFICS and pdata['house'] == mars_house and pname != 'mars':
            severity += 2
            amplifiers.append(f"Mars is conjunct with malefic {pname}.")
    if 'saturn' in aspects_on_mars_house:
        severity += 2
        amplifiers.append("Receives a malefic aspect from Saturn.")

    if severity <= 1: status = "cancelled"
    elif severity <= 3: status = "low"
    elif severity <= 6: status = "moderate"
    else: status = "high"

    return {
        "type": "dosha",
        "name": "Mangal Dosha",
        "key": f"mangal_dosha_{status}",
        "is_present": True,
        "status": status,
        "severity_score": severity,
        "mitigating_factors": mitigators,
        "amplifying_factors": amplifiers,
        "relevant_planet": "mars",
        "relevant_house": mars_house,
        "pre_written_insight_keys": [
            {"category": "dosha_severity", "key": f"mangal_dosha_{status}"}
        ]
    }

# STRICT Kaal Sarp Dosha (empty semicircle rule)
def check_kaal_sarp_dosha(kundli):
    """
    STRICT definition:
    Kaal Sarp is PRESENT iff all 7 classical planets lie strictly within one node-defined semicircle
    (Rahu→Ketu OR Ketu→Rahu). Otherwise, NOT present.
    """
    planets = kundli.get('planets', {})
    try:
        rahu = _norm_angle(planets['rahu']['longitude'])
        ketu = _norm_angle(planets['ketu']['longitude'])
    except Exception:
        return {"type": "dosha", "name": "Kaal Sarp Dosha", "key": "kaal_sarp_dosha_not_present", "is_present": False}

    classical_lons = []
    for key in ("sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn"):
        p = planets.get(key)
        if not p or p.get('longitude') is None:
            return {"type": "dosha", "name": "Kaal Sarp Dosha", "key": "kaal_sarp_dosha_not_present", "is_present": False}
        classical_lons.append(_norm_angle(p['longitude']))

    # Count planets strictly inside each arc (endpoints excluded)
    in_A = sum(1 for lon in classical_lons if _arc_contains_forward(rahu, ketu, lon, inclusive_edges=False))
    in_B = sum(1 for lon in classical_lons if _arc_contains_forward(ketu, rahu, lon, inclusive_edges=False))

    present = (in_A == 7) or (in_B == 7)
    if not present:
        return {"type": "dosha", "name": "Kaal Sarp Dosha", "key": "kaal_sarp_dosha_not_present", "is_present": False}

    # If present (strict), basic severity and optional factors (YOUR BLOCK KEPT)
    severity = 6
    mitigators, amplifiers = []

    rahu_house = planets['rahu']['house']
    aspects_on_rahu = evaluator.get_aspects_on_house(rahu_house, kundli)
    if 'jupiter' in aspects_on_rahu or 'venus' in aspects_on_rahu:
        severity -= 3
        mitigators.append("Rahu is aspected by a powerful benefic like Jupiter or Venus.")

    if rahu_house in [1, 7]:
        severity += 2
        amplifiers.append("The axis falls on the 1/7 axis, intensifying the effect on self and relationships.")

    if (planets['sun']['house'] == rahu_house) or \
       (planets['moon']['house'] == rahu_house) or \
       (planets['sun']['house'] == planets['ketu']['house']) or \
       (planets['moon']['house'] == planets['ketu']['house']):
        severity += 2
        amplifiers.append("A luminary (Sun or Moon) is conjunct with Rahu/Ketu.")

    if severity <= 2: status = "cancelled_very_low"
    elif severity <= 5: status = "moderate"
    else: status = "high"

    return {
        "type": "dosha",
        "name": "Kaal Sarp Dosha",
        "key": f"kaal_sarp_dosha_{status}",
        "is_present": True,
        "status": status,
        "severity_score": severity,
        "mitigating_factors": mitigators,
        "amplifying_factors": amplifiers,
        "pre_written_insight_keys": [
            {"category": "dosha_severity", "key": f"kaal_sarp_dosha_{status}"}
        ]
    }

def check_gandmool_dosha(kundli):
    """Gandmool Dosha with basic modifiers."""
    moon_nakshatra = kundli['planets']['moon']['nakshatra']
    is_present = moon_nakshatra in ac.GANDMOOL_NAKSHATRAS

    if not is_present:
        return {"type": "dosha", "name": "Gandmool Dosha", "key": "gandmool_dosha_not_present", "is_present": False}

    severity = 5
    mitigators, amplifiers = [], []

    moon_house = kundli['planets']['moon']['house']
    aspects_on_moon_house = evaluator.get_aspects_on_house(moon_house, kundli)
    if 'jupiter' in aspects_on_moon_house:
        severity -= 3
        mitigators.append("Moon receives a benefic aspect from Jupiter.")
    nak_lord = ac.NAKSHATRA_LORDS.get(moon_nakshatra, "")
    if nak_lord and nak_lord in kundli['planets'] and kundli['planets'][nak_lord]['dignity'] in ["Exaltation", "Own Sign"]:
        severity -= 2
        mitigators.append(f"Nakshatra lord {nak_lord} is very strong.")
    if 'saturn' in aspects_on_moon_house or 'mars' in aspects_on_moon_house:
        severity += 2
        amplifiers.append("Moon is afflicted by Saturn or Mars.")

    if severity <= 2: status = "low"
    elif severity <= 5: status = "moderate"
    else: status = "high"

    return {
        "type": "dosha",
        "name": "Gandmool Dosha",
        "key": f"gandmool_dosha_{status}",
        "is_present": True,
        "status": status,
        "severity_score": severity,
        "mitigating_factors": mitigators,
        "amplifying_factors": amplifiers,
        "relevant_planet": "moon",
        "relevant_nakshatra": moon_nakshatra,
        "pre_written_insight_keys": [
            {"category": "dosha_severity", "key": f"gandmool_dosha_{status}"}
        ]
    }

def identify_all_doshas(kundli):
    """Master dosha list (STRICT Kaal Sarp is used)."""
    return [
        check_mangal_dosha(kundli),
        check_kaal_sarp_dosha(kundli),  # strict version above
        check_gandmool_dosha(kundli),
    ]

# -------------------------------------------------------------------
# Yogas & Conjunctions
# -------------------------------------------------------------------

def identify_all_yogas(user_chart):
    yogas_found = []

    # Kemadruma (handled via evaluator)
    kem = evaluator.check_kemadruma_yoga(user_chart)
    if kem["status"]:
        yogas_found.append({
            "type": "yoga",
            "name": "Kemadruma Yoga",
            "key": f"kemadruma_yoga_{kem['type'].lower()}",
            "is_present": True,
            "status": kem['type'],
            "cancellation_reasons": [r for r in kem.get('cancellation_reasons', [])],
            "relevant_planets": ["moon"],
            "relevant_houses": [user_chart['planets']['moon']['house']],
            "pre_written_insight_keys": [
                {"category": "yoga_interpretation", "key": f"kemadruma_yoga_{kem['type'].lower()}"}
            ] + ([{"category": "kemadruma_cancellation_reasons", "key": r} for r in kem.get('cancellation_reasons', [])] if kem.get('cancellation_reasons') else [])
        })

    # Neecha Bhanga Raja Yoga (if any planet debilitated)
    for pname, pdata in user_chart['planets'].items():
        if pdata['dignity'] == 'Debilitation':
            reasons = evaluator.check_neecha_bhanga(pname, user_chart)
            if reasons:
                yogas_found.append({
                    "type": "yoga",
                    "name": "Neecha Bhanga Raja Yoga",
                    "key": "neecha_bhanga_raja_yoga",
                    "is_present": True,
                    "relevant_planet": pname,
                    "cancellation_reasons": [r['key'] for r in reasons],
                    "relevant_houses": [pdata['house']],
                    "pre_written_insight_keys": [
                        {"category": "raja_yoga", "key": "neecha_bhanga_raja_yoga"}
                    ] + [{"category": "neecha_bhanga_reasons", "key": r['key']} for r in reasons]
                })

    # Gajakesari (Moon-Jupiter Kendra)
    moon_house = user_chart['planets']['moon']['house']
    jup_house = user_chart['planets']['jupiter']['house']
    if jup_house in [(moon_house + i - 1) % 12 + 1 for i in [1, 4, 7, 10]]:
        yogas_found.append({
            "type": "yoga",
            "name": "Gajakesari Yoga",
            "key": "gajakesari_yoga",
            "relevant_planets": ["jupiter", "moon"],
            "relevant_houses": [jup_house, moon_house],
            "formation_rule_met": True,
            "pre_written_insight_keys": [{"category": "yoga_interpretation", "key": "gajakesari_yoga"}]
        })

    # Dharma Karmadhipati (lords of 9 & 10 conjoin)
    lord_9 = evaluator.get_house_lord(user_chart, 9)
    lord_10 = evaluator.get_house_lord(user_chart, 10)
    if lord_9 and lord_10 and user_chart['planets'].get(lord_9) and user_chart['planets'].get(lord_10):
        if user_chart['planets'][lord_9]['house'] == user_chart['planets'][lord_10]['house']:
            yogas_found.append({
                "type": "yoga",
                "name": "Dharma Karmadhipati Yoga",
                "key": "dharma_karmadhipati_yoga",
                "relevant_planets": [lord_9, lord_10],
                "relevant_houses": [user_chart['planets'][lord_9]['house']],
                "formation_rule_met": True,
                "pre_written_insight_keys": [{"category": "yoga_interpretation", "key": "dharma_karmadhipati_yoga"}]
            })

    # Conjunctions (wrap-aware + same house)
    yogas_found.extend(identify_conjunctions(user_chart))
    return yogas_found

def identify_conjunctions(user_chart, orb=5):
    findings = []
    planets_to_check = [p for p in user_chart['planets'].keys() if p not in ['rahu', 'ketu']]

    for i in range(len(planets_to_check)):
        p1 = planets_to_check[i]
        lon1 = user_chart['planets'][p1]['longitude']
        h1 = user_chart['planets'][p1]['house']
        for j in range(i + 1, len(planets_to_check)):
            p2 = planets_to_check[j]
            lon2 = user_chart['planets'][p2]['longitude']
            h2 = user_chart['planets'][p2]['house']
            if h1 == h2 and _min_sep(lon1, lon2) <= orb:
                a, b = sorted([p1, p2])
                key = f"{a}-{b}_conjunction"
                findings.append({
                    "type": "yoga",
                    "name": f"{a.capitalize()} - {b.capitalize()} Conjunction",
                    "key": key,
                    "relevant_planets": [a, b],
                    "relevant_houses": [h1],
                    "pre_written_insight_keys": [{"category": "yuti_2_graha", "key": key}]
                })
    return findings

# -------------------------------------------------------------------
# Aspects (Drishtis)
# -------------------------------------------------------------------

def identify_drishtis(user_chart):
    findings = []
    for p_name, p_data in user_chart['planets'].items():
        if p_name in ["rahu", "ketu"]:
            continue
        h = p_data['house']

        # 7th aspect
        h7 = ((h + 7 - 1) % 12) + 1
        findings.append({
            "type": "aspect",
            "name": f"{p_name.capitalize()}'s 7th Aspect",
            "key": f"{p_name}_7th_aspect_on_house_{h7}",
            "aspecting_planet": p_name,
            "aspected_house": h7,
            "pre_written_insight_keys": [{"category": "special_aspects", "key": f"{p_name}_7th_aspect_on_house_{h7}"}]
        })

        # Special aspects
        if p_name == 'mars':
            for off, lab in [(4, 4), (8, 8)]:
                hx = ((h + off - 1) % 12) + 1
                findings.append({
                    "type": "aspect",
                    "name": f"{p_name.capitalize()}'s {lab}th Aspect",
                    "key": f"{p_name}_{lab}th_aspect_on_house_{hx}",
                    "aspecting_planet": p_name,
                    "aspected_house": hx,
                    "pre_written_insight_keys": [{"category": "special_aspects", "key": f"{p_name}_{lab}th_aspect_on_house_{hx}"}]
                })
        elif p_name == 'jupiter':
            for off, lab in [(5, 5), (9, 9)]:
                hx = ((h + off - 1) % 12) + 1
                findings.append({
                    "type": "aspect",
                    "name": f"{p_name.capitalize()}'s {lab}th Aspect",
                    "key": f"{p_name}_{lab}th_aspect_on_house_{hx}",
                    "aspecting_planet": p_name,
                    "aspected_house": hx,
                    "pre_written_insight_keys": [{"category": "special_aspects", "key": f"{p_name}_{lab}th_aspect_on_house_{hx}"}]
                })
        elif p_name == 'saturn':
            for off, lab in [(3, 3), (10, 10)]:
                hx = ((h + off - 1) % 12) + 1
                findings.append({
                    "type": "aspect",
                    "name": f"{p_name.capitalize()}'s {lab}th Aspect",
                    "key": f"{p_name}_{lab}th_aspect_on_house_{hx}",
                    "aspecting_planet": p_name,
                    "aspected_house": hx,
                    "pre_written_insight_keys": [{"category": "special_aspects", "key": f"{p_name}_{lab}th_aspect_on_house_{hx}"}]
                })
    return findings

# -------------------------------------------------------------------
# Transit Analysis
# -------------------------------------------------------------------

def analyze_transits(natal_chart, transit_positions, ashtakavarga_scores):
    analysis = []

    moon_sign_idx = ac.SIGNS.index(natal_chart['planets']['moon']['sign'])
    saturn_sign = transit_positions['saturn']['sign']
    sade_sati_status_key = "not_in_sade_sati"
    if saturn_sign == ac.SIGNS[(moon_sign_idx - 1 + 12) % 12]:
        sade_sati_status_key = "first_phase"
    elif saturn_sign == natal_chart['planets']['moon']['sign']:
        sade_sati_status_key = "peak_phase"
    elif saturn_sign == ac.SIGNS[(moon_sign_idx + 1) % 12]:
        sade_sati_status_key = "final_phase"

    sav_bindus = ashtakavarga_scores.get('sarvashtakavarga', {}).get(saturn_sign, 0)
    analysis.append({
        "type": "transit",
        "name": "Sade Sati",
        "key": sade_sati_status_key,
        "status_key": "messages.saturn_in_bindus",
        "status_values": [saturn_sign, sav_bindus],
        "pre_written_insight_keys": [{"category": "sade_sati", "key": sade_sati_status_key}]
    })

    jupiter_sign = transit_positions['jupiter']['sign']
    jup_house_from_moon = (ac.SIGNS.index(jupiter_sign) - moon_sign_idx + 12) % 12 + 1
    analysis.append({
        "type": "transit",
        "name": "Jupiter Transit",
        "key": f"jupiter_transiting_house_{jup_house_from_moon}",
        "status_key": "messages.jupiter_transiting_house",
        "status_values": [jup_house_from_moon],
        "pre_written_insight_keys": [
            {"category": "jupiter_transit_from_moon", "key": f"jupiter_transiting_house_{jup_house_from_moon}"},
            {"category": "planet_transit_principle", "key": "jupiter"}
        ]
    })

    return analysis

def analyze_dasha_gochar_synthesis(natal_chart, transit_positions):
    findings = []
    today = datetime.now().date()
    for md in natal_chart.get('vimshottari_dasha', []):
        md_start = datetime.strptime(md['start_date'], "%Y-%m-%d").date()
        md_end = datetime.strptime(md['end_date'], "%Y-%m-%d").date()
        if md_start <= today <= md_end:
            md_lord = md['dasha_lord']
            for ad in md.get('antardashas', []):
                ad_start = datetime.strptime(ad['start_date'], "%Y-%m-%d").date()
                ad_end = datetime.strptime(ad['end_date'], "%Y-%m-%d").date()
                if ad_start <= today <= ad_end:
                    ad_lord = ad['antardasha_lord']
                    findings.append({
                        "type": "dasha",
                        "name": "Current Dasha Period",
                        "key": f"{md_lord}_mahadasha_{ad_lord}_antardasha",
                        "mahadasha_lord": md_lord,
                        "antardasha_lord": ad_lord,
                        "start_date": ad_start.strftime("%Y-%m-%d"),
                        "end_date": ad_end.strftime("%Y-%m-%d"),
                        "pre_written_insight_keys": [
                            {"category": "mahadasha_interpretation", "key": md_lord},
                            {"category": "antardasha_interpretation", "key": f"{md_lord}_{ad_lord}"}
                        ]
                    })
                    return findings
    return [{"type": "dasha", "name": "Current Dasha Period", "key": "dasha_period_not_found", "finding": "Current Dasha period not found.", "pre_written_insight_keys": []}]

# -------------------------------------------------------------------
# Master Orchestrator
# -------------------------------------------------------------------

def run_all_analysis(user_chart, transit_positions, ashtakavarga_scores, related_charts=None):
    """
    Orchestrates core analysis functions and returns raw findings.
    """
    findings = []
    # 1) Planetary packets
    findings.extend(get_all_planetary_data(user_chart))
    # 2) Yogas & Conjunctions
    findings.extend(identify_all_yogas(user_chart))
    # 3) Aspects
    findings.extend(identify_drishtis(user_chart))
    # 4) Doshas (STRICT Kaal Sarp)
    findings.extend(identify_all_doshas(user_chart))
    # 5) Transits
    findings.extend(analyze_transits(user_chart, transit_positions, ashtakavarga_scores))
    findings.extend(analyze_dasha_gochar_synthesis(user_chart, transit_positions))

    # Placeholder for multi-chart extensions
    if related_charts:
        findings.append({
            "type": "family_analysis_raw_placeholder",
            "name": "Raw Multi-Chart Analysis (Placeholder)",
            "key": "raw_multi_chart_analysis_todo",
            "details": {"message": "Raw multi-chart analysis will be implemented here."}
        })
    return findings
```

### astrological_evaluator.py


```python
# astrological_evaluator.py

import astrological_constants as ac

def get_house_lord(user_chart, house_number):
    """Calculates the lord of a given house based on the ascendant sign."""
    if 'ascendant' not in user_chart or 'sign' not in user_chart['ascendant']:
        return None
    ascendant_sign = user_chart['ascendant']['sign']
    try:
        house_sign_index = (ac.SIGNS.index(ascendant_sign) + house_number - 1) % 12
        house_sign = ac.SIGNS[house_sign_index]
        return ac.SIGN_LORDS.get(house_sign)
    except (ValueError, IndexError):
        return None

def get_house_classification(house_number):
    classifications = []
    if house_number in [1, 4, 7, 10]: classifications.append("kendra_angular")
    if house_number in [1, 5, 9]: classifications.append("trikona_trinal")
    if house_number in [6, 8, 12]: classifications.append("dusthana_malefic")
    if house_number in [3, 6, 10, 11]: classifications.append("upachaya_growth")
    return classifications if classifications else ["neutral"]

def evaluate_planetary_positional_strength(planet_lon, planet_house, house_cusps):
    """Evaluates if a planet is at the center (Madhya) or junction (Sandhi) of a house."""
    if not house_cusps or planet_house < 1 or planet_house > 12:
        return {"status": "normal_position"}
    # Bhava Madhya (center of the house) is typically 15 degrees from the cusp.
    # Bhava Sandhi (cusp/junction) is the cusp itself.

    # Get the cusp longitude for the current house
    current_house_cusp = house_cusps[planet_house - 1]

    # Get the cusp longitude for the next house (for calculating house span)
    # Handle wrapping around from 12th to 1st house
    next_house_cusp = house_cusps[planet_house % 12]

    # Calculate the span of the current house
    # Ensure correct calculation across 0/360 degree boundary
    if next_house_cusp < current_house_cusp:
        house_span = (360 - current_house_cusp) + next_house_cusp
    else:
        house_span = next_house_cusp - current_house_cusp

    # Calculate the Bhava Madhya (center) of the current house
    bhava_madhya = (current_house_cusp + house_span / 2) % 360

    # Calculate distance to Bhava Madhya
    dist_to_madhya = abs(planet_lon - bhava_madhya)
    if dist_to_madhya > 180: # Handle wrap-around for shortest distance
        dist_to_madhya = 360 - dist_to_madhya

    # Check if planet is near Bhava Madhya (e.g., within 3 degrees)
    if dist_to_madhya <= 3:
        return {"status": "very_strong_at_bhava_madhya", "house": planet_house}

    # Check if planet is near Bhava Sandhi (cusp) (e.g., within 3 degrees of current or next cusp)
    dist_to_current_cusp = abs(planet_lon - current_house_cusp)
    if dist_to_current_cusp > 180:
        dist_to_current_cusp = 360 - dist_to_current_cusp

    dist_to_next_cusp = abs(planet_lon - next_house_cusp)
    if dist_to_next_cusp > 180:
        dist_to_next_cusp = 360 - dist_to_next_cusp

    if dist_to_current_cusp <= 3 or dist_to_next_cusp <= 3:
        # Determine which sandhi it's closer to
        if dist_to_current_cusp <= dist_to_next_cusp:
            return {"status": "weak_at_bhava_sandhi", "house": planet_house, "cusp_type": "current_house_cusp"}
        else:
            return {"status": "weak_at_bhava_sandhi", "house": planet_house, "cusp_type": "next_house_cusp", "next_house": planet_house % 12 + 1}

    return {"status": "normal_position"}

def evaluate_bhavesha_strength(kundli, house_num):
    """Synthesizes the condition of a house lord (Bhavesha) to determine its strength."""
    lord_name = get_house_lord(kundli, house_num)
    if not lord_name: return {"status": "house_could_not_be_determined", "house": house_num}
    lord_data = kundli['planets'].get(lord_name)
    if not lord_data: return {"status": "lord_not_found", "house": house_num, "lord": lord_name}
    
    dignity, status, p_house = lord_data['dignity'], lord_data['status'], lord_data['house']
    
    classifications = get_house_classification(p_house)
    
    return {
        "status": "summary",
        "house_num": house_num,
        "lord_name": lord_name,
        "dignity": dignity.replace(' ', '_').lower(),
        "is_combust": "combust" in status,
        "is_retrograde": "retrograde" in status,
        "placed_house": p_house,
        "house_quality": classifications
    }

def check_neecha_bhanga(planet_name, kundli):
    """MODIFIED: Expands rules for Neecha Bhanga Raja Yoga."""
    planet_info = kundli['planets'].get(planet_name)
    if not planet_info or planet_info.get('dignity') != 'Debilitation': return []
    
    reasons = []
    dispositor_name = ac.SIGN_LORDS[planet_info['sign']]
    dispositor_info = kundli['planets'].get(dispositor_name)
    exalt_lord_name = ac.SIGN_LORDS[ac.PLANETARY_RULES[planet_name]['exalt']]
    exalt_lord_info = kundli['planets'].get(exalt_lord_name)

    if dispositor_info:
        if "kendra_angular" in get_house_classification(dispositor_info['house']):
            reasons.append({"key": "messages.debilitated_planet_reasons", "value": dispositor_name})
        moon_house = kundli['planets']['moon']['house']
        if dispositor_info['house'] in [(moon_house + i - 1) % 12 + 1 for i in [1, 4, 7, 10]]:
             reasons.append({"key": "messages.dispositor_in_kendra_moon", "value": dispositor_name})

    if exalt_lord_info:
         if "kendra_angular" in get_house_classification(exalt_lord_info['house']):
            reasons.append({"key": "messages.exaltation_lord_in_kendra", "value": exalt_lord_name})

    exalt_sign = ac.PLANETARY_RULES[planet_name]['exalt']
    if planet_info['d9_sign'] == exalt_sign:
        reasons.append({"key": "messages.exalted_in_navamsa", "value": exalt_sign})
        
    if "retrograde" in planet_info['status']:
        reasons.append({"key": "messages.is_retrograde_overcome"})
        
    if dispositor_info and dispositor_info['sign'] == planet_info['sign'] and ac.SIGN_LORDS[dispositor_info['sign']] == planet_name:
         reasons.append({"key": "messages.parivartana_yoga", "value": dispositor_name})

    return list({v['key']: v for v in reasons}.values())

def check_kemadruma_yoga(kundli):
    """MODIFIED: Expands rules for Kemadruma Yoga cancellation."""
    moon_house = kundli['planets']['moon']['house']
    moon_info = kundli['planets']['moon']
    
    adjacent_houses = [(moon_house - 2 + 12) % 12 + 1, (moon_house % 12) + 1]
    if any(p['house'] in adjacent_houses for n, p in kundli['planets'].items() if n not in ["sun", "moon", "rahu", "ketu"]):
        return {"status": False, "reason_key": "messages.no_planets_adjacent_moon"}

    cancellations = []
    
    if "kendra_angular" in get_house_classification(moon_house):
        cancellations.append("messages.moon_in_kendra")
        
    if any("kendra_angular" in get_house_classification(p['house']) for n, p in kundli['planets'].items() if n not in ["sun", "moon", "rahu", "ketu"]):
        cancellations.append("messages.planets_in_kendra_asc")

    moon_kendra = [(moon_house + i - 1) % 12 + 1 for i in [1, 4, 7, 10]]
    if any(p['house'] in moon_kendra for n, p in kundli['planets'].items() if n not in ["sun", "moon", "rahu", "ketu"]):
        cancellations.append("messages.planets_in_kendra_moon")

    aspects = get_aspects_on_house(moon_house, kundli)
    if 'jupiter' in aspects:
        cancellations.append("messages.moon_aspected_by_jupiter")
        
    if moon_info['d9_sign'] == 'taurus':
        cancellations.append("messages.moon_exalted_in_navamsa")
        
    if cancellations:
        return {"status": True, "type": "Cancelled", "cancellation_reasons": list(set(cancellations))}
        
    return {"status": True, "type": "Active", "reason_key": "messages.no_planets_adjacent_moon"}

def get_panchadha_maitri(p1, p2, kundli):
    p1_info, p2_info = kundli['planets'][p1], kundli['planets'][p2]
    natural_rel = "neutral"
    if p2 in ac.PLANETARY_RULES[p1]['friends']: natural_rel = "friend"
    elif p2 in ac.PLANETARY_RULES[p1]['enemies']: natural_rel = "enemy"
    p1_h, p2_h = p1_info['house'], p2_info['house']
    pos_diff = (p2_h - p1_h + 12) % 12
    temporal_rel = "friend" if pos_diff in [1, 2, 3, 9, 10, 11] else "enemy"
    if natural_rel == "friend" and temporal_rel == "friend": return "best_friend"
    if natural_rel == "enemy" and temporal_rel == "enemy": return "bitter_enemy"
    if natural_rel == "neutral": return temporal_rel
    if natural_rel != temporal_rel: return "neutral"
    return natural_rel

def get_aspects_on_house(target_house, kundli):
    """
    Calculates which planets are aspecting a given house.
    Note: This is house-based aspect, not planet-on-planet degree-based aspect.
    """
    aspecting_planets = []
    for p_name, p_data in kundli['planets'].items():
        if 'house' not in p_data: continue
        p_house = p_data['house']
        
        if target_house == ((p_house + 7 - 1) % 12) + 1:
            aspecting_planets.append(p_name)

        if p_name == 'mars':
            special_aspect_houses = [((p_house + 4 - 1) % 12) + 1, ((p_house + 8 - 1) % 12) + 1]
            if target_house in special_aspect_houses:
                aspecting_planets.append(p_name)
        elif p_name == 'jupiter':
            special_aspect_houses = [((p_house + 5 - 1) % 12) + 1, ((p_house + 9 - 1) % 12) + 1]
            if target_house in special_aspect_houses:
                aspecting_planets.append(p_name)
        elif p_name == 'saturn':
            special_aspect_houses = [((p_house + 3 - 1) % 12) + 1, ((p_house + 10 - 1) % 12) + 1]
            if target_house in special_aspect_houses:
                aspecting_planets.append(p_name)

    return list(set(aspecting_planets))
```

### rohini_engine.py


```python
# rohini_engine.py
# ------------------------------------------------------------
# North-Indian Vedic engine (Lahiri sidereal, mean node, whole-sign houses)
# - CHANGE: consistent sidereal everywhere
# - CHANGE: convert tropical Asc -> sidereal Asc using ayanamsa
# - CHANGE: whole-sign house mapping (no house_pos for Rashi)
# - CHANGE: lowercased planet keys end-to-end to avoid blanks/zeros
# - CHANGE: planetary war uses wrapped separation
# - CHANGE: transit calc fixed (flags + unpack)
# - CHANGE: Panchang at birth (tithi/vara/nakshatra/yoga/karana) + Chandra Lagna chart
# ------------------------------------------------------------

from datetime import datetime, timedelta
import pytz
import os
import swisseph as swe
import math
import astrological_constants as ac

# --- Init Swiss Ephemeris ---
swe.set_ephe_path(os.path.join(os.path.dirname(__file__), 'sweph_data'))
swe.set_sid_mode(swe.SIDM_LAHIRI)  # CHANGE: ensure sidereal mode globally

# ----------------- Helpers -----------------

SIGNS = ac.SIGNS  # ensure this is the standard Aries..Pisces list

def _norm_lon(lon):
    return lon % 360.0

def _format_abs(lon):
    lon = _norm_lon(lon)
    d = int(lon)
    m = int((lon - d) * 60)
    s = int(round((((lon - d) * 60) - m) * 60))
    if s == 60: s, m = 0, m + 1
    if m == 60: m, d = 0, d + 1
    return f"{d}° {m}' {s}\""

def _format_sign(lon):
    lon = _norm_lon(lon)
    si = int(lon // 30)
    x = lon - si * 30
    d = int(x)
    m = int((x - d) * 60)
    s = int(round((((x - d) * 60) - m) * 60))
    if s == 60: s, m = 0, m + 1
    if m == 60: m, d = 0, d + 1
    return f"{d}° {m}' {s}\" {SIGNS[si]}"

def get_sign(longitude):
    if longitude is None: return None
    return SIGNS[int(_norm_lon(longitude) // 30)]

def get_nakshatra(longitude):
    if longitude is None: return (None, None)
    lon = _norm_lon(longitude)
    span = 13 + 1/3
    idx = int(lon // span)
    into = lon - idx * span
    pada = int(into // (span / 4.0)) + 1
    return ac.NAKSHATRAS[idx], pada

def get_navamsa_sign(longitude):
    nak, pada = get_nakshatra(longitude)
    if not nak: return None
    ni = ac.NAKSHATRAS.index(nak)
    total = ni * 4 + (pada - 1)
    return SIGNS[total % 12]

def get_planet_dignity(planet_name, sign):
    # CHANGE: expect lowercase planet_name, lower-case keys in constants
    rules = ac.PLANETARY_RULES.get(planet_name)
    if not rules: return "Node"
    if rules.get('exalt') == sign: return "Exaltation"
    if rules.get('debilitate') == sign: return "Debilitation"
    if rules.get('moolatrikona') == sign: return "Moolatrikona"
    if sign in rules.get('own', []): return "Own Sign"
    dispo = ac.SIGN_LORDS.get(sign)
    if dispo in rules.get('friends', []): return "Friendly Sign"
    if dispo in rules.get('enemies', []): return "Enemy Sign"
    return "Neutral Sign"

def _wrapped_sep(a, b):
    """minimal angular separation, degrees"""
    a = _norm_lon(a); b = _norm_lon(b)
    d = abs(a - b)
    return min(d, 360 - d)

def check_combustion(planet_name, planet_lon, sun_lon, is_retro):
    if planet_name in ["sun", "rahu", "ketu"]:
        return False, 0.0
    # CHANGE: run in sidereal space (we now ensure both longitudes are sidereal)
    orb = ac.COMBUSTION_ORBS.get(planet_name, 0.0)
    if is_retro:
        if planet_name == "mercury": orb = max(orb, 12.0)
        if planet_name == "venus":   orb = max(orb, 8.0)
    dist = _wrapped_sep(planet_lon, sun_lon)
    return dist <= orb, round(dist, 2)

def check_planetary_war(planets_data):
    # CHANGE: wrap separation logic; only war-capable planets
    cand = ["mars", "mercury", "jupiter", "venus", "saturn"]
    items = [(n, d) for n, d in planets_data.items() if n in cand]
    in_war, victors = {}, {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            n1, d1 = items[i]; n2, d2 = items[j]
            sep = _wrapped_sep(d1['longitude'], d2['longitude'])
            if sep <= 1.0:
                in_war.setdefault(n1, []).append(n2)
                in_war.setdefault(n2, []).append(n1)
                victor = n1 if _norm_lon(d1['longitude']) < _norm_lon(d2['longitude']) else n2
                victors[tuple(sorted((n1, n2)))] = victor
    return in_war, victors

# -------- Simplified Shadbala (placeholder) --------
def calculate_shadbala(planet_name, kundli):
    if planet_name in ["rahu", "ketu"]:
        return {"total": 0.0, "ishta_phala": "N/A", "kashta_phala": "N/A"}
    p = kundli['planets'].get(planet_name, {})
    dign = p.get('dignity', "Neutral Sign")
    pts = {"Exaltation":1.0,"Moolatrikona":0.75,"Own Sign":0.5,"Friendly Sign":0.25,"Neutral Sign":0.125,"Enemy Sign":0.06,"Debilitation":0.0}
    uchcha = pts.get(dign, 0.0)
    dig_map = {"jupiter":1,"mercury":1,"sun":10,"mars":10,"saturn":7,"venus":4,"moon":4}
    dig = 1.0 if p.get('house') == dig_map.get(planet_name) else 0.0
    cheshta = 1.0 if p.get('is_retrograde', False) else 0.0
    nais = ac.NAISARGIKA_STRENGTHS.get(planet_name, 0.0)
    total = uchcha + dig + cheshta + nais
    ishta = math.sqrt(uchcha * cheshta) * 4.86 if cheshta > 0 else 0.0
    kashta = math.sqrt((1 - uchcha) * (1 - cheshta)) * 4.86 if cheshta < 1 else 0.0
    return {"total": round(total, 2), "ishta_phala": round(ishta, 2), "kashta_phala": round(kashta, 2)}

# -------- Vimshottari Dasha --------
# --- Vimshottari Dasha (MD → AD → PD) ---

def calculate_vimshottari_dasha(moon_longitude, birth_date):
    """
    Return a full Vimshottari timeline with Mahadashas, each with Antardashas,
    each with Pratyantardashas (3rd level), using standard 120-year proportions.

    Dates are naive (no TZ) and based on 365.2425-day tropical year for duration math consistency.
    """
    # constants
    ORDER = ac.DASHA_ORDER  # e.g., ["ketu","venus","sun","moon","mars","rahu","jupiter","saturn","mercury"]
    YEARS = ac.DASHA_DURATIONS  # dict mapping lord -> years (must sum to 120)
    DAYS_PER_YEAR = 365.2425

    # Find Moon's nakshatra and starting lord
    nak, _ = get_nakshatra(moon_longitude)
    if not nak:
        return []

    start_lord = ac.NAKSHATRA_LORDS.get(nak)
    if not start_lord:
        return []

    # Portion of the *starting* nakshatra consumed/remaining
    span = 13 + 1/3  # 13°20'
    ni = ac.NAKSHATRAS.index(nak)
    into = (_norm_lon(moon_longitude) - ni * span) % 360.0
    remaining_ratio = (span - into) / span  # fraction of first MD remaining

    # Helper to step through a cyclic list starting at "start"
    def cycle_from(lst, start):
        idx = lst.index(start)
        for i in range(len(lst)):
            yield lst[(idx + i) % len(lst)]

    # Helper to add a block with length in days
    def add_block(start_dt, length_days):
        end_dt = start_dt + timedelta(days=length_days)
        return start_dt, end_dt

    # Antardasha sequence for a given Mahadasha lord M starts from M itself
    def antardasha_sequence(md_lord, md_days):
        seq = list(cycle_from(ORDER, md_lord))
        # Each AD length is MD_days * (years(AD_lord)/120)
        ad_list = []
        t = birth_anchor  # will be overridden by caller
        for ad_lord in seq:
            ad_days = md_days * (YEARS[ad_lord] / 120.0)
            s, e = add_block(t, ad_days)
            ad_list.append({"antardasha_lord": ad_lord, "start": s, "end": e, "days": ad_days})
            t = e
        return ad_list

    # Pratyantardasha sequence for a given AD lord A starts from A itself
    def pratyantardasha_sequence(ad_lord, ad_start, ad_days):
        seq = list(cycle_from(ORDER, ad_lord))
        pd_list = []
        t = ad_start
        for pd_lord in seq:
            pd_days = ad_days * (YEARS[pd_lord] / 120.0)
            s, e = add_block(t, pd_days)
            pd_list.append({"pratyantardasha_lord": pd_lord, "start": s, "end": e, "days": pd_days})
            t = e
        return pd_list

    # Build the full tree
    md_list = []
    # Anchor time for the very first MD
    birth_anchor = datetime(birth_date.year, birth_date.month, birth_date.day)

    # Walk through full 120-year cycle starting at start_lord
    md_time = birth_anchor
    for md_lord in cycle_from(ORDER, start_lord):
        md_years = YEARS[md_lord]
        # First MD is partial (remaining fraction of its full years); others are full
        years_factor = remaining_ratio if (len(md_list) == 0) else 1.0
        md_days = md_years * DAYS_PER_YEAR * years_factor  # FIXED: use DAYS_PER_YEAR

        md_start, md_end = add_block(md_time, md_days)

        # Antardashas within this MD
        # For the first MD, we still start AD sequence from md_lord (classical rule)
        ads = antardasha_sequence(md_lord, md_days)
        # Fix their absolute times to the MD window
        t_ad = md_start
        ad_full = []
        for ad in ads:
            ad_len = ad["days"]
            s, e = add_block(t_ad, ad_len)
            # Pratyantardashas
            pds = pratyantardasha_sequence(ad["antardasha_lord"], s, ad_len)
            ad_full.append({
                "antardasha_lord": ad["antardasha_lord"],
                "start": s.strftime("%Y-%m-%d"),
                "end": e.strftime("%Y-%m-%d"),
                "duration": int(round(ad_len)),
                "pratyantardashas": [
                    {
                        "pratyantardasha_lord": pd["pratyantardasha_lord"],
                        "start": pd["start"].strftime("%Y-%m-%d"),
                        "end": pd["end"].strftime("%Y-%m-%d"),
                        "duration": int(round(pd["days"]))
                    } for pd in pds
                ]
            })
            t_ad = e

        md_list.append({
            "dasha_lord": md_lord,
            "start_date": md_start.strftime("%Y-%m-%d"),
            "end_date": md_end.strftime("%Y-%m-%d"),
            "duration": int(round(md_days)),
            "antardashas": ad_full
        })

        md_time = md_end

        # Stop once we’ve completed a full 120-year cycle from birth
        total_days = sum(md['duration'] for md in md_list)
        if total_days >= 120 * DAYS_PER_YEAR - 1:  # tolerance
            break

    return md_list

# -------- Transits (sidereal) --------
def get_transit_positions(date_obj):
    # CHANGE: sidereal + correct unpack
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    utc_dt = date_obj.astimezone(pytz.utc) if date_obj.tzinfo else pytz.utc.localize(date_obj)
    jd = swe.utc_to_jd(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute, utc_dt.second)[0]
    out = {}
    planets = {
        "sun": swe.SUN, "moon": swe.MOON, "mars": swe.MARS, "mercury": swe.MERCURY,
        "jupiter": swe.JUPITER, "venus": swe.VENUS, "saturn": swe.SATURN, "rahu": swe.MEAN_NODE
    }
    for n, pid in planets.items():
        xx, _ = swe.calc_ut(jd, pid, swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = _norm_lon(xx[0])
        out[n] = {"longitude": lon, "sign": get_sign(lon)}
    if "rahu" in out:
        ketu = _norm_lon(out["rahu"]["longitude"] + 180)
        out["ketu"] = {"longitude": ketu, "sign": get_sign(ketu)}
    return out

# ============================================================
# CHANGE: Panchang helpers (tithi, karana, yoga, vara) + Chandra Lagna
# ============================================================

def _tithi_info(sun_lon, moon_lon):
    # Tithi = (Moon - Sun) / 12°, 1..30
    diff = _norm_lon(moon_lon - sun_lon)
    tithi_num = int(diff // 12) + 1  # 1..30
    paksha = "Shukla" if diff < 180.0 else "Krishna"
    # Names 1..30
    TITHIS = [
        "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
        "Ekadashi","Dvadashi","Trayodashi","Chaturdashi","Purnima",
        "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
        "Ekadashi","Dvadashi","Trayodashi","Chaturdashi","Amavasya"
    ]
    name = f"{paksha} {TITHIS[tithi_num-1]}"
    return {"number": tithi_num, "paksha": paksha, "name": name}

def _karana_name(sun_lon, moon_lon):
    # 60 karanas per lunation, each 6°
    diff = _norm_lon(moon_lon - sun_lon)
    idx = int(diff // 6)  # 0..59
    # Fixed + repeating scheme:
    repeating = ["Bava","Balava","Kaulava","Taitila","Garaja","Vanija","Vishti"]  # 7 repeating
    if idx == 0:
        return "Kimstughna"
    elif 1 <= idx <= 56:
        return repeating[(idx - 1) % 7]
    elif idx == 57:
        return "Shakuni"
    elif idx == 58:
        return "Chatushpada"
    else:  # idx == 59
        return "Naga"

def _yoga_name(sun_lon, moon_lon):
    # Yoga = (Sun + Moon) / 13°20' (13.333333...)
    span = 13.3333333333
    total = _norm_lon(sun_lon + moon_lon)
    yi = int(total // span)  # 0..26
    YOGAS = [
        "Vishkambha","Priti","Ayushman","Saubhagya","Shobhana","Atiganda","Sukarma","Dhriti","Shula","Ganda",
        "Vriddhi","Dhruva","Vyaghata","Harshana","Vajra","Siddhi","Vyatipata","Variyan","Parigha","Shiva",
        "Siddha","Sadhya","Shubha","Shukla","Brahma","Indra","Vaidhriti"
    ]
    return YOGAS[yi]

def _vara_info(local_dt):
    # Monday=0 in Python; Vara starts with Sunday
    weekday = local_dt.weekday()  # 0..6 (Mon..Sun)
    # Map to Sunday..Saturday index
    py_to_vara = [1, 2, 3, 4, 5, 6, 0]  # Mon->1 (Mon), ..., Sun->0
    vara_idx = py_to_vara[weekday]
    VARA = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
    LORD = ["Sun","Moon","Mars","Mercury","Jupiter","Venus","Saturn"]
    return {"vara": VARA[vara_idx], "lord": LORD[vara_idx]}

def compute_panchang(kundli, local_dt):
    """Compute Panchang limbs at birth (tithi, vara, nakshatra+pada, yoga, karana)."""
    sun_lon = kundli["planets"]["sun"]["longitude"]
    moon_lon = kundli["planets"]["moon"]["longitude"]

    t = _tithi_info(sun_lon, moon_lon)
    y = _yoga_name(sun_lon, moon_lon)
    k = _karana_name(sun_lon, moon_lon)
    v = _vara_info(local_dt)

    # We already have Moon's nakshatra/pada from your engine
    nak = kundli["planets"]["moon"]["nakshatra"]
    pada = kundli["planets"]["moon"]["pada"]

    return {
        "tithi": t["name"], "tithi_number": t["number"], "paksha": t["paksha"],
        "vara": v["vara"], "vara_lord": v["lord"],
        "nakshatra": nak, "pada": pada,
        "yoga": y,
        "karana": k
    }

def compute_chandra_lagna_chart(kundli):
    """Whole-sign chart counted from Moon's sign as Lagna (house 1)."""
    moon_sign = kundli["planets"]["moon"]["sign"]
    moon_idx = SIGNS.index(moon_sign)
    # Build houses: 1..12 from Moon
    houses = []
    for i in range(12):
        sign = SIGNS[(moon_idx + i) % 12]
        houses.append({"house": i+1, "sign": sign, "planets": []})
    # Place planets by sign distance from Moon-sign
    for pname, p in kundli["planets"].items():
        si = SIGNS.index(p["sign"])
        house_from_moon = (si - moon_idx + 12) % 12 + 1
        houses[house_from_moon - 1]["planets"].append(pname)
    return {"lagna_sign": moon_sign, "houses": houses}

# ---------------- Main Engine ----------------
def generate_full_kundli(birth_data):
    """
    Expected birth_data:
    {
      'timezone_str': 'Asia/Kolkata', 'year':..., 'month':..., 'day':...,
      'hour':..., 'minute':..., 'second':..., 'latitude':..., 'longitude':...
    }
    """
    swe.set_sid_mode(swe.SIDM_LAHIRI)

    # Local -> UTC -> JD
    tz = pytz.timezone(birth_data['timezone_str'])
    local_dt = tz.localize(datetime(birth_data['year'], birth_data['month'], birth_data['day'],
                                    birth_data['hour'], birth_data['minute'], birth_data['second']), is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    jd = swe.utc_to_jd(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour, utc_dt.minute, utc_dt.second)[0]

    # ----- Asc and houses (convert tropical → sidereal) -----
    cusps_trop, ascmc_t = swe.houses(jd, birth_data['latitude'], birth_data['longitude'], b'P')
    ayan = swe.get_ayanamsa_ut(jd)  # CHANGE
    asc_sid = _norm_lon(ascmc_t[0] - ayan)  # CHANGE: sidereal asc
    asc_sign_idx = int(asc_sid // 30)
    asc_sign = SIGNS[asc_sign_idx]

    houses = [{"house_number": i + 1, "sign": SIGNS[(asc_sign_idx + i) % 12]} for i in range(12)]  # CHANGE: whole-sign

    kundli = {
        "birth_data": birth_data,
        "ascendant": {
            "longitude": asc_sid,
            "formatted_abs": _format_abs(asc_sid),
            "formatted_sign": _format_sign(asc_sid),
            "sign": asc_sign,
            "d9_sign": get_navamsa_sign(asc_sid)
        },
        "house_cusps_sidereal": [_norm_lon(c - ayan) for c in cusps_trop],  # for reference
        "houses": houses,
        "planets": {}
    }

    # ----- Planets (sidereal) -----
    planets_map = {
        "sun": swe.SUN, "moon": swe.MOON, "mars": swe.MARS, "mercury": swe.MERCURY,
        "jupiter": swe.JUPITER, "venus": swe.VENUS, "saturn": swe.SATURN, "rahu": swe.MEAN_NODE
    }

    p_data = {}
    sun_lon = None
    for name, pid in planets_map.items():
        xx, _ = swe.calc_ut(jd, pid, swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED)  # CHANGE
        lon = _norm_lon(xx[0])
        speed = xx[3]
        if name == "sun":
            sun_lon = lon
        # CHANGE: whole-sign house calc
        sign_idx = int(lon // 30)
        house_num = (sign_idx - asc_sign_idx + 12) % 12 + 1
        p_data[name] = {"longitude": lon, "speed": speed, "is_retrograde": speed < 0, "house": house_num}

    # Ketu = Rahu + 180
    rahu_lon = p_data["rahu"]["longitude"]
    ketu_lon = _norm_lon(rahu_lon + 180.0)
    p_data["ketu"] = {"longitude": ketu_lon, "speed": -p_data["rahu"]["speed"], "is_retrograde": True,
                      "house": (int(ketu_lon // 30) - asc_sign_idx + 12) % 12 + 1}

    # planetary war
    war_status, war_victors = check_planetary_war(p_data)

    # finalize planets
    for name, data in p_data.items():
        sign = get_sign(data['longitude'])
        combust, dist = check_combustion(name, data['longitude'], sun_lon, data['is_retrograde'])
        tags = []
        if data['is_retrograde'] and name not in ["rahu", "ketu"]:
            tags.append("retrograde")
        if combust:
            tags.append(f"combust ({dist}°)")
        d9 = get_navamsa_sign(data['longitude'])
        if d9 == sign:
            tags.append("vargottama")
        if name in war_status:
            tags.append("in_planetary_war" + ("_victor" if any(v == name for v in war_victors.values()) else "_defeated"))

        nak, pada = get_nakshatra(data['longitude'])

        kundli["planets"][name] = {
            **data,
            "formatted_abs": _format_abs(data['longitude']),
            "formatted_sign": _format_sign(data['longitude']),
            "sign": sign,
            "d9_sign": d9,
            "nakshatra": nak,
            "pada": pada,
            "dignity": get_planet_dignity(name, sign),
            "status": ", ".join(tags) if tags else "normal"
        }

    # shadbala (simplified)
    for n in list(kundli["planets"].keys()):
        kundli["planets"][n]["shadbala"] = calculate_shadbala(n, kundli)

    # vimshottari
    kundli["vimshottari_dasha"] = calculate_vimshottari_dasha(kundli["planets"]["moon"]["longitude"], utc_dt.date())

    # CHANGE: Panchang at birth (use LOCAL datetime) + Chandra Lagna chart
    kundli["panchang"] = compute_panchang(kundli, local_dt)
    kundli["chandra_lagna_chart"] = compute_chandra_lagna_chart(kundli)

    return kundli

```

### orchestration_engine.py


```python
# orchestration_engine.py (NEW Module: Handles orchestration, weighting, and LLM brief compilation)

import math
from datetime import datetime
import db_utils # For fetching weights, interpretations, and LLM prompts
import rohini_engine # For transit positions (called by analysis_engine)
import analysis_engine # The re-scoped analysis core

# --- Dynamic Weights Loading ---
_dynamic_weights = {} # Module-level variable to store loaded weights

def _load_dynamic_weights():
    """
    Fetches the dynamic weighting configuration from the database.
    This function is called once when the module is imported.
    """
    global _dynamic_weights
    try:
        weights_config = db_utils.fetch_analysis_weights_config('current_weights')
        if weights_config:
            _dynamic_weights = weights_config
        else:
            _dynamic_weights = {}
            print("Warning: Analysis weights not found in DB or empty. Using default empty weights.")
    except Exception as e:
        print(f"Error loading dynamic weights: {e}. Using default empty weights.")
        _dynamic_weights = {}

# Call this function to load weights when the module is imported
_load_dynamic_weights()

# --- Prominence Score Calculation ---
def _calculate_prominence_score(finding, trigger_context):
    """
    Calculates the prominence score for a given finding using dynamic weights.
    """
    score = _dynamic_weights.get('default_base_scores', {}).get(finding['type'], 0)

    # Handle Dosha specific base scores
    if finding['type'] == 'dosha' and finding.get('status'):
        dosha_status = finding['status'].lower()
        score = _dynamic_weights.get('dosha_severity_base_scores', {}).get(dosha_status, score)

    # Handle Special Yoga base scores
    special_yogas_keys = ["gajakesari_yoga", "dharma_karmadhipati_yoga", "sasa_yoga",
                          "vasumati_yoga", "neecha_bhanga_raja_yoga"] # Add other Raja/Dhana/Pancha Mahapurusha Yogas
    if finding['type'] == 'yoga' and finding.get('key') in special_yogas_keys:
        score = _dynamic_weights.get('special_yoga_base_score', score)


    # Apply modifiers based on dynamic weights
    details = finding.get('details', {})
    
    # Dignity modifiers
    if 'dignity' in details:
        dignity_status = details['dignity'].lower().replace(' ', '_')
        score += _dynamic_weights.get('dignity_modifiers', {}).get(dignity_status, 0)

    # Vargottama modifier
    if details.get('vargottama_status'): # Assuming vargottama_status is added to details
        score += _dynamic_weights.get('vargottama_modifier', 0)

    # Shadbala modifiers
    shadbala_total_val = details.get('shadbala_total') # Get the numerical value
    if shadbala_total_val is not None: # Ensure it's not None
        shadbala_score = shadbala_total_val 

        if shadbala_score > 1.25:
            score += _dynamic_weights.get('shadbala_modifiers', {}).get('very_strong', 0)
        elif shadbala_score >= 1.0:
            score += _dynamic_weights.get('shadbala_modifiers', {}).get('strong', 0)
        elif shadbala_score >= 0.8:
            score += _dynamic_weights.get('shadbala_modifiers', {}).get('average', 0)
        elif shadbala_score >= 0.5:
            score += _dynamic_weights.get('shadbala_modifiers', {}).get('weak', 0)
        else:
            score += _dynamic_weights.get('shadbala_modifiers', {}).get('very_weak', 0)

    # Avastha modifiers
    avastha_modifiers = _dynamic_weights.get('avastha_modifiers', {})
    if 'baladi_avastha' in details:
        score += avastha_modifiers.get('baladi', {}).get(details['baladi_avastha'], 0)
    if 'deeptadi_avastha' in details:
        score += avastha_modifiers.get('deeptadi', {}).get(details['deeptadi_avastha'], 0)
    if 'lajjitaadi_avastha' in details:
        score += avastha_modifiers.get('lajjitaadi', {}).get(details['lajjitaadi_avastha'], 0)

    # Bhava Positional Strength modifiers
    if 'bhava_positional_strength' in details: # Assuming this is added to details
        bhava_pos_key = details['bhava_positional_strength'].lower().replace(' ', '_')
        score += _dynamic_weights.get('bhava_positional_modifiers', {}).get(bhava_pos_key, 0)

    # Bhavesha Strength modifiers (assuming summary is part of finding details or context)
    if 'bhavesha_strength_summary' in details: # Assuming this is added to details
        bhavesha_summary = details['bhavesha_strength_summary']
        if bhavesha_summary.get('dignity') in ["Exaltation", "Own Sign", "Moolatrikona"]:
            score += _dynamic_weights.get('bhavesha_strength_modifiers', {}).get('strong_lord', 0)
        elif bhavesha_summary.get('dignity') in ["Debilitation", "Enemy Sign"] or bhavesha_summary.get('is_combust') or bhavesha_summary.get('is_retrograde'):
            score += _dynamic_weights.get('bhavesha_strength_modifiers', {}).get('weak_lord', 0)

    # Ashtakavarga modifiers
    if 'sarvashtakavarga_score_for_house' in details: # Assuming this is added to details
        sav_score = details['sarvashtakavarga_score_for_house']
        if sav_score > 30:
            score += _dynamic_weights.get('ashtakavarga_modifiers', {}).get('high_bindus', 0)
        elif sav_score < 28:
            score += _dynamic_weights.get('ashtakavarga_modifiers', {}).get('low_bindus', 0)
        else:
            score += _dynamic_weights.get('ashtakavarga_modifiers', {}).get('average_bindus', 0)

    # Apply query-specific boosts dynamically
    if trigger_context.get('type') == 'reactive' and trigger_context.get('query'):
        query_type = trigger_context['query'].lower()
        query_boosts = _dynamic_weights.get('query_type_boosts', {}).get(query_type, {})

        if query_type == 'career':
            # Boost for houses relevant to career
            if details.get('house') in [10, 6, 2, 11]:
                score += query_boosts.get(f"{details['house']}th_house", 0)
            # Boost for planets relevant to career
            if details.get('relevant_planet') in ['saturn', 'mercury', 'jupiter']:
                score += query_boosts.get(details['relevant_planet'], 0)
            # Boost for specific yogas/features
            if finding.get('key') in ['dharma_karmadhipati_yoga', 'dhana_yoga']:
                score += query_boosts.get(finding['key'], 0)
            # Add D10 emphasis if applicable (assuming a finding type for D10)
            if finding.get('type') == 'd10_chart_emphasis': # Placeholder type
                score += query_boosts.get('d10_chart_emphasis', 0)

        elif query_type == 'marriage':
            # Boost for houses relevant to marriage
            if details.get('house') in [7, 5, 1]:
                score += query_boosts.get(f"{details['house']}th_house", 0)
            # Boost for planets relevant to marriage
            if details.get('relevant_planet') in ['venus', 'moon', 'jupiter']:
                score += query_boosts.get(details['relevant_planet'], 0)
            # Boost for compatibility features
            if finding.get('type') == 'guna_milan': # Placeholder type
                score += query_boosts.get('guna_milan_match_emphasis', 0)
            if finding.get('key') == 'mangal_dosha': # Check for Mangal Dosha impact
                score += query_boosts.get('mangal_dosha_impact_emphasis', 0)


    # Cap the score between 0 and 100
    return min(100, max(0, score))

# --- Insight Fetching and Formatting ---
def _fetch_and_format_insights(finding, language):
    """
    Fetches pre-written interpretations for a given finding from the database.
    """
    insights_list = []
    for insight_key_obj in finding.get('pre_written_insight_keys', []):
        category = insight_key_obj.get('category')
        key = insight_key_obj.get('key')
        if category and key:
            interpretation_data = db_utils.fetch_interpretation(category, key) # db_utils should handle language
            if interpretation_data:
                # Assuming interpretation_data is a list of dicts or a single dict
                if isinstance(interpretation_data, list):
                    insights_list.extend(interpretation_data)
                else:
                    insights_list.append(interpretation_data)
    return insights_list

# --- Analytical Brief Compiler ---
def compile_analytical_brief(user_chart, trigger_context, related_charts=None):
    """
    Orchestrates the entire analysis, filters findings, fetches insights,
    and assembles the final structured Analytical Brief for the LLM.
    """
    # Assume transit_positions and ashtakavarga_scores are directly available or fetched here
    # For now, fetch transit_positions dynamically as it's time-dependent
    transit_positions = rohini_engine.get_transit_positions(datetime.now())
    # Assume user_chart has 'ashtakavarga_scores' or pass a placeholder if not
    ashtakavarga_scores = user_chart.get('ashtakavarga_scores', {}) 

    # 1. Run all core analysis to get raw findings
    all_raw_findings = analysis_engine.run_all_analysis(
        user_chart, transit_positions, ashtakavarga_scores, related_charts
    )

    # Determine language from user_chart or trigger_context
    language = user_chart.get('birth_data', {}).get('language', 'en') # Default to English

    # 2. Apply Prominence Score to all raw findings
    all_findings_with_scores = []
    for finding in all_raw_findings:
        finding.setdefault('type', 'general_finding') # Default type if missing
        finding['prominence_score'] = _calculate_prominence_score(finding, trigger_context)
        all_findings_with_scores.append(finding)

    # 3. Filter Prominent Findings
    prominent_findings = []
    default_threshold = _dynamic_weights.get('filtering_thresholds', {}).get('default_inclusion_threshold', 60)
    max_findings = _dynamic_weights.get('filtering_thresholds', {}).get('max_findings_in_brief', 15) # Example limit

    # Sort by prominence_score descending
    sorted_findings = sorted(all_findings_with_scores, key=lambda x: x.get('prominence_score', 0), reverse=True)

    for finding in sorted_findings:
        if finding.get('prominence_score', 0) >= default_threshold:
            prominent_findings.append(finding)
        if len(prominent_findings) >= max_findings: # Limit the number of findings
            break
    
    # 4. Fetch Pre-written Insights for prominent findings
    for finding in prominent_findings:
        # Add pre_written_insight_keys to the finding before fetching
        # This part requires the analysis_engine to have added these keys to the raw findings
        # For now, we'll assume they are there or add a placeholder list if missing
        if 'pre_written_insight_keys' not in finding:
            finding['pre_written_insight_keys'] = [] # Ensure the key exists

        finding['pre_written_insights'] = _fetch_and_format_insights(finding, language)

    # 5. Assemble Analytical Brief
    analytical_brief = {
        "request_details": {
            "user_id": user_chart.get('user_id', 'unknown'), # Assume user_id is in user_chart
            "language": language,
            "trigger_type": trigger_context.get('type'),
            "query_or_event": trigger_context.get('query') or trigger_context.get('event'),
            "requested_detail_level": "detailed" if user_chart.get('is_premium', False) else "summary" # Assume is_premium
        },
        "primary_chart_analysis": {
            "birth_data": {
                "name": user_chart['birth_data'].get('name'),
                "year": user_chart['birth_data'].get('year'),
                "month": user_chart['birth_data'].get('month'),
                "day": user_chart['birth_data'].get('day'),
                "hour": user_chart['birth_data'].get('hour'),
                "minute": user_chart['birth_data'].get('minute'),
                "timezone_str": user_chart['birth_data'].get('timezone_str')
            },
            "ascendant": {
                "sign": user_chart['ascendant'].get('sign'),
                "d9_sign": user_chart['ascendant'].get('d9_sign')
            },
            "foundational_details": analysis_engine.get_avakahada_chakra(user_chart),
            "all_planets_data": {}, # Populate this with a simplified version of all planets data
            "prominent_findings": prominent_findings,
            "current_dasha_gochar": {} # Populate this from findings or dedicated function
        },
        "related_chart_analysis": {} # Placeholder for family analysis
    }

    # Populate all_planets_data with simplified version
    for planet_name, planet_data in user_chart['planets'].items():
        analytical_brief['primary_chart_analysis']['all_planets_data'][planet_name] = {
            "sign": planet_data.get('sign'),
            "house": planet_data.get('house'),
            "dignity": planet_data.get('dignity'),
            "status": planet_data.get('status'),
            "shadbala_total": planet_data.get('shadbala', {}).get('total'),
            "is_retrograde": planet_data.get('is_retrograde', False)
        }

    # Extract current dasha/gochar from findings and populate current_dasha_gochar
    dasha_finding = next((f for f in all_findings_with_scores if f.get('type') == 'dasha'), None)
    if dasha_finding:
        analytical_brief['primary_chart_analysis']['current_dasha_gochar']['current_mahadasha_lord'] = dasha_finding.get('mahadasha_lord')
        analytical_brief['primary_chart_analysis']['current_dasha_gochar']['current_antardasha_lord'] = dasha_finding.get('antardasha_lord')
    
    transit_finding = next((f for f in all_findings_with_scores if f.get('type') == 'transit' and f.get('name') == 'Sade Sati'), None)
    if transit_finding:
        analytical_brief['primary_chart_analysis']['current_dasha_gochar']['sade_sati_status'] = transit_finding.get('key') # Use key as status

    # Fetch LLM Prompts
    analytical_brief['master_persona_prompt'] = db_utils.fetch_prompt_template('master_persona', language)
    analytical_brief['analytical_brief_template'] = db_utils.fetch_prompt_template('analytical_brief', language)

    # Placeholder for related_chart_analysis if provided
    if related_charts:
        analytical_brief['related_chart_analysis'] = {
            "chart_details": { # Simplified details for related chart
                "name": related_charts[0].get('birth_data', {}).get('name'),
                "relation_type": related_charts[0].get('relation_type', 'unknown')
            },
            "compatibility_findings": [], # To be populated in future tasks
            "progeny_indicators": [] # To be populated in future tasks
        }

    return analytical_brief

```

### astrological_constants.py


```python
# astrological_constants.py

# A single, centralized source for all astrological data and rules.

# Sign and Nakshatra Data (Refactored to canonical lowercase keys)
SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
]

NAKSHATRAS = [
    "ashwini", "bharani", "krittika", "rohini", "mrigashira", "ardra",
    "punarvasu", "pushya", "ashlesha", "magha", "purva_phalguni",
    "uttara_phalguni", "hasta", "chitra", "swati", "vishakha",
    "anuradha", "jyeshtha", "moola", "purva_ashadha", "uttara_ashadha",
    "shravana", "dhanishta", "shatabhisha", "purva_bhadrapada",
    "uttara_bhadrapada", "revati"
]

GANDMOOL_NAKSHATRAS = ["ashwini", "ashlesha", "magha", "jyeshtha", "moola", "revati"]

SIGN_LORDS = {
    "aries": "mars", "taurus": "venus", "gemini": "mercury", "cancer": "moon",
    "leo": "sun", "virgo": "mercury", "libra": "venus", "scorpio": "mars",
    "sagittarius": "jupiter", "capricorn": "saturn", "aquarius": "saturn", "pisces": "jupiter"
}

# Comprehensive Planetary Rules (Dignities, Friendships, etc.)
PLANETARY_RULES = {
    "sun": {"exalt": "aries", "debilitate": "libra", "moolatrikona": "leo", "own": ["leo"], "friends": ["moon", "mars", "jupiter"], "enemies": ["venus", "saturn"], "neutral": ["mercury"]},
    "moon": {"exalt": "taurus", "debilitate": "scorpio", "moolatrikona": "taurus", "own": ["cancer"], "friends": ["sun", "mercury"], "enemies": [], "neutral": ["mars", "jupiter", "venus", "saturn"]},
    "mars": {"exalt": "capricorn", "debilitate": "cancer", "moolatrikona": "aries", "own": ["aries", "scorpio"], "friends": ["sun", "moon", "jupiter"], "enemies": ["mercury"], "neutral": ["venus", "saturn"]},
    "mercury": {"exalt": "virgo", "debilitate": "pisces", "moolatrikona": "virgo", "own": ["gemini", "virgo"], "friends": ["sun", "venus"], "enemies": ["moon"], "neutral": ["mars", "jupiter", "saturn"]},
    "jupiter": {"exalt": "cancer", "debilitate": "capricorn", "moolatrikona": "sagittarius", "own": ["sagittarius", "pisces"], "friends": ["sun", "moon", "mars"], "enemies": ["mercury", "venus"], "neutral": ["saturn"]},
    "venus": {"exalt": "pisces", "debilitate": "virgo", "moolatrikona": "libra", "own": ["taurus", "libra"], "friends": ["mercury", "saturn"], "enemies": ["sun", "moon"], "neutral": ["mars", "jupiter"]},
    "saturn": {"exalt": "libra", "debilitate": "aries", "moolatrikona": "aquarius", "own": ["capricorn", "aquarius"], "friends": ["mercury", "venus"], "enemies": ["sun", "moon", "mars"], "neutral": ["jupiter"]}
}

# Combustion Orbs (degrees)
COMBUSTION_ORBS = {
    "moon": 12, "mars": 17, "mercury": 14, "jupiter": 11, "venus": 10, "saturn": 15
}

# Vimshottari Dasha Data
NAKSHATRA_LORDS = {
    "ashwini": "ketu", "magha": "ketu", "moola": "ketu", "bharani": "venus", "purva_phalguni": "venus", "purva_ashadha": "venus",
    "krittika": "sun", "uttara_phalguni": "sun", "uttara_ashadha": "sun", "rohini": "moon", "hasta": "moon", "shravana": "moon",
    "mrigashira": "mars", "chitra": "mars", "dhanishta": "mars", "ardra": "rahu", "swati": "rahu", "shatabhisha": "rahu",
    "punarvasu": "jupiter", "vishakha": "jupiter", "purva_bhadrapada": "jupiter", "pushya": "saturn", "anuradha": "saturn", "uttara_bhadrapada": "saturn",
    "ashlesha": "mercury", "jyeshtha": "mercury", "revati": "mercury"
}
DASHA_DURATIONS = {"ketu": 7, "venus": 20, "sun": 6, "moon": 10, "mars": 7, "rahu": 18, "jupiter": 16, "saturn": 19, "mercury": 17}
DASHA_ORDER = ["ketu", "venus", "sun", "moon", "mars", "rahu", "jupiter", "saturn", "mercury"]

# Shadbala Constants
NAISARGIKA_STRENGTHS = {"sun": 1.0, "moon": 0.85, "mars": 0.7, "mercury": 0.55, "jupiter": 0.4, "venus": 0.25, "saturn": 0.1}

# Avakahada Chakra Data Maps (Refactored to canonical lowercase keys)
VARNA_MAP = {'cancer': 'brahmin', 'scorpio': 'brahmin', 'pisces': 'brahmin', 'aries': 'kshatriya', 'leo': 'kshatriya', 'sagittarius': 'kshatriya', 'taurus': 'vaishya', 'virgo': 'vaishya', 'capricorn': 'vaishya', 'gemini': 'shudra', 'libra': 'shudra', 'aquarius': 'shudra'}
VASHYA_MAP = {'aries': 'chatushpada', 'taurus': 'chatushpada', 'leo': 'chatushpada', 'sagittarius': 'chatushpada', 'capricorn': 'chatushpada', 'gemini': 'manava', 'virgo': 'manava', 'libra': 'manava', 'aquarius': 'manava', 'cancer': 'jalachara', 'pisces': 'jalachara', 'scorpio': 'keeta'}
YONI_MAP = {'ashwini': 'horse', 'shatabhisha': 'horse', 'bharani': 'elephant', 'revati': 'elephant', 'krittika': 'goat', 'pushya': 'goat', 'rohini': 'serpent', 'mrigashira': 'serpent', 'ardra': 'dog', 'moola': 'dog', 'punarvasu': 'cat', 'ashlesha': 'cat', 'magha': 'rat', 'purva_phalguni': 'rat', 'uttara_phalguni': 'cow', 'uttara_bhadrapada': 'cow', 'hasta': 'buffalo', 'swati': 'buffalo', 'chitra': 'tiger', 'vishakha': 'tiger', 'anuradha': 'deer', 'jyeshtha': 'deer', 'purva_ashadha': 'monkey', 'shravana': 'monkey', 'dhanishta': 'lion', 'purva_bhadrapada': 'lion'}
GANA_MAP = {'deva': ['ashwini', 'mrigashira', 'punarvasu', 'pushya', 'hasta', 'swati', 'anuradha', 'shravana', 'revati'], 'manushya': ['bharani', 'rohini', 'ardra', 'purva_phalguni', 'uttara_phalguni', 'purva_ashadha', 'uttara_ashadha', 'purva_bhadrapada', 'uttara_bhadrapada'], 'rakshasa': ['krittika', 'ashlesha', 'magha', 'chitra', 'vishakha', 'jyeshtha', 'moola', 'dhanishta', 'shatabhisha']}
NADI_MAP = {'adi': ['ashwini', 'ardra', 'punarvasu', 'uttara_phalguni', 'hasta', 'jyeshtha', 'moola', 'shatabhisha', 'purva_bhadrapada'], 'madhya': ['bharani', 'mrigashira', 'pushya', 'purva_phalguni', 'chitra', 'anuradha', 'purva_ashadha', 'dhanishta', 'uttara_bhadrapada'], 'antya': ['krittika', 'rohini', 'ashlesha', 'magha', 'swati', 'vishakha', 'uttara_ashadha', 'shravana', 'revati']}

# Avastha Constants (Refactored to canonical lowercase keys)
BALADI_AVASTHA_RANGES = {
    "odd": [("bala", 0, 6), ("kumara", 6, 12), ("yuva", 12, 18), ("vriddha", 18, 24), ("mrita", 24, 30)],
    "even": [("mrita", 0, 6), ("vriddha", 6, 12), ("yuva", 12, 18), ("kumara", 18, 24), ("bala", 24, 30)]
}
BENEFICS = ["jupiter", "venus", "moon", "mercury"]
MALEFICS = ["sun", "mars", "saturn", "rahu", "ketu"]


# ===================================================================
# PANCHANG CONSTANTS (REPLACE THE PREVIOUS PANCHANG SECTION WITH THIS)
# ===================================================================

VARA_RULERS = {
    0: "moon", 1: "mars", 2: "mercury", 3: "jupiter",
    4: "venus", 5: "saturn", 6: "sun"
} # Monday is 0, Sunday is 6

TITHIS = [
    "shukla_pratipada", "shukla_dwitiya", "shukla_tritiya", "shukla_chaturthi", "shukla_panchami",
    "shukla_shashthi", "shukla_saptami", "shukla_ashtami", "shukla_navami", "shukla_dashami",
    "shukla_ekadashi", "shukla_dwadashi", "shukla_trayodashi", "shukla_chaturdashi", "purnima",
    "krishna_pratipada", "krishna_dwitiya", "krishna_tritiya", "krishna_chaturthi", "krishna_panchami",
    "krishna_shashthi", "krishna_saptami", "krishna_ashtami", "krishna_navami", "krishna_dashami",
    "krishna_ekadashi", "krishna_dwadashi", "krishna_trayodashi", "krishna_chaturdashi", "amavasya"
]

YOGAS = [
    "vishkambha", "priti", "ayushman", "saubhagya", "shobhana", "atiganda", "sukarman", "dhriti",
    "shula", "ganda", "vriddhi", "dhruva", "vyaghata", "harshana", "vajra", "siddhi", "vyatipata",
    "variyana", "parigha", "shiva", "siddha", "sadhya", "shubha", "shukla", "brahma", "indra", "vaidhriti"
]

KARANAS = {
    1: "kimstughna", 2: "bava", 3: "balava", 4: "kaulava", 5: "taitila", 6: "garija", 7: "vanija",
    8: "vishti_bhadra", 9: "bava", 10: "balava", 11: "kaulava"
}
FIXED_KARANAS = ["kimstughna", "shakuni", "chatushpada", "naga"]
MOVABLE_KARANAS = ["bava", "balava", "kaulava", "taitila", "garija", "vanija", "vishti_bhadra"]

# For Detailed Panchang
HINDU_MONTHS = [
    "chaitra", "vaisakha", "jyeshtha", "ashadha", "shravana", "bhadrapada",
    "ashvina", "kartika", "margashirsha", "pausha", "magha", "phalguna"
]

RITU_MAP = {
    "vaisakha": "vasanta", "jyeshtha": "vasanta",
    "ashadha": "grishma", "shravana": "grishma",
    "bhadrapada": "varsha", "ashvina": "varsha",
    "kartika": "sharad", "margashirsha": "sharad",
    "pausha": "hemanta", "magha": "hemanta",
    "phalguna": "shishira", "chaitra": "shishira"
}

DISHA_SHULA = {
    0: "east",       # Monday
    1: "north",      # Tuesday
    2: "north",      # Wednesday
    3: "south",      # Thursday
    4: "west",       # Friday
    5: "east",       # Saturday
    6: "west"        # Sunday
}

CHOGHADIYA_DAY = {
    0: ["amrit", "kaal", "shubh", "rog", "udveg", "chal", "labh", "amrit"], # Monday
    1: ["rog", "udveg", "chal", "labh", "amrit", "kaal", "shubh", "rog"],   # Tuesday
    2: ["labh", "amrit", "kaal", "shubh", "rog", "udveg", "chal", "labh"],  # Wednesday
    3: ["shubh", "rog", "udveg", "chal", "labh", "amrit", "kaal", "shubh"], # Thursday
    4: ["chal", "labh", "amrit", "kaal", "shubh", "rog", "udveg", "chal"],  # Friday
    5: ["kaal", "shubh", "rog", "udveg", "chal", "labh", "amrit", "kaal"],  # Saturday
    6: ["udveg", "chal", "labh", "amrit", "kaal", "shubh", "rog", "udveg"]  # Sunday
}

CHOGHADIYA_NIGHT = {
    0: ["chal", "rog", "kaal", "labh", "udveg", "shubh", "amrit", "chal"],  # Monday
    1: ["kaal", "labh", "udveg", "shubh", "amrit", "chal", "rog", "kaal"],   # Tuesday
    2: ["udveg", "shubh", "amrit", "chal", "rog", "kaal", "labh", "udveg"],  # Wednesday
    3: ["amrit", "chal", "rog", "kaal", "labh", "udveg", "shubh", "amrit"], # Thursday
    4: ["rog", "kaal", "labh", "udveg", "shubh", "amrit", "chal", "rog"],  # Friday
    5: ["labh", "udveg", "shubh", "amrit", "chal", "rog", "kaal", "labh"],  # Saturday
    6: ["shubh", "amrit", "chal", "rog", "kaal", "labh", "udveg", "shubh"]  # Sunday
}

CHOGHADIYA_NATURE = {
    "amrit": "auspicious", "shubh": "auspicious", "labh": "auspicious",
    "chal": "neutral", "udveg": "inauspicious", "rog": "inauspicious", "kaal": "inauspicious"
}
```

### db_utils.py


```python
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_supabase_connection():
    """
    Establishes a connection to the PostgreSQL database using the DATABASE_URL environment variable.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL environment variable not set.")
        return None

    try:
        conn = psycopg2.connect(db_url)
        print("Connection to database successful.")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_system_data(system_id):
    """
    Fetches system data from the knowledge_base_systems table.
    """
    conn = None
    try:
        conn = get_supabase_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT data_en FROM knowledge_base_systems WHERE system_id = %s", (system_id,))
            result = cur.fetchone()
            if result:
                return result[0]
            else:
                return None
    except Exception as e:
        print(f"Error fetching system data: {e}")
        return None
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if conn:
            conn.close()

def fetch_interpretation(category, key, language='en'):
    """
    Fetches interpretation data from the knowledge_base_interpretations table.
    """
    conn = None
    try:
        conn = get_supabase_connection()
        if conn:
            cur = conn.cursor()
            column_name = f"data_{language}"
            cur.execute(f"SELECT {column_name} FROM knowledge_base_interpretations WHERE category = %s AND key = %s", (category, key))
            result = cur.fetchone()
            if result:
                return result[0]
            else:
                return None
    except Exception as e:
        print(f"Error fetching interpretation data: {e}")
        return None
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if conn:
            conn.close()

def fetch_prompt_template(prompt_id, language='en'):
    """
    Fetches LLM prompt templates from the llm_prompts table.
    """
    conn = None
    try:
        conn = get_supabase_connection()
        if conn:
            cur = conn.cursor()
            # Assuming llm_prompts table has prompt_id and template_en columns
            column_name = f"template_{language}"
            cur.execute(f"SELECT {column_name} FROM llm_prompts WHERE prompt_id = %s", (prompt_id,))
            result = cur.fetchone()
            if result:
                return result[0]
            else:
                return ""
    except Exception as e:
        print(f"Error fetching prompt template: {e}")
        return ""
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if conn:
            conn.close()

def fetch_analysis_weights_config(config_name='current_weights'):
    """
    Retrieves the analysis weighting configuration from the analysis_weights_config table.
    """
    conn = None
    try:
        conn = get_supabase_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT weights_data FROM analysis_weights_config WHERE config_name = %s", (config_name,))
            result = cur.fetchone()
            if result:
                return result[0]
            else:
                return {}
    except Exception as e:
        print(f"Error fetching analysis weights config: {e}")
        return {}
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if conn:
            conn.close()

```

### multilingual_strings.py


```python
# multilingual_strings.py
# A centralized repository for all translatable strings.

STRINGS = {
    "en": {
        "signs": {
            "aries": "Aries", "taurus": "Taurus", "gemini": "Gemini", "cancer": "Cancer",
            "leo": "Leo", "virgo": "Virgo", "libra": "Libra", "scorpio": "Scorpio",
            "sagittarius": "Sagittarius", "capricorn": "Capricorn", "aquarius": "Aquarius",
            "pisces": "Pisces"
        },
        "nakshatras": {
            "ashwini": "Ashwini", "bharani": "Bharani", "krittika": "Krittika", "rohini": "Rohini",
            "mrigashira": "Mrigashira", "ardra": "Ardra", "punarvasu": "Punarvasu", "pushya": "Pushya",
            "ashlesha": "Ashlesha", "magha": "Magha", "purva_phalguni": "Purva Phalguni",
            "uttara_phalguni": "Uttara Phalguni", "hasta": "Hasta", "chitra": "Chitra",
            "swati": "Swati", "vishakha": "Vishakha", "anuradha": "Anuradha", "jyeshtha": "Jyeshtha",
            "moola": "Moola", "purva_ashadha": "Purva Ashadha", "uttara_ashadha": "Uttara Ashadha",
            "shravana": "Shravana", "dhanishta": "Dhanishta", "shatabhisha": "Shatabhisha",
            "purva_bhadrapada": "Purva Bhadrapada", "uttara_bhadrapada": "Uttara Bhadrapada",
            "revati": "Revati"
        },
        "planets": {
            "sun": "Sun", "moon": "Moon", "mars": "Mars", "mercury": "Mercury",
            "jupiter": "Jupiter", "venus": "Venus", "saturn": "Saturn",
            "rahu": "Rahu", "ketu": "Ketu"
        },
        "avasthas": {
            "bala": "Bala (Infant)", "kumara": "Kumara (Youthful)", "yuva": "Yuva (Adolescent/Prime)",
            "vriddha": "Vriddha (Aged)", "mrita": "Mrita (Dead)", "n/a": "N/A", "unknown": "Unknown",
            "deepta": "Deepta (Radiant)", "svastha": "Svastha (Healthy)", "pramudita": "Pramudita (Delighted)",
            "shanta": "Shanta (Peaceful)", "duhkhita": "Duhkhita (Miserable)", "garvita": "Garvita (Proud)",
            "lajjita": "Lajjita (Ashamed)", "mudita": "Mudita (Delighted)", "kshobhita": "Kshobhita (Agitated)",
            "kshudhita": "Kshudhita (Starved)"
        },
        "dignities": {
            "exaltation": "Exaltation", "moolatrikona": "Moolatrikona", "own_sign": "Own Sign",
            "friendly_sign": "Friendly Sign", "neutral_sign": "Neutral Sign", "enemy_sign": "Enemy Sign",
            "debilitation": "Debilitation", "node": "Node"
        },
        "doshas": {
            "mangal_dosha": "Mangal Dosha", "kaal_sarp_dosha": "Kaal Sarp Dosha",
            "gandmool_dosha": "Gandmool Dosha"
        },
        "yogas": {
            "gajakesari_yoga": "Gajakesari Yoga", "dharma_karmadhipati_yoga": "Dharma Karmadhipati Yoga"
        },
        "relations": {
            "best_friend": "Best Friend", "friend": "Friend", "neutral": "Neutral",
            "enemy": "Enemy", "bitter_enemy": "Bitter Enemy"
        },
        "positions": {
            "kendra_angular": "Kendra (Angular)", "trikona_trinal": "Trikona (Trinal)",
            "dusthana_malefic": "Dusthana (Malefic)", "upachaya_growth": "Upachaya (Growth)",
            "neutral": "Neutral"
        },
        "status": {
            "normal": "Normal", "retrograde": "Retrograde", "combust": "Combust",
            "vargottama": "Vargottama", "in_planetary_war_victor": "In Planetary War (Victor)",
            "in_planetary_war_defeated": "In Planetary War (Defeated)", "in_planetary_war": "In Planetary War",
            "combust_with_dist": "Combust ({0}°)"
        },
        "report": {
            "astrological_chart_for": "Astrological Chart for",
            "birth_details": "Birth Details", "birth_data": "Birth Data",
            "avakahada_chakra": "Avakahada Chakra (Vedic Foundational Details)",
            "nakshatra_lord": "Nakshatra Lord", "pada_charan": "Pada (Charan)",
            "rasi_sign": "Rasi (Sign)", "lagna_ascendant": "Lagna (Ascendant)",
            "varna_function": "Varna (Function)", "vashya_influence": "Vashya (Influence)",
            "yoni_nature": "Yoni (Nature)", "gana_temperament": "Gana (Temperament)",
            "nadi_constitution": "Nadi (Constitution)",
            "planetary_analysis": "Planetary Analysis",
            "planetary_positions_shadbala": "Planetary Positions & Shadbala",
            "planet": "Planet", "longitude": "Longitude", "d9_sign": "D9 Sign",
            "nakshatra_pada": "Nakshatra (Pada)", "house": "House", "dignity": "Dignity",
            "status": "Status", "ishta": "Ishta", "kashta": "Kashta", "total_shadbala": "Total Shadbala",
            "advanced_planetary_states": "Advanced Planetary States (Avasthas)",
            "baladi_avastha": "Baladi Avastha (Age)", "deeptadi_avastha": "Deeptaadi Avastha (Disposition)",
            "lajjitaadi_avastha": "Lajjitaadi Avastha (Mood)",
            "core_astrological_evaluation": "Core Astrological Evaluation",
            "house_classification_analysis": "House Classification Analysis",
            "debilitation_cancellation_analysis": "Debilitation Cancellation (Neecha Bhanga) Analysis",
            "kemadruma_yoga_analysis": "Kemadruma Yoga (Loneliness of the Moon) Analysis",
            "panchadha_maitri": "Panchadha Maitri (Five-Fold Friendship)",
            "advanced_bhava_analysis": "Advanced House (Bhava) Analysis",
            "bhavesha_strength": "Bhavesha Strength (Condition of House Lords)",
            "planetary_positional_strength": "Planetary Positional Strength (Bhava Madhya & Sandhi)",
            "major_dosha_analysis": "Major Dosha Analysis",
            "yogas_and_special_formations": "Yogas and Special Formations",
            "transit_analysis": "Transit Analysis (Gochar)",
            "sarvashtakavarga_scores": "Sarvashtakavarga Scores",
            "predictive_synthesis": "Predictive Synthesis (Dasha + Gochar)",
            "vimshottari_dasha_periods": "Vimshottari Dasha Periods",
            "mahadasha": "Mahadasha", "antardasha": "Antardasha",
            "pratyantardasha_lord": "Pratyantardasha Lord", "start_date": "Start Date",
            "end_date": "End Date"
        },
        "messages": {
            "is_in": "is in a",
            "is_combust": "Is Combust, weakening its ability to give results.",
            "is_retrograde": "Is Retrograde, causing unconventional or delayed results.",
            "placed_in_house": "Is placed in house **{0}** ({1}).",
            "lord_name_not_found": "Lord of house {0} ({1}) not found.",
            "house_lord": "The lord of house {0}, **{1}**:",
            "debilitation_cancelled": "'s debilitation is **CANCELLED**:",
            "debilitated_planet_reasons": "Its dispositor, {0}, is in a Kendra from the Ascendant.",
            "dispositor_in_kendra_moon": "Its dispositor, {0}, is in a Kendra from the Moon.",
            "exaltation_lord_in_kendra": "The lord of its exaltation sign, {0}, is in a Kendra.",
            "exalted_in_navamsa": "It is exalted in the Navamsa chart (in {0}).",
            "is_retrograde_overcome": "It is retrograde, giving it strength to overcome debilitation.",
            "parivartana_yoga": "It forms a Parivartana Yoga (exchange of signs) with its dispositor, {0}.",
            "kemadruma_cancelled": "Kemadruma Yoga is present but CANCELLED.",
            "overridden_by": "Overridden by:",
            "moon_in_kendra": "The Moon is in a Kendra house.",
            "planets_in_kendra_asc": "Planets are present in Kendra houses from the Ascendant.",
            "planets_in_kendra_moon": "Planets are present in Kendra houses from the Moon.",
            "moon_aspected_by_jupiter": "The Moon is aspected by Jupiter.",
            "moon_exalted_in_navamsa": "The Moon is exalted in the Navamsa chart.",
            "no_planets_adjacent_moon": "No planets are adjacent to the Moon and no powerful cancellations apply.",
            "no_major_doshas": "- No major doshas (Mangal, Kaal Sarp, Gandmool) were identified in the preliminary scan.",
            "no_significant_yogas": "- No other significant yogas were identified.",
            "present_severity": "Present (Severity: **{0}**)",
            "details": "Details:",
            "mitigating_factors": "Mitigating Factors (Weakening the dosha):",
            "amplifying_factors": "Amplifying Factors (Strengthening the dosha):",
            "mars_in_house": "Mars is in the {0}th house.",
            "hemmed_between": "All planets are hemmed between Rahu and Ketu.",
            "moon_in_nakshatra": "The Moon is in {0} nakshatra.",
            "mars_weak_in": "Mars is weak in an {0}.",
            "mars_strong_in": "Mars is strong in its {0}.",
            "mars_conjunct_with": "Mars is conjunct with malefic {0}.",
            "dispositor_strong": "Mars' dispositor, {0}, is strong in {1}.",
            "receives_benefic_aspect": "Receives a powerful benefic aspect from {0}.",
            "receives_malefic_aspect": "Receives a malefic aspect from {0}.",
            "axis_on_1_7": "The axis falls on the 1/7 axis, intensifying the effect on self and relationships.",
            "luminary_conjunct_rahu": "A luminary (Sun or Moon) is conjunct with Rahu/Ketu.",
            "nakshatra_lord_strong": "Nakshatra lord {0} is very strong.",
            "moon_afflicted": "Moon is afflicted by Saturn or Mars.",
            "gajakesari_yoga": "Jupiter is in a Kendra (1,4,7,10) from the Moon.",
            "dharma_karmadhipati_yoga": "A powerful Raja Yoga formed by the association of the 9th and 10th lords.",
            "current_period": "Current Period:",
            "mahadasha_antardasha": "**{0} Mahadasha - {1} Antardasha**.",
            "current_dasha_not_found": "Current Dasha period not found.",
            "saturn_in_bindus": "Saturn is in {0} ({1} bindus).",
            "jupiter_transiting_house": "Jupiter is transiting the {0}th house from your Moon."
        }
    },
    "hi": {
        "signs": {
            "aries": "मेष", "taurus": "वृष", "gemini": "मिथुन", "cancer": "कर्क",
            "leo": "सिंह", "virgo": "कन्या", "libra": "तुला", "scorpio": "वृश्चिक",
            "sagittarius": "धनु", "capricorn": "मकर", "aquarius": "कुंभ",
            "pisces": "मीन"
        },
        "nakshatras": {
            "ashwini": "अश्विनी", "bharani": "भरणी", "krittika": "कृत्तिका", "rohini": "रोहिणी",
            "mrigashira": "मृगशीर्ष", "ardra": "आर्द्रा", "punarvasu": "पुनर्वसु", "pushya": "पुष्य",
            "ashlesha": "आश्लेषा", "magha": "मघा", "purva_phalguni": "पूर्व फाल्गुनी",
            "uttara_phalguni": "उत्तर फाल्गुनी", "hasta": "हस्त", "chitra": "चित्रा",
            "swati": "स्वाति", "vishakha": "विशाखा", "anuradha": "अनुराधा", "jyeshtha": "ज्येष्ठा",
            "moola": "मूल", "purva_ashadha": "पूर्वाषाढ़ा", "uttara_ashadha": "उत्तराषाढ़ा",
            "shravana": "श्रवण", "dhanishta": "धनिष्ठा", "shatabhisha": "शतभिषा",
            "purva_bhadrapada": "पूर्व भाद्रपद", "uttara_bhadrapada": "उत्तर भाद्रपद",
            "revati": "रेवती"
        },
        "planets": {
            "sun": "सूर्य", "moon": "चंद्रमा", "mars": "मंगल", "mercury": "बुध",
            "jupiter": "गुरु", "venus": "शुक्र", "saturn": "शनि",
            "rahu": "राहु", "ketu": "केतु"
        },
        "avasthas": {
            "bala": "बाल (शिशु)", "kumara": "कुमार (युवा)", "yuva": "युवा (किशोरावस्था/प्रमुख)",
            "vriddha": "वृद्ध (बुजुर्ग)", "mrita": "मृत (मृत)", "n/a": "लागू नहीं", "unknown": "अज्ञात",
            "deepta": "दीप्त (चमकदार)", "svastha": "स्वस्थ (स्वस्थ)", "pramudita": "प्रमुदित (प्रसन्न)",
            "shanta": "शांत (शांतिपूर्ण)", "duhkhita": "दुःखित (दुखी)", "garvita": "गर्वित (गर्व)",
            "lajjita": "लज्जित (शर्मिंदा)", "mudita": "मुदित (हर्षित)", "kshobhita": "क्षोभित (परेशान)",
            "kshudhita": "क्षुधित (भूखा)"
        },
        "dignities": {
            "exaltation": "उच्च", "moolatrikona": "मूलत्रिकोण", "own_sign": "स्वराशि",
            "friendly_sign": "मित्र राशि", "neutral_sign": "तटस्थ राशि", "enemy_sign": "शत्रु राशि",
            "debilitation": "नीच", "node": "नोड"
        },
        "doshas": {
            "mangal_dosha": "मांगलिक दोष", "kaal_sarp_dosha": "काल सर्प दोष",
            "gandmool_dosha": "गंडमूल दोष"
        },
        "yogas": {
            "gajakesari_yoga": "गजकेसरी योग", "dharma_karmadhipati_yoga": "धर्म कर्मधिपति योग"
        },
        "relations": {
            "best_friend": "परम मित्र", "friend": "मित्र", "neutral": "तटस्थ",
            "enemy": "शत्रु", "bitter_enemy": "कट्टर शत्रु"
        },
        "positions": {
            "kendra_angular": "केंद्र (कोणीय)", "trikona_trinal": "त्रिकोण",
            "dusthana_malefic": "दुस्थान (अशुभ)", "upachaya_growth": "उपचय (वृद्धि)",
            "neutral": "तटस्थ"
        },
        "status": {
            "normal": "सामान्य", "retrograde": "वक्री", "combust": "अस्त",
            "vargottama": "वर्गोत्तम", "in_planetary_war_victor": "ग्रहीय युद्ध में (विजेता)",
            "in_planetary_war_defeated": "ग्रहीय युद्ध में (पराजित)", "in_planetary_war": "ग्रहीय युद्ध में",
            "combust_with_dist": "अस्त ({0}°)"
        },
        "report": {
            "astrological_chart_for": " का ज्योतिषीय चार्ट",
            "birth_details": "जन्म विवरण", "birth_data": "जन्म डेटा",
            "avakahada_chakra": "अवकहदा चक्र (वैदिक मूलभूत विवरण)",
            "nakshatra_lord": "नक्षत्र स्वामी", "pada_charan": "चरण",
            "rasi_sign": "राशि", "lagna_ascendant": "लग्न",
            "varna_function": "वर्ण (कार्य)", "vashya_influence": "वश्य (प्रभाव)",
            "yoni_nature": "योनि (प्रकृति)", "gana_temperament": "गण (स्वभाव)",
            "nadi_constitution": "नाड़ी (संविधान)",
            "planetary_analysis": "ग्रह विश्लेषण",
            "planetary_positions_shadbala": "ग्रहों की स्थिति और षड्बल",
            "planet": "ग्रह", "longitude": "देशांतर", "d9_sign": "डी9 राशि",
            "nakshatra_pada": "नक्षत्र (चरण)", "house": "भाव", "dignity": "गरिमा",
            "status": "स्थिति", "ishta": "इष्ट", "kashta": "कष्ट", "total_shadbala": "कुल षड्बल",
            "advanced_planetary_states": "उन्नत ग्रहों की अवस्थाएं (अवस्था)",
            "baladi_avastha": "बलादि अवस्था (आयु)", "deeptadi_avastha": "दीप्तदि अवस्था (स्वभाव)",
            "lajjitaadi_avastha": "लज्जितादि अवस्था (मनोदशा)",
            "core_astrological_evaluation": "मुख्य ज्योतिषीय मूल्यांकन",
            "house_classification_analysis": "भाव वर्गीकरण विश्लेषण",
            "debilitation_cancellation_analysis": "नीच भंग (नीच भंग राज योग) विश्लेषण",
            "kemadruma_yoga_analysis": "केमद्रुम योग (चंद्रमा का अकेलापन) विश्लेषण",
            "panchadha_maitri": "पंचधा मैत्री",
            "advanced_bhava_analysis": "उन्नत भाव विश्लेषण",
            "bhavesha_strength": "भवेश की शक्ति (भाव स्वामी की स्थिति)",
            "planetary_positional_strength": "ग्रहों की स्थितिजन्य शक्ति (भाव मध्य और संधि)",
            "major_dosha_analysis": "प्रमुख दोष विश्लेषण",
            "yogas_and_special_formations": "योग और विशेष संरचनाएं",
            "transit_analysis": "गोचर विश्लेषण",
            "sarvashtakavarga_scores": "सर्वाष्टकवर्ग अंक",
            "predictive_synthesis": "भविष्यवाणी संश्लेषण (दशा + गोचर)",
            "vimshottari_dasha_periods": "विंशोत्तरी दशा काल",
            "mahadasha": "महादशा", "antardasha": "अंतर्दशा",
            "pratyantardasha_lord": "प्रत्यंतरदशा स्वामी", "start_date": "आरंभ तिथि",
            "end_date": "समाप्ति तिथि"
        },
        "messages": {
            "is_in": "में है",
            "is_combust": "अस्त है, जो परिणाम देने की उसकी क्षमता को कमजोर करता है।",
            "is_retrograde": "वक्री है, जिससे अपरंपरागत या विलंबित परिणाम होते हैं।",
            "placed_in_house": "भाव **{0}** ({1}) में स्थित है।",
            "lord_name_not_found": "भाव {0} का स्वामी ({1}) नहीं मिला।",
            "house_lord": "भाव {0} का स्वामी, **{1}**:",
            "debilitation_cancelled": " का नीचत्व **भंग** हो गया है:",
            "debilitated_planet_reasons": "इसका स्वामी, {0}, लग्न से केंद्र में है।",
            "dispositor_in_kendra_moon": "इसका स्वामी, {0}, चंद्रमा से केंद्र में है।",
            "exaltation_lord_in_kendra": "इसके उच्च राशि का स्वामी, {0}, केंद्र में है।",
            "exalted_in_navamsa": "यह नवमांश चार्ट में उच्च का है ( {0} में)।",
            "is_retrograde_overcome": "यह वक्री है, जो इसे नीचत्व पर काबू पाने की शक्ति देता है।",
            "parivartana_yoga": "यह अपने स्वामी, {0}, के साथ एक परिवर्तन योग बनाता है।",
            "kemadruma_cancelled": "केमद्रुम योग मौजूद है लेकिन **भंग हो गया है**।",
            "overridden_by": "इसके द्वारा अधिग्रहित:",
            "moon_in_kendra": "चंद्रमा केंद्र भाव में है।",
            "planets_in_kendra_asc": "लग्न से केंद्र भावों में ग्रह मौजूद हैं।",
            "planets_in_kendra_moon": "चंद्रमा से केंद्र भावों में ग्रह मौजूद हैं।",
            "moon_aspected_by_jupiter": "चंद्रमा को गुरु से दृष्टि मिलती है।",
            "moon_exalted_in_navamsa": "चंद्रमा नवमांश चार्ट में उच्च का है।",
            "no_planets_adjacent_moon": "चंद्रमा के निकट कोई ग्रह नहीं हैं और कोई शक्तिशाली भंग लागू नहीं है।",
            "no_major_doshas": "- प्रारंभिक जांच में कोई प्रमुख दोष (मांगलिक, काल सर्प, गंडमूल) नहीं पाए गए।",
            "no_significant_yogas": "- कोई अन्य महत्वपूर्ण योग नहीं पाए गए।",
            "present_severity": "उपस्थित (गंभीरता: **{0}**)",
            "details": "विवरण:",
            "mitigating_factors": "शमन कारक (दोष को कमजोर करने वाले):",
            "amplifying_factors": "बढ़ाने वाले कारक (दोष को मजबूत करने वाले):",
            "mars_in_house": "मंगल {0}वें भाव में है।",
            "hemmed_between": "सभी ग्रह राहु और केतु के बीच घिरे हुए हैं।",
            "moon_in_nakshatra": "चंद्रमा {0} नक्षत्र में है।",
            "mars_weak_in": "मंगल {0} में कमजोर है।",
            "mars_strong_in": "मंगल {0} में मजबूत है।",
            "mars_conjunct_with": "मंगल की अशुभ ग्रह {0} के साथ युति है।",
            "dispositor_strong": "मंगल का स्वामी, {0}, {1} में मजबूत है।",
            "receives_benefic_aspect": "को शक्तिशाली शुभ ग्रह {0} से दृष्टि मिलती है।",
            "receives_malefic_aspect": "को अशुभ ग्रह {0} से दृष्टि मिलती है।",
            "axis_on_1_7": "अक्ष 1/7 भाव पर पड़ता है, जिससे स्वयं और रिश्तों पर प्रभाव बढ़ता है।",
            "luminary_conjunct_rahu": "एक प्रकाशमान ग्रह (सूर्य या चंद्रमा) राहु/केतु के साथ युति में है।",
            "nakshatra_lord_strong": "नक्षत्र स्वामी {0} बहुत मजबूत है।",
            "moon_afflicted": "चंद्रमा शनि या मंगल से पीड़ित है।",
            "gajakesari_yoga": "गुरु चंद्रमा से केंद्र (1,4,7,10) में है।",
            "dharma_karmadhipati_yoga": "9वें और 10वें भाव के स्वामियों के संबंध से बना एक शक्तिशाली राज योग।",
            "current_period": "वर्तमान अवधि:",
            "mahadasha_antardasha": "**{0} महादशा - {1} अंतर्दशा**।",
            "current_dasha_not_found": "वर्तमान दशा काल नहीं मिला।",
            "saturn_in_bindus": "शनि {0} में है ({1} बिंदु)।",
            "jupiter_transiting_house": "गुरु आपके चंद्रमा से {0}वें भाव में गोचर कर रहा है।"
        }
    }
}
```

### database_schema.md


```markdown
--- Inspecting table: user ---
Columns:
  - id: INTEGER (nullable: False)
  - email: VARCHAR(120) (nullable: False)
  - password_hash: VARCHAR(128) (nullable: False)
  - created_at: TIMESTAMP (nullable: True)
Primary Key: id
Foreign Keys:
  No foreign keys found.
Indexes:
  - user_email_key (unique: True): email
Unique Constraints:
  - user_email_key: (columns not available or empty)

--- Inspecting table: cosmic_circle ---
Columns:
  - id: INTEGER (nullable: False)
  - name: VARCHAR(150) (nullable: False)
  - description: TEXT (nullable: True)
  - circle_type: VARCHAR(50) (nullable: False)
  - created_at: TIMESTAMP (nullable: True)
Primary Key: id
Foreign Keys:
  No foreign keys found.
Indexes:
  - cosmic_circle_name_key (unique: True): name
Unique Constraints:
  - cosmic_circle_name_key: (columns not available or empty)

--- Inspecting table: llm_prompts ---
Columns:
  - prompt_id: VARCHAR(255) (nullable: False)
  - trigger_type: TEXT (nullable: False)
  - template_en: TEXT (nullable: False)
  - template_hi: TEXT (nullable: False)
  - template_hinglish: TEXT (nullable: False)
Primary Key: prompt_id
Foreign Keys:
  No foreign keys found.
Indexes:
  No indexes found.
Unique Constraints:
  No unique constraints found.

--- Inspecting table: analysis_weights_config ---
Columns:
  - config_name: VARCHAR(255) (nullable: False)
  - weights_data: JSONB (nullable: False)
Primary Key: config_name
Foreign Keys:
  No foreign keys found.
Indexes:
  No indexes found.
Unique Constraints:
  No unique constraints found.

--- Inspecting table: alembic_version ---
Columns:
  - version_num: VARCHAR(32) (nullable: False)
Primary Key: version_num
Foreign Keys:
  No foreign keys found.
Indexes:
  No indexes found.
Unique Constraints:
  No unique constraints found.

--- Inspecting table: user_chart ---
Columns:
  - id: INTEGER (nullable: False)
  - user_id: INTEGER (nullable: False)
  - birth_data: JSON (nullable: False)
  - chart_json: JSONB (nullable: False)
  - created_at: TIMESTAMP (nullable: True)
  - updated_at: TIMESTAMP (nullable: False)
  - relation_type: VARCHAR(50) (nullable: False)
  - is_astrologer: BOOLEAN (nullable: False)
  - astrologer_id: VARCHAR(36) (nullable: True)
Primary Key: id
Foreign Keys:
  - From columns: ['astrologer_id'] to table: astrologer_profile on columns: ['id']
Indexes:
  - _user_relation_uc (unique: True): user_id, relation_type
Unique Constraints:
  - _user_relation_uc: (columns not available or empty)

--- Inspecting table: knowledge_base_interpretations ---
Columns:
  - category: TEXT (nullable: False)
  - key: TEXT (nullable: False)
  - data_en: JSONB (nullable: True)
  - data_hi: JSONB (nullable: True)
  - data_hinglish: JSONB (nullable: True)
Primary Key: category, key
Foreign Keys:
  No foreign keys found.
Indexes:
  No indexes found.
Unique Constraints:
  No unique constraints found.

--- Inspecting table: knowledge_base_systems ---
Columns:
  - system_name: VARCHAR(255) (nullable: False)
  - data_en: JSONB (nullable: True)
  - data_hi: JSONB (nullable: True)
  - data_hinglish: JSONB (nullable: True)
Primary Key: system_name
Foreign Keys:
  No foreign keys found.
Indexes:
  No indexes found.
Unique Constraints:
  No unique constraints found.

--- Inspecting table: knowledge_base_systems_backup ---
Columns:
  - id: BIGINT (nullable: True)
  - system_name: TEXT (nullable: True)
  - data: JSONB (nullable: True)
  - created_at: TIMESTAMP (nullable: True)
No Primary Key found.
Foreign Keys:
  No foreign keys found.
Indexes:
  No indexes found.
Unique Constraints:
  No unique constraints found.

--- Inspecting table: social_post ---
Columns:
  - id: INTEGER (nullable: False)
  - source_platform: VARCHAR(50) (nullable: False)
  - unique_post_id: VARCHAR(255) (nullable: False)
  - direct_url: VARCHAR(500) (nullable: False)
  - author: VARCHAR(100) (nullable: False)
  - content: TEXT (nullable: False)
  - image_url: VARCHAR(500) (nullable: True)
  - original_timestamp: TIMESTAMP (nullable: False)
  - analysis_status: VARCHAR(50) (nullable: False)
  - sentiment: VARCHAR(50) (nullable: True)
Primary Key: id
Foreign Keys:
  No foreign keys found.
Indexes:
  - social_post_unique_post_id_key (unique: True): unique_post_id
Unique Constraints:
  - social_post_unique_post_id_key: (columns not available or empty)

--- Inspecting table: keyword ---
Columns:
  - id: INTEGER (nullable: False)
  - term: VARCHAR(100) (nullable: False)
  - is_active: BOOLEAN (nullable: False)
Primary Key: id
Foreign Keys:
  No foreign keys found.
Indexes:
  - keyword_term_key (unique: True): term
Unique Constraints:
  - keyword_term_key: (columns not available or empty)

--- Inspecting table: data_source ---
Columns:
  - id: INTEGER (nullable: False)
  - platform: VARCHAR(50) (nullable: False)
  - identifier: VARCHAR(255) (nullable: False)
  - is_active: BOOLEAN (nullable: False)
Primary Key: id
Foreign Keys:
  No foreign keys found.
Indexes:
  - data_source_identifier_key (unique: True): identifier
Unique Constraints:
  - data_source_identifier_key: (columns not available or empty)

--- Inspecting table: admin_user ---
Columns:
  - id: INTEGER (nullable: False)
  - email: VARCHAR(120) (nullable: False)
  - password_hash: VARCHAR(128) (nullable: False)
  - role: VARCHAR(50) (nullable: False)
  - permissions: VARCHAR(255) (nullable: False)
  - created_at: TIMESTAMP (nullable: False)
Primary Key: id
Foreign Keys:
  No foreign keys found.
Indexes:
  - admin_user_email_key (unique: True): email
Unique Constraints:
  - admin_user_email_key: (columns not available or empty)

--- Inspecting table: video ---
Columns:
  - id: INTEGER (nullable: False)
  - video_id: VARCHAR(120) (nullable: False)
  - title: VARCHAR(255) (nullable: False)
  - channel_id: VARCHAR(120) (nullable: False)
```

### akundli_report_FINAL.md


```markdown
# Astrological Chart for Manish

## Birth Details
### Birth Data
- **Name**: Manish
- **Year**: 1972
- **Month**: 6
- **Day**: 5
- **Hour**: 20
- **Minute**: 0
- **Second**: 0
- **Latitude**: 27.8974
- **Longitude**: 78.088
- **Timezone Str**: Asia/Kolkata

## Avakahada Chakra (Vedic Foundational Details)
- **Nakshatra**: Purva Bhadrapada
- **Pada (Charan)**: 4
- **Rasi (Sign)**: Pisces
- **Lagna (Ascendant)**: Sagittarius
- **Varna (Function)**: Brahmin
- **Vashya (Influence)**: Jalachara
- **Yoni (Nature)**: Lion
- **Gana (Temperament)**: Manushya
- **Nadi (Constitution)**: Adi
- **Nakshatra Lord**: Jupiter
## Panchang at Birth
| Limb | Value | Lord | Notes |
|---|---|---|---|
| Tithi | Krishna Navami | — | Paksha: Krishna |
| Vara | Monday | Moon | — |
| Nakshatra | purva_bhadrapada, Pada 4 | Nakshatra Lord: Jupiter | — |
| Yoga | Priti | — | — |
| Karana | Taitila | — | — |


---

## Planetary Analysis
### Planetary Positions & Shadbala
| Planet | Longitude | Rasi D1 | D9 Sign | Nakshatra (Pada) | House | Dignity | Status | Ishta | Kashta | Total Shadbala |
|---|---|---|---|---|---|---|---|---|---|---|
| **Sun** | 51° 31' 48" / 21° 31' 48" taurus | Taurus | **Cancer** | Rohini (4) | 6 | Enemy Sign | Normal | 0.0 | 4.71 | **1.06** |
| **Moon** | 330° 42' 51" / 0° 42' 51" pisces | Pisces | **Cancer** | Purva Bhadrapada (4) | 4 | Neutral Sign | Normal | 0.0 | 4.55 | **1.98** |
| **Mars** | 81° 52' 56" / 21° 52' 56" gemini | Gemini | **Aries** | Punarvasu (1) | 7 | Enemy Sign | Normal | 0.0 | 4.71 | **0.76** |
| **Mercury** | 52° 25' 12" / 22° 25' 12" taurus | Taurus | **Cancer** | Rohini (4) | 6 | Friendly Sign | Combust (0.89°) | 0.0 | 4.21 | **0.8** |
| **Jupiter** | 252° 19' 34" / 12° 19' 34" sagittarius | Sagittarius | **Cancer** | Moola (4) | 1 | Moolatrikona | Retrograde | 4.21 | 0.0 | **3.15** |
| **Venus** | 69° 28' 48" / 9° 28' 48" gemini | Gemini | **Sagittarius** | Ardra (1) | 7 | Friendly Sign | Retrograde | 2.43 | 0.0 | **1.5** |
| **Saturn** | 47° 9' 51" / 17° 9' 51" taurus | Taurus | **Gemini** | Rohini (3) | 6 | Friendly Sign | Combust (4.37°) | 0.0 | 4.21 | **0.35** |
| **Rahu** | 274° 51' 52" / 4° 51' 52" capricorn | Capricorn | **Aquarius** | Uttara Ashadha (3) | 2 | Node | Normal | N/A | N/A | **0.0** |
| **Ketu** | 94° 51' 52" / 4° 51' 52" cancer | Cancer | **Leo** | Pushya (1) | 8 | Node | Normal | N/A | N/A | **0.0** |

## Chandra Lagna Chart
- Moon as Lagna (Ascendant): **pisces**

| House | Sign | Planets |
|---|---|---|
| 1 | pisces | Moon |
| 2 | aries | — |
| 3 | taurus | Sun, Mercury, Saturn |
| 4 | gemini | Mars, Venus |
| 5 | cancer | Ketu |
| 6 | leo | — |
| 7 | virgo | — |
| 8 | libra | — |
| 9 | scorpio | — |
| 10 | sagittarius | Jupiter |
| 11 | capricorn | Rahu |
| 12 | aquarius | — |

---

## Advanced Planetary States (Avasthas)
| Planet | Baladi Avastha (Age) | Deeptaadi Avastha (Disposition) | Lajjitaadi Avastha (Mood) |
|---|---|---|---|
| **Sun** | Kumara (Youthful) | Duhkhita (Miserable) | **Kshudhita (Starved)** |
| **Moon** | Mrita (Dead) | Shanta (Peaceful) | **Shanta (Peaceful)** |
| **Mars** | Vriddha (Aged) | Duhkhita (Miserable) | **Kshudhita (Starved)** |
| **Mercury** | Kumara (Youthful) | Pramudita (Delighted) | **Mudita (Delighted)** |
| **Jupiter** | Yuva (Adolescent/Prime) | Unknown | **Garvita (Proud)** |
| **Venus** | Kumara (Youthful) | Pramudita (Delighted) | **Mudita (Delighted)** |
| **Saturn** | Yuva (Adolescent/Prime) | Pramudita (Delighted) | **Mudita (Delighted)** |

---

## Core Astrological Evaluation
### House Classification Analysis
- **Sun** Planet In House 6: Dusthana (Malefic), Upachaya (Growth)
- **Moon** Planet In House 4: Kendra (Angular)
- **Mars** Planet In House 7: Kendra (Angular)
- **Mercury** Planet In House 6: Dusthana (Malefic), Upachaya (Growth)
- **Jupiter** Planet In House 1: Kendra (Angular), Trikona (Trinal)
- **Venus** Planet In House 7: Kendra (Angular)
- **Saturn** Planet In House 6: Dusthana (Malefic), Upachaya (Growth)
- **Rahu** Planet In House 2: Neutral
- **Ketu** Planet In House 8: Dusthana (Malefic)

### Debilitation Cancellation (Neecha Bhanga) Analysis
No Debilitated Planets

### Kemadruma Yoga (Loneliness of the Moon) Analysis
- **Kemadruma Cancelled** Overridden by:
  - Planets In Kendra Asc
  - Moon In Kendra

---

## Planetary Positional Strength (Bhava Madhya & Sandhi)
| Planet | Planetary Positional Strength (Bhava Madhya & Sandhi) |
|---|---|
| **Sun** | Very Strong At Bhava Madhya |
| **Moon** | Normal Position |
| **Mars** | Very Strong At Bhava Madhya |
| **Mercury** | Very Strong At Bhava Madhya |
| **Jupiter** | Normal Position |
| **Venus** | Normal Position |
| **Saturn** | Normal Position |
| **Rahu** | Weak At Bhava Sandhi |
| **Ketu** | Weak At Bhava Sandhi |

---

## Major Dosha Analysis
### Mangal Dosha Analysis
- **Status**: Present (Severity: **High**)
- **Amplifying Factors (Strengthening the dosha):**
  - Mars is weak in an Enemy Sign.

---

## Yogas Conjunctions
- **Kemadruma Yoga Cancelled**
  - Relevant Planets: Moon
  - Relevant Houses: 4
- **Dharma Karmadhipati Yoga**
  - Relevant Planets: Sun, Mercury
  - Relevant Houses: 6
  - Formation Rule Met: Yes
- **Mercury-Sun Conjunction**
  - Relevant Planets: Mercury, Sun
  - Relevant Houses: 6
- **Saturn-Sun Conjunction**
  - Relevant Planets: Saturn, Sun
  - Relevant Houses: 6

---

## Transit Analysis (Gochar)
### Peak Phase
- **Status**: Saturn In Bindus
### Jupiter Transiting House 4
- **Status**: Jupiter Transiting House

---

## Ashtakavarga Scores
### Sarvashtakavarga
| Sign | Bindus |
|---|---|
| Aries | 28 |
| Taurus | 29 |
| Gemini | 16 |
| Cancer | 27 |
| Leo | 24 |
| Virgo | 31 |
| Libra | 35 |
| Scorpio | 22 |
| Sagittarius | 26 |
| Capricorn | 28 |
| Aquarius | 28 |
| Pisces | 38 |

---

### Bhinnashtakavarga
#### Sun
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 3 | 6 | 3 | 2 | 4 | 3 | 3 | 4 | 5 | 5 | 5 | 5 |

#### Moon
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 2 | 3 | 1 | 5 | 4 | 6 | 6 | 4 | 5 | 1 | 5 | 7 |

#### Mars
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 2 | 4 | 1 | 3 | 1 | 5 | 4 | 2 | 2 | 3 | 3 | 4 |

#### Mercury
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 5 | 4 | 4 | 5 | 3 | 5 | 6 | 3 | 4 | 6 | 4 | 5 |

#### Jupiter
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 6 | 4 | 4 | 5 | 4 | 6 | 5 | 3 | 4 | 6 | 3 | 6 |

#### Venus
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 6 | 2 | 2 | 6 | 5 | 4 | 6 | 2 | 3 | 5 | 5 | 6 |

#### Saturn
| Sign | Aries | Taurus | Gemini | Cancer | Leo | Virgo | Libra | Scorpio | Sagittarius | Capricorn | Aquarius | Pisces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bindus | 4 | 6 | 1 | 1 | 3 | 2 | 5 | 4 | 3 | 2 | 3 | 5 |

## Vimshottari Dasha

### Mahadashas
| Dasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Jupiter** | 1972-06-05 | 1975-07-27 | 1148 |
| **Saturn** | 1975-07-27 | 1994-07-27 | 6940 |
| **Mercury** | 1994-07-27 | 2011-07-27 | 6209 |
| **Ketu** | 2011-07-27 | 2018-07-27 | 2557 |
| **Venus** | 2018-07-27 | 2038-07-27 | 7305 |
| **Sun** | 2038-07-27 | 2044-07-26 | 2191 |
| **Moon** | 2044-07-26 | 2054-07-27 | 3652 |
| **Mars** | 2054-07-27 | 2061-07-26 | 2557 |
| **Rahu** | 2061-07-26 | 2079-07-27 | 6574 |

### All Mahadashas With Antardashas Pratyantardashas
#### Mahadasha: Jupiter (1972-06-05 - 1975-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Jupiter** | 1972-06-05 | 1972-11-05 | 153 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 1972-06-05 | 1972-06-25 | 20 |
| Saturn | 1972-06-25 | 1972-07-19 | 24 |
| Mercury | 1972-07-19 | 1972-08-10 | 22 |
| Ketu | 1972-08-10 | 1972-08-19 | 9 |
| Venus | 1972-08-19 | 1972-09-13 | 26 |
| Sun | 1972-09-13 | 1972-09-21 | 8 |
| Moon | 1972-09-21 | 1972-10-04 | 13 |
| Mars | 1972-10-04 | 1972-10-13 | 9 |
| Rahu | 1972-10-13 | 1972-11-05 | 23 |
| **Saturn** | 1972-11-05 | 1973-05-05 | 182 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 1972-11-05 | 1972-12-03 | 29 |
| Mercury | 1972-12-03 | 1972-12-29 | 26 |
| Ketu | 1972-12-29 | 1973-01-09 | 11 |
| Venus | 1973-01-09 | 1973-02-08 | 30 |
| Sun | 1973-02-08 | 1973-02-17 | 9 |
| Moon | 1973-02-17 | 1973-03-04 | 15 |
| Mars | 1973-03-04 | 1973-03-15 | 11 |
| Rahu | 1973-03-15 | 1973-04-11 | 27 |
| Jupiter | 1973-04-11 | 1973-05-05 | 24 |
| **Mercury** | 1973-05-05 | 1973-10-15 | 163 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 1973-05-05 | 1973-05-28 | 23 |
| Ketu | 1973-05-28 | 1973-06-07 | 9 |
| Venus | 1973-06-07 | 1973-07-04 | 27 |
| Sun | 1973-07-04 | 1973-07-12 | 8 |
| Moon | 1973-07-12 | 1973-07-26 | 14 |
| Mars | 1973-07-26 | 1973-08-04 | 9 |
| Rahu | 1973-08-04 | 1973-08-28 | 24 |
| Jupiter | 1973-08-28 | 1973-09-19 | 22 |
| Saturn | 1973-09-19 | 1973-10-15 | 26 |
| **Ketu** | 1973-10-15 | 1973-12-21 | 67 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 1973-10-15 | 1973-10-19 | 4 |
| Venus | 1973-10-19 | 1973-10-30 | 11 |
| Sun | 1973-10-30 | 1973-11-02 | 3 |
| Moon | 1973-11-02 | 1973-11-08 | 6 |
| Mars | 1973-11-08 | 1973-11-12 | 4 |
| Rahu | 1973-11-12 | 1973-11-22 | 10 |
| Jupiter | 1973-11-22 | 1973-12-01 | 9 |
| Saturn | 1973-12-01 | 1973-12-11 | 11 |
| Mercury | 1973-12-11 | 1973-12-21 | 9 |
| **Venus** | 1973-12-21 | 1974-06-30 | 191 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 1973-12-21 | 1974-01-22 | 32 |
| Sun | 1974-01-22 | 1974-01-31 | 10 |
| Moon | 1974-01-31 | 1974-02-16 | 16 |
| Mars | 1974-02-16 | 1974-02-27 | 11 |
| Rahu | 1974-02-27 | 1974-03-28 | 29 |
| Jupiter | 1974-03-28 | 1974-04-23 | 26 |
| Saturn | 1974-04-23 | 1974-05-23 | 30 |
| Mercury | 1974-05-23 | 1974-06-19 | 27 |
| Ketu | 1974-06-19 | 1974-06-30 | 11 |
| **Sun** | 1974-06-30 | 1974-08-27 | 57 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 1974-06-30 | 1974-07-03 | 3 |
| Moon | 1974-07-03 | 1974-07-08 | 5 |
| Mars | 1974-07-08 | 1974-07-11 | 3 |
| Rahu | 1974-07-11 | 1974-07-20 | 9 |
| Jupiter | 1974-07-20 | 1974-07-27 | 8 |
| Saturn | 1974-07-27 | 1974-08-06 | 9 |
| Mercury | 1974-08-06 | 1974-08-14 | 8 |
| Ketu | 1974-08-14 | 1974-08-17 | 3 |
| Venus | 1974-08-17 | 1974-08-27 | 10 |
| **Moon** | 1974-08-27 | 1974-11-30 | 96 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 1974-08-27 | 1974-09-04 | 8 |
| Mars | 1974-09-04 | 1974-09-09 | 6 |
| Rahu | 1974-09-09 | 1974-09-24 | 14 |
| Jupiter | 1974-09-24 | 1974-10-06 | 13 |
| Saturn | 1974-10-06 | 1974-10-21 | 15 |
| Mercury | 1974-10-21 | 1974-11-04 | 14 |
| Ketu | 1974-11-04 | 1974-11-10 | 6 |
| Venus | 1974-11-10 | 1974-11-25 | 16 |
| Sun | 1974-11-25 | 1974-11-30 | 5 |
| **Mars** | 1974-11-30 | 1975-02-05 | 67 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 1974-11-30 | 1974-12-04 | 4 |
| Rahu | 1974-12-04 | 1974-12-14 | 10 |
| Jupiter | 1974-12-14 | 1974-12-23 | 9 |
| Saturn | 1974-12-23 | 1975-01-03 | 11 |
| Mercury | 1975-01-03 | 1975-01-12 | 9 |
| Ketu | 1975-01-12 | 1975-01-16 | 4 |
| Venus | 1975-01-16 | 1975-01-27 | 11 |
| Sun | 1975-01-27 | 1975-01-31 | 3 |
| Moon | 1975-01-31 | 1975-02-05 | 6 |
| **Rahu** | 1975-02-05 | 1975-07-27 | 172 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 1975-02-05 | 1975-03-03 | 26 |
| Jupiter | 1975-03-03 | 1975-03-26 | 23 |
| Saturn | 1975-03-26 | 1975-04-22 | 27 |
| Mercury | 1975-04-22 | 1975-05-17 | 24 |
| Ketu | 1975-05-17 | 1975-05-27 | 10 |
| Venus | 1975-05-27 | 1975-06-24 | 29 |
| Sun | 1975-06-24 | 1975-07-03 | 9 |
| Moon | 1975-07-03 | 1975-07-17 | 14 |
| Mars | 1975-07-17 | 1975-07-27 | 10 |

#### Mahadasha: Saturn (1975-07-27 - 1994-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Saturn** | 1975-07-27 | 1978-07-30 | 1099 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 1975-07-27 | 1976-01-17 | 174 |
| Mercury | 1976-01-17 | 1976-06-21 | 156 |
| Ketu | 1976-06-21 | 1976-08-24 | 64 |
| Venus | 1976-08-24 | 1977-02-23 | 183 |
| Sun | 1977-02-23 | 1977-04-19 | 55 |
| Moon | 1977-04-19 | 1977-07-20 | 92 |
| Mars | 1977-07-20 | 1977-09-22 | 64 |
| Rahu | 1977-09-22 | 1978-03-06 | 165 |
| Jupiter | 1978-03-06 | 1978-07-30 | 147 |
| **Mercury** | 1978-07-30 | 1981-04-08 | 983 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 1978-07-30 | 1978-12-16 | 139 |
| Ketu | 1978-12-16 | 1979-02-12 | 57 |
| Venus | 1979-02-12 | 1979-07-26 | 164 |
| Sun | 1979-07-26 | 1979-09-13 | 49 |
| Moon | 1979-09-13 | 1979-12-04 | 82 |
| Mars | 1979-12-04 | 1980-01-30 | 57 |
| Rahu | 1980-01-30 | 1980-06-26 | 147 |
| Jupiter | 1980-06-26 | 1980-11-04 | 131 |
| Saturn | 1980-11-04 | 1981-04-08 | 156 |
| **Ketu** | 1981-04-08 | 1982-05-18 | 405 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 1981-04-08 | 1981-05-02 | 24 |
| Venus | 1981-05-02 | 1981-07-08 | 67 |
| Sun | 1981-07-08 | 1981-07-29 | 20 |
| Moon | 1981-07-29 | 1981-08-31 | 34 |
| Mars | 1981-08-31 | 1981-09-24 | 24 |
| Rahu | 1981-09-24 | 1981-11-24 | 61 |
| Jupiter | 1981-11-24 | 1982-01-17 | 54 |
| Saturn | 1982-01-17 | 1982-03-22 | 64 |
| Mercury | 1982-03-22 | 1982-05-18 | 57 |
| **Venus** | 1982-05-18 | 1985-07-18 | 1157 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 1982-05-18 | 1982-11-27 | 193 |
| Sun | 1982-11-27 | 1983-01-24 | 58 |
| Moon | 1983-01-24 | 1983-04-30 | 96 |
| Mars | 1983-04-30 | 1983-07-07 | 67 |
| Rahu | 1983-07-07 | 1983-12-27 | 173 |
| Jupiter | 1983-12-27 | 1984-05-29 | 154 |
| Saturn | 1984-05-29 | 1984-11-28 | 183 |
| Mercury | 1984-11-28 | 1985-05-11 | 164 |
| Ketu | 1985-05-11 | 1985-07-18 | 67 |
| **Sun** | 1985-07-18 | 1986-06-30 | 347 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 1985-07-18 | 1985-08-04 | 17 |
| Moon | 1985-08-04 | 1985-09-02 | 29 |
| Mars | 1985-09-02 | 1985-09-22 | 20 |
| Rahu | 1985-09-22 | 1985-11-13 | 52 |
| Jupiter | 1985-11-13 | 1985-12-30 | 46 |
| Saturn | 1985-12-30 | 1986-02-22 | 55 |
| Mercury | 1986-02-22 | 1986-04-13 | 49 |
| Ketu | 1986-04-13 | 1986-05-03 | 20 |
| Venus | 1986-05-03 | 1986-06-30 | 58 |
| **Moon** | 1986-06-30 | 1988-01-29 | 578 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 1986-06-30 | 1986-08-17 | 48 |
| Mars | 1986-08-17 | 1986-09-20 | 34 |
| Rahu | 1986-09-20 | 1986-12-15 | 87 |
| Jupiter | 1986-12-15 | 1987-03-02 | 77 |
| Saturn | 1987-03-02 | 1987-06-02 | 92 |
| Mercury | 1987-06-02 | 1987-08-23 | 82 |
| Ketu | 1987-08-23 | 1987-09-26 | 34 |
| Venus | 1987-09-26 | 1987-12-31 | 96 |
| Sun | 1987-12-31 | 1988-01-29 | 29 |
| **Mars** | 1988-01-29 | 1989-03-09 | 405 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 1988-01-29 | 1988-02-22 | 24 |
| Rahu | 1988-02-22 | 1988-04-22 | 61 |
| Jupiter | 1988-04-22 | 1988-06-15 | 54 |
| Saturn | 1988-06-15 | 1988-08-18 | 64 |
| Mercury | 1988-08-18 | 1988-10-15 | 57 |
| Ketu | 1988-10-15 | 1988-11-07 | 24 |
| Venus | 1988-11-07 | 1989-01-14 | 67 |
| Sun | 1989-01-14 | 1989-02-03 | 20 |
| Moon | 1989-02-03 | 1989-03-09 | 34 |
| **Rahu** | 1989-03-09 | 1992-01-14 | 1041 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 1989-03-09 | 1989-08-12 | 156 |
| Jupiter | 1989-08-12 | 1989-12-29 | 139 |
| Saturn | 1989-12-29 | 1990-06-12 | 165 |
| Mercury | 1990-06-12 | 1990-11-06 | 147 |
| Ketu | 1990-11-06 | 1991-01-06 | 61 |
| Venus | 1991-01-06 | 1991-06-28 | 173 |
| Sun | 1991-06-28 | 1991-08-19 | 52 |
| Moon | 1991-08-19 | 1991-11-14 | 87 |
| Mars | 1991-11-14 | 1992-01-14 | 61 |
| **Jupiter** | 1992-01-14 | 1994-07-27 | 925 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 1992-01-14 | 1992-05-16 | 123 |
| Saturn | 1992-05-16 | 1992-10-10 | 147 |
| Mercury | 1992-10-10 | 1993-02-18 | 131 |
| Ketu | 1993-02-18 | 1993-04-13 | 54 |
| Venus | 1993-04-13 | 1993-09-14 | 154 |
| Sun | 1993-09-14 | 1993-10-30 | 46 |
| Moon | 1993-10-30 | 1994-01-15 | 77 |
| Mars | 1994-01-15 | 1994-03-10 | 54 |
| Rahu | 1994-03-10 | 1994-07-27 | 139 |

#### Mahadasha: Mercury (1994-07-27 - 2011-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Mercury** | 1994-07-27 | 1996-12-23 | 880 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 1994-07-27 | 1994-11-29 | 125 |
| Ketu | 1994-11-29 | 1995-01-19 | 51 |
| Venus | 1995-01-19 | 1995-06-15 | 147 |
| Sun | 1995-06-15 | 1995-07-29 | 44 |
| Moon | 1995-07-29 | 1995-10-10 | 73 |
| Mars | 1995-10-10 | 1995-11-30 | 51 |
| Rahu | 1995-11-30 | 1996-04-10 | 132 |
| Jupiter | 1996-04-10 | 1996-08-05 | 117 |
| Saturn | 1996-08-05 | 1996-12-23 | 139 |
| **Ketu** | 1996-12-23 | 1997-12-20 | 362 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 1996-12-23 | 1997-01-13 | 21 |
| Venus | 1997-01-13 | 1997-03-14 | 60 |
| Sun | 1997-03-14 | 1997-04-01 | 18 |
| Moon | 1997-04-01 | 1997-05-01 | 30 |
| Mars | 1997-05-01 | 1997-05-23 | 21 |
| Rahu | 1997-05-23 | 1997-07-16 | 54 |
| Jupiter | 1997-07-16 | 1997-09-02 | 48 |
| Saturn | 1997-09-02 | 1997-10-30 | 57 |
| Mercury | 1997-10-30 | 1997-12-20 | 51 |
| **Venus** | 1997-12-20 | 2000-10-20 | 1035 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 1997-12-20 | 1998-06-10 | 172 |
| Sun | 1998-06-10 | 1998-08-01 | 52 |
| Moon | 1998-08-01 | 1998-10-26 | 86 |
| Mars | 1998-10-26 | 1998-12-26 | 60 |
| Rahu | 1998-12-26 | 1999-05-30 | 155 |
| Jupiter | 1999-05-30 | 1999-10-15 | 138 |
| Saturn | 1999-10-15 | 2000-03-27 | 164 |
| Mercury | 2000-03-27 | 2000-08-20 | 147 |
| Ketu | 2000-08-20 | 2000-10-20 | 60 |
| **Sun** | 2000-10-20 | 2001-08-26 | 310 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2000-10-20 | 2000-11-04 | 16 |
| Moon | 2000-11-04 | 2000-11-30 | 26 |
| Mars | 2000-11-30 | 2000-12-18 | 18 |
| Rahu | 2000-12-18 | 2001-02-03 | 47 |
| Jupiter | 2001-02-03 | 2001-03-16 | 41 |
| Saturn | 2001-03-16 | 2001-05-04 | 49 |
| Mercury | 2001-05-04 | 2001-06-17 | 44 |
| Ketu | 2001-06-17 | 2001-07-05 | 18 |
| Venus | 2001-07-05 | 2001-08-26 | 52 |
| **Moon** | 2001-08-26 | 2003-01-26 | 517 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2001-08-26 | 2001-10-08 | 43 |
| Mars | 2001-10-08 | 2001-11-07 | 30 |
| Rahu | 2001-11-07 | 2002-01-24 | 78 |
| Jupiter | 2002-01-24 | 2002-04-03 | 69 |
| Saturn | 2002-04-03 | 2002-06-24 | 82 |
| Mercury | 2002-06-24 | 2002-09-05 | 73 |
| Ketu | 2002-09-05 | 2002-10-05 | 30 |
| Venus | 2002-10-05 | 2002-12-31 | 86 |
| Sun | 2002-12-31 | 2003-01-26 | 26 |
| **Mars** | 2003-01-26 | 2004-01-23 | 362 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2003-01-26 | 2003-02-16 | 21 |
| Rahu | 2003-02-16 | 2003-04-11 | 54 |
| Jupiter | 2003-04-11 | 2003-05-29 | 48 |
| Saturn | 2003-05-29 | 2003-07-26 | 57 |
| Mercury | 2003-07-26 | 2003-09-15 | 51 |
| Ketu | 2003-09-15 | 2003-10-06 | 21 |
| Venus | 2003-10-06 | 2003-12-05 | 60 |
| Sun | 2003-12-05 | 2003-12-24 | 18 |
| Moon | 2003-12-24 | 2004-01-23 | 30 |
| **Rahu** | 2004-01-23 | 2006-08-11 | 931 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2004-01-23 | 2004-06-10 | 140 |
| Jupiter | 2004-06-10 | 2004-10-13 | 124 |
| Saturn | 2004-10-13 | 2005-03-09 | 147 |
| Mercury | 2005-03-09 | 2005-07-19 | 132 |
| Ketu | 2005-07-19 | 2005-09-11 | 54 |
| Venus | 2005-09-11 | 2006-02-14 | 155 |
| Sun | 2006-02-14 | 2006-04-01 | 47 |
| Moon | 2006-04-01 | 2006-06-18 | 78 |
| Mars | 2006-06-18 | 2006-08-11 | 54 |
| **Jupiter** | 2006-08-11 | 2008-11-16 | 828 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2006-08-11 | 2006-11-30 | 110 |
| Saturn | 2006-11-30 | 2007-04-10 | 131 |
| Mercury | 2007-04-10 | 2007-08-05 | 117 |
| Ketu | 2007-08-05 | 2007-09-22 | 48 |
| Venus | 2007-09-22 | 2008-02-07 | 138 |
| Sun | 2008-02-07 | 2008-03-20 | 41 |
| Moon | 2008-03-20 | 2008-05-28 | 69 |
| Mars | 2008-05-28 | 2008-07-15 | 48 |
| Rahu | 2008-07-15 | 2008-11-16 | 124 |
| **Saturn** | 2008-11-16 | 2011-07-27 | 983 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2008-11-16 | 2009-04-21 | 156 |
| Mercury | 2009-04-21 | 2009-09-07 | 139 |
| Ketu | 2009-09-07 | 2009-11-03 | 57 |
| Venus | 2009-11-03 | 2010-04-16 | 164 |
| Sun | 2010-04-16 | 2010-06-04 | 49 |
| Moon | 2010-06-04 | 2010-08-25 | 82 |
| Mars | 2010-08-25 | 2010-10-22 | 57 |
| Rahu | 2010-10-22 | 2011-03-18 | 147 |
| Jupiter | 2011-03-18 | 2011-07-27 | 131 |

#### Mahadasha: Ketu (2011-07-27 - 2018-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Ketu** | 2011-07-27 | 2011-12-23 | 149 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 2011-07-27 | 2011-08-05 | 9 |
| Venus | 2011-08-05 | 2011-08-30 | 25 |
| Sun | 2011-08-30 | 2011-09-06 | 7 |
| Moon | 2011-09-06 | 2011-09-19 | 12 |
| Mars | 2011-09-19 | 2011-09-27 | 9 |
| Rahu | 2011-09-27 | 2011-10-20 | 22 |
| Jupiter | 2011-10-20 | 2011-11-09 | 20 |
| Saturn | 2011-11-09 | 2011-12-02 | 24 |
| Mercury | 2011-12-02 | 2011-12-23 | 21 |
| **Venus** | 2011-12-23 | 2013-02-21 | 426 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 2011-12-23 | 2012-03-03 | 71 |
| Sun | 2012-03-03 | 2012-03-25 | 21 |
| Moon | 2012-03-25 | 2012-04-29 | 36 |
| Mars | 2012-04-29 | 2012-05-24 | 25 |
| Rahu | 2012-05-24 | 2012-07-27 | 64 |
| Jupiter | 2012-07-27 | 2012-09-22 | 57 |
| Saturn | 2012-09-22 | 2012-11-28 | 67 |
| Mercury | 2012-11-28 | 2013-01-28 | 60 |
| Ketu | 2013-01-28 | 2013-02-21 | 25 |
| **Sun** | 2013-02-21 | 2013-06-29 | 128 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2013-02-21 | 2013-02-28 | 6 |
| Moon | 2013-02-28 | 2013-03-10 | 11 |
| Mars | 2013-03-10 | 2013-03-18 | 7 |
| Rahu | 2013-03-18 | 2013-04-06 | 19 |
| Jupiter | 2013-04-06 | 2013-04-23 | 17 |
| Saturn | 2013-04-23 | 2013-05-13 | 20 |
| Mercury | 2013-05-13 | 2013-05-31 | 18 |
| Ketu | 2013-05-31 | 2013-06-08 | 7 |
| Venus | 2013-06-08 | 2013-06-29 | 21 |
| **Moon** | 2013-06-29 | 2014-01-28 | 213 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2013-06-29 | 2013-07-17 | 18 |
| Mars | 2013-07-17 | 2013-07-29 | 12 |
| Rahu | 2013-07-29 | 2013-08-30 | 32 |
| Jupiter | 2013-08-30 | 2013-09-28 | 28 |
| Saturn | 2013-09-28 | 2013-11-01 | 34 |
| Mercury | 2013-11-01 | 2013-12-01 | 30 |
| Ketu | 2013-12-01 | 2013-12-13 | 12 |
| Venus | 2013-12-13 | 2014-01-18 | 36 |
| Sun | 2014-01-18 | 2014-01-28 | 11 |
| **Mars** | 2014-01-28 | 2014-06-26 | 149 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2014-01-28 | 2014-02-06 | 9 |
| Rahu | 2014-02-06 | 2014-02-28 | 22 |
| Jupiter | 2014-02-28 | 2014-03-20 | 20 |
| Saturn | 2014-03-20 | 2014-04-13 | 24 |
| Mercury | 2014-04-13 | 2014-05-04 | 21 |
| Ketu | 2014-05-04 | 2014-05-13 | 9 |
| Venus | 2014-05-13 | 2014-06-07 | 25 |
| Sun | 2014-06-07 | 2014-06-14 | 7 |
| Moon | 2014-06-14 | 2014-06-26 | 12 |
| **Rahu** | 2014-06-26 | 2015-07-15 | 384 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2014-06-26 | 2014-08-23 | 58 |
| Jupiter | 2014-08-23 | 2014-10-13 | 51 |
| Saturn | 2014-10-13 | 2014-12-13 | 61 |
| Mercury | 2014-12-13 | 2015-02-05 | 54 |
| Ketu | 2015-02-05 | 2015-02-28 | 22 |
| Venus | 2015-02-28 | 2015-05-02 | 64 |
| Sun | 2015-05-02 | 2015-05-22 | 19 |
| Moon | 2015-05-22 | 2015-06-23 | 32 |
| Mars | 2015-06-23 | 2015-07-15 | 22 |
| **Jupiter** | 2015-07-15 | 2016-06-20 | 341 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2015-07-15 | 2015-08-29 | 45 |
| Saturn | 2015-08-29 | 2015-10-22 | 54 |
| Mercury | 2015-10-22 | 2015-12-10 | 48 |
| Ketu | 2015-12-10 | 2015-12-30 | 20 |
| Venus | 2015-12-30 | 2016-02-24 | 57 |
| Sun | 2016-02-24 | 2016-03-12 | 17 |
| Moon | 2016-03-12 | 2016-04-10 | 28 |
| Mars | 2016-04-10 | 2016-04-30 | 20 |
| Rahu | 2016-04-30 | 2016-06-20 | 51 |
| **Saturn** | 2016-06-20 | 2017-07-30 | 405 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2016-06-20 | 2016-08-23 | 64 |
| Mercury | 2016-08-23 | 2016-10-19 | 57 |
| Ketu | 2016-10-19 | 2016-11-12 | 24 |
| Venus | 2016-11-12 | 2017-01-18 | 67 |
| Sun | 2017-01-18 | 2017-02-08 | 20 |
| Moon | 2017-02-08 | 2017-03-13 | 34 |
| Mars | 2017-03-13 | 2017-04-06 | 24 |
| Rahu | 2017-04-06 | 2017-06-06 | 61 |
| Jupiter | 2017-06-06 | 2017-07-30 | 54 |
| **Mercury** | 2017-07-30 | 2018-07-27 | 362 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 2017-07-30 | 2017-09-19 | 51 |
| Ketu | 2017-09-19 | 2017-10-10 | 21 |
| Venus | 2017-10-10 | 2017-12-09 | 60 |
| Sun | 2017-12-09 | 2017-12-28 | 18 |
| Moon | 2017-12-28 | 2018-01-27 | 30 |
| Mars | 2018-01-27 | 2018-02-17 | 21 |
| Rahu | 2018-02-17 | 2018-04-12 | 54 |
| Jupiter | 2018-04-12 | 2018-05-30 | 48 |
| Saturn | 2018-05-30 | 2018-07-27 | 57 |

#### Mahadasha: Venus (2018-07-27 - 2038-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Venus** | 2018-07-27 | 2021-11-25 | 1217 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 2018-07-27 | 2019-02-15 | 203 |
| Sun | 2019-02-15 | 2019-04-17 | 61 |
| Moon | 2019-04-17 | 2019-07-27 | 101 |
| Mars | 2019-07-27 | 2019-10-06 | 71 |
| Rahu | 2019-10-06 | 2020-04-06 | 183 |
| Jupiter | 2020-04-06 | 2020-09-15 | 162 |
| Saturn | 2020-09-15 | 2021-03-27 | 193 |
| Mercury | 2021-03-27 | 2021-09-15 | 172 |
| Ketu | 2021-09-15 | 2021-11-25 | 71 |
| **Sun** | 2021-11-25 | 2022-11-26 | 365 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2021-11-25 | 2021-12-14 | 18 |
| Moon | 2021-12-14 | 2022-01-13 | 30 |
| Mars | 2022-01-13 | 2022-02-03 | 21 |
| Rahu | 2022-02-03 | 2022-03-30 | 55 |
| Jupiter | 2022-03-30 | 2022-05-18 | 49 |
| Saturn | 2022-05-18 | 2022-07-15 | 58 |
| Mercury | 2022-07-15 | 2022-09-04 | 52 |
| Ketu | 2022-09-04 | 2022-09-26 | 21 |
| Venus | 2022-09-26 | 2022-11-26 | 61 |
| **Moon** | 2022-11-26 | 2024-07-26 | 609 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2022-11-26 | 2023-01-15 | 51 |
| Mars | 2023-01-15 | 2023-02-20 | 36 |
| Rahu | 2023-02-20 | 2023-05-22 | 91 |
| Jupiter | 2023-05-22 | 2023-08-11 | 81 |
| Saturn | 2023-08-11 | 2023-11-16 | 96 |
| Mercury | 2023-11-16 | 2024-02-10 | 86 |
| Ketu | 2024-02-10 | 2024-03-16 | 36 |
| Venus | 2024-03-16 | 2024-06-26 | 101 |
| Sun | 2024-06-26 | 2024-07-26 | 30 |
| **Mars** | 2024-07-26 | 2025-09-25 | 426 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2024-07-26 | 2024-08-20 | 25 |
| Rahu | 2024-08-20 | 2024-10-23 | 64 |
| Jupiter | 2024-10-23 | 2024-12-19 | 57 |
| Saturn | 2024-12-19 | 2025-02-24 | 67 |
| Mercury | 2025-02-24 | 2025-04-26 | 60 |
| Ketu | 2025-04-26 | 2025-05-21 | 25 |
| Venus | 2025-05-21 | 2025-07-31 | 71 |
| Sun | 2025-07-31 | 2025-08-21 | 21 |
| Moon | 2025-08-21 | 2025-09-25 | 36 |
| **Rahu** | 2025-09-25 | 2028-09-25 | 1096 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2025-09-25 | 2026-03-09 | 164 |
| Jupiter | 2026-03-09 | 2026-08-02 | 146 |
| Saturn | 2026-08-02 | 2027-01-22 | 173 |
| Mercury | 2027-01-22 | 2027-06-27 | 155 |
| Ketu | 2027-06-27 | 2027-08-30 | 64 |
| Venus | 2027-08-30 | 2028-02-28 | 183 |
| Sun | 2028-02-28 | 2028-04-23 | 55 |
| Moon | 2028-04-23 | 2028-07-23 | 91 |
| Mars | 2028-07-23 | 2028-09-25 | 64 |
| **Jupiter** | 2028-09-25 | 2031-05-27 | 974 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2028-09-25 | 2029-02-02 | 130 |
| Saturn | 2029-02-02 | 2029-07-06 | 154 |
| Mercury | 2029-07-06 | 2029-11-21 | 138 |
| Ketu | 2029-11-21 | 2030-01-17 | 57 |
| Venus | 2030-01-17 | 2030-06-28 | 162 |
| Sun | 2030-06-28 | 2030-08-16 | 49 |
| Moon | 2030-08-16 | 2030-11-05 | 81 |
| Mars | 2030-11-05 | 2031-01-01 | 57 |
| Rahu | 2031-01-01 | 2031-05-27 | 146 |
| **Saturn** | 2031-05-27 | 2034-07-27 | 1157 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2031-05-27 | 2031-11-26 | 183 |
| Mercury | 2031-11-26 | 2032-05-08 | 164 |
| Ketu | 2032-05-08 | 2032-07-15 | 67 |
| Venus | 2032-07-15 | 2033-01-23 | 193 |
| Sun | 2033-01-23 | 2033-03-22 | 58 |
| Moon | 2033-03-22 | 2033-06-27 | 96 |
| Mars | 2033-06-27 | 2033-09-02 | 67 |
| Rahu | 2033-09-02 | 2034-02-23 | 173 |
| Jupiter | 2034-02-23 | 2034-07-27 | 154 |
| **Mercury** | 2034-07-27 | 2037-05-27 | 1035 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 2034-07-27 | 2034-12-20 | 147 |
| Ketu | 2034-12-20 | 2035-02-19 | 60 |
| Venus | 2035-02-19 | 2035-08-10 | 172 |
| Sun | 2035-08-10 | 2035-10-01 | 52 |
| Moon | 2035-10-01 | 2035-12-26 | 86 |
| Mars | 2035-12-26 | 2036-02-25 | 60 |
| Rahu | 2036-02-25 | 2036-07-29 | 155 |
| Jupiter | 2036-07-29 | 2036-12-14 | 138 |
| Saturn | 2036-12-14 | 2037-05-27 | 164 |
| **Ketu** | 2037-05-27 | 2038-07-27 | 426 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 2037-05-27 | 2037-06-20 | 25 |
| Venus | 2037-06-20 | 2037-08-30 | 71 |
| Sun | 2037-08-30 | 2037-09-21 | 21 |
| Moon | 2037-09-21 | 2037-10-26 | 36 |
| Mars | 2037-10-26 | 2037-11-20 | 25 |
| Rahu | 2037-11-20 | 2038-01-23 | 64 |
| Jupiter | 2038-01-23 | 2038-03-21 | 57 |
| Saturn | 2038-03-21 | 2038-05-27 | 67 |
| Mercury | 2038-05-27 | 2038-07-27 | 60 |

#### Mahadasha: Sun (2038-07-27 - 2044-07-26)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Sun** | 2038-07-27 | 2038-11-13 | 110 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2038-07-27 | 2038-08-01 | 5 |
| Moon | 2038-08-01 | 2038-08-10 | 9 |
| Mars | 2038-08-10 | 2038-08-17 | 6 |
| Rahu | 2038-08-17 | 2038-09-02 | 16 |
| Jupiter | 2038-09-02 | 2038-09-17 | 15 |
| Saturn | 2038-09-17 | 2038-10-04 | 17 |
| Mercury | 2038-10-04 | 2038-10-20 | 16 |
| Ketu | 2038-10-20 | 2038-10-26 | 6 |
| Venus | 2038-10-26 | 2038-11-13 | 18 |
| **Moon** | 2038-11-13 | 2039-05-15 | 183 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2038-11-13 | 2038-11-28 | 15 |
| Mars | 2038-11-28 | 2038-12-09 | 11 |
| Rahu | 2038-12-09 | 2039-01-06 | 27 |
| Jupiter | 2039-01-06 | 2039-01-30 | 24 |
| Saturn | 2039-01-30 | 2039-02-28 | 29 |
| Mercury | 2039-02-28 | 2039-03-26 | 26 |
| Ketu | 2039-03-26 | 2039-04-05 | 11 |
| Venus | 2039-04-05 | 2039-05-06 | 30 |
| Sun | 2039-05-06 | 2039-05-15 | 9 |
| **Mars** | 2039-05-15 | 2039-09-20 | 128 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2039-05-15 | 2039-05-22 | 7 |
| Rahu | 2039-05-22 | 2039-06-11 | 19 |
| Jupiter | 2039-06-11 | 2039-06-28 | 17 |
| Saturn | 2039-06-28 | 2039-07-18 | 20 |
| Mercury | 2039-07-18 | 2039-08-05 | 18 |
| Ketu | 2039-08-05 | 2039-08-12 | 7 |
| Venus | 2039-08-12 | 2039-09-03 | 21 |
| Sun | 2039-09-03 | 2039-09-09 | 6 |
| Moon | 2039-09-09 | 2039-09-20 | 11 |
| **Rahu** | 2039-09-20 | 2040-08-13 | 329 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2039-09-20 | 2039-11-08 | 49 |
| Jupiter | 2039-11-08 | 2039-12-22 | 44 |
| Saturn | 2039-12-22 | 2040-02-12 | 52 |
| Mercury | 2040-02-12 | 2040-03-29 | 47 |
| Ketu | 2040-03-29 | 2040-04-18 | 19 |
| Venus | 2040-04-18 | 2040-06-11 | 55 |
| Sun | 2040-06-11 | 2040-06-28 | 16 |
| Moon | 2040-06-28 | 2040-07-25 | 27 |
| Mars | 2040-07-25 | 2040-08-13 | 19 |
| **Jupiter** | 2040-08-13 | 2041-06-02 | 292 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2040-08-13 | 2040-09-21 | 39 |
| Saturn | 2040-09-21 | 2040-11-07 | 46 |
| Mercury | 2040-11-07 | 2040-12-18 | 41 |
| Ketu | 2040-12-18 | 2041-01-04 | 17 |
| Venus | 2041-01-04 | 2041-02-22 | 49 |
| Sun | 2041-02-22 | 2041-03-08 | 15 |
| Moon | 2041-03-08 | 2041-04-02 | 24 |
| Mars | 2041-04-02 | 2041-04-19 | 17 |
| Rahu | 2041-04-19 | 2041-06-02 | 44 |
| **Saturn** | 2041-06-02 | 2042-05-15 | 347 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2041-06-02 | 2041-07-27 | 55 |
| Mercury | 2041-07-27 | 2041-09-14 | 49 |
| Ketu | 2041-09-14 | 2041-10-04 | 20 |
| Venus | 2041-10-04 | 2041-12-01 | 58 |
| Sun | 2041-12-01 | 2041-12-18 | 17 |
| Moon | 2041-12-18 | 2042-01-16 | 29 |
| Mars | 2042-01-16 | 2042-02-05 | 20 |
| Rahu | 2042-02-05 | 2042-03-29 | 52 |
| Jupiter | 2042-03-29 | 2042-05-15 | 46 |
| **Mercury** | 2042-05-15 | 2043-03-21 | 310 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 2042-05-15 | 2042-06-28 | 44 |
| Ketu | 2042-06-28 | 2042-07-16 | 18 |
| Venus | 2042-07-16 | 2042-09-05 | 52 |
| Sun | 2042-09-05 | 2042-09-21 | 16 |
| Moon | 2042-09-21 | 2042-10-17 | 26 |
| Mars | 2042-10-17 | 2042-11-04 | 18 |
| Rahu | 2042-11-04 | 2042-12-21 | 47 |
| Jupiter | 2042-12-21 | 2043-01-31 | 41 |
| Saturn | 2043-01-31 | 2043-03-21 | 49 |
| **Ketu** | 2043-03-21 | 2043-07-27 | 128 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 2043-03-21 | 2043-03-29 | 7 |
| Venus | 2043-03-29 | 2043-04-19 | 21 |
| Sun | 2043-04-19 | 2043-04-25 | 6 |
| Moon | 2043-04-25 | 2043-05-06 | 11 |
| Mars | 2043-05-06 | 2043-05-13 | 7 |
| Rahu | 2043-05-13 | 2043-06-02 | 19 |
| Jupiter | 2043-06-02 | 2043-06-19 | 17 |
| Saturn | 2043-06-19 | 2043-07-09 | 20 |
| Mercury | 2043-07-09 | 2043-07-27 | 18 |
| **Venus** | 2043-07-27 | 2044-07-26 | 365 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 2043-07-27 | 2043-09-26 | 61 |
| Sun | 2043-09-26 | 2043-10-14 | 18 |
| Moon | 2043-10-14 | 2043-11-13 | 30 |
| Mars | 2043-11-13 | 2043-12-05 | 21 |
| Rahu | 2043-12-05 | 2044-01-29 | 55 |
| Jupiter | 2044-01-29 | 2044-03-17 | 49 |
| Saturn | 2044-03-17 | 2044-05-14 | 58 |
| Mercury | 2044-05-14 | 2044-07-05 | 52 |
| Ketu | 2044-07-05 | 2044-07-26 | 21 |

#### Mahadasha: Moon (2044-07-26 - 2054-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Moon** | 2044-07-26 | 2045-05-27 | 304 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2044-07-26 | 2044-08-21 | 25 |
| Mars | 2044-08-21 | 2044-09-07 | 18 |
| Rahu | 2044-09-07 | 2044-10-23 | 46 |
| Jupiter | 2044-10-23 | 2044-12-02 | 41 |
| Saturn | 2044-12-02 | 2045-01-20 | 48 |
| Mercury | 2045-01-20 | 2045-03-04 | 43 |
| Ketu | 2045-03-04 | 2045-03-22 | 18 |
| Venus | 2045-03-22 | 2045-05-11 | 51 |
| Sun | 2045-05-11 | 2045-05-27 | 15 |
| **Mars** | 2045-05-27 | 2045-12-26 | 213 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2045-05-27 | 2045-06-08 | 12 |
| Rahu | 2045-06-08 | 2045-07-10 | 32 |
| Jupiter | 2045-07-10 | 2045-08-07 | 28 |
| Saturn | 2045-08-07 | 2045-09-10 | 34 |
| Mercury | 2045-09-10 | 2045-10-10 | 30 |
| Ketu | 2045-10-10 | 2045-10-23 | 12 |
| Venus | 2045-10-23 | 2045-11-27 | 36 |
| Sun | 2045-11-27 | 2045-12-08 | 11 |
| Moon | 2045-12-08 | 2045-12-26 | 18 |
| **Rahu** | 2045-12-26 | 2047-06-26 | 548 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2045-12-26 | 2046-03-18 | 82 |
| Jupiter | 2046-03-18 | 2046-05-30 | 73 |
| Saturn | 2046-05-30 | 2046-08-25 | 87 |
| Mercury | 2046-08-25 | 2046-11-10 | 78 |
| Ketu | 2046-11-10 | 2046-12-12 | 32 |
| Venus | 2046-12-12 | 2047-03-13 | 91 |
| Sun | 2047-03-13 | 2047-04-10 | 27 |
| Moon | 2047-04-10 | 2047-05-25 | 46 |
| Mars | 2047-05-25 | 2047-06-26 | 32 |
| **Jupiter** | 2047-06-26 | 2048-10-25 | 487 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2047-06-26 | 2047-08-30 | 65 |
| Saturn | 2047-08-30 | 2047-11-15 | 77 |
| Mercury | 2047-11-15 | 2048-01-23 | 69 |
| Ketu | 2048-01-23 | 2048-02-21 | 28 |
| Venus | 2048-02-21 | 2048-05-12 | 81 |
| Sun | 2048-05-12 | 2048-06-05 | 24 |
| Moon | 2048-06-05 | 2048-07-16 | 41 |
| Mars | 2048-07-16 | 2048-08-13 | 28 |
| Rahu | 2048-08-13 | 2048-10-25 | 73 |
| **Saturn** | 2048-10-25 | 2050-05-27 | 578 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2048-10-25 | 2049-01-25 | 92 |
| Mercury | 2049-01-25 | 2049-04-17 | 82 |
| Ketu | 2049-04-17 | 2049-05-21 | 34 |
| Venus | 2049-05-21 | 2049-08-25 | 96 |
| Sun | 2049-08-25 | 2049-09-23 | 29 |
| Moon | 2049-09-23 | 2049-11-10 | 48 |
| Mars | 2049-11-10 | 2049-12-14 | 34 |
| Rahu | 2049-12-14 | 2050-03-11 | 87 |
| Jupiter | 2050-03-11 | 2050-05-27 | 77 |
| **Mercury** | 2050-05-27 | 2051-10-26 | 517 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 2050-05-27 | 2050-08-08 | 73 |
| Ketu | 2050-08-08 | 2050-09-07 | 30 |
| Venus | 2050-09-07 | 2050-12-02 | 86 |
| Sun | 2050-12-02 | 2050-12-28 | 26 |
| Moon | 2050-12-28 | 2051-02-09 | 43 |
| Mars | 2051-02-09 | 2051-03-12 | 30 |
| Rahu | 2051-03-12 | 2051-05-28 | 78 |
| Jupiter | 2051-05-28 | 2051-08-05 | 69 |
| Saturn | 2051-08-05 | 2051-10-26 | 82 |
| **Ketu** | 2051-10-26 | 2052-05-26 | 213 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 2051-10-26 | 2051-11-08 | 12 |
| Venus | 2051-11-08 | 2051-12-13 | 36 |
| Sun | 2051-12-13 | 2051-12-24 | 11 |
| Moon | 2051-12-24 | 2052-01-10 | 18 |
| Mars | 2052-01-10 | 2052-01-23 | 12 |
| Rahu | 2052-01-23 | 2052-02-24 | 32 |
| Jupiter | 2052-02-24 | 2052-03-23 | 28 |
| Saturn | 2052-03-23 | 2052-04-26 | 34 |
| Mercury | 2052-04-26 | 2052-05-26 | 30 |
| **Venus** | 2052-05-26 | 2054-01-25 | 609 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 2052-05-26 | 2052-09-05 | 101 |
| Sun | 2052-09-05 | 2052-10-05 | 30 |
| Moon | 2052-10-05 | 2052-11-25 | 51 |
| Mars | 2052-11-25 | 2052-12-30 | 36 |
| Rahu | 2052-12-30 | 2053-04-01 | 91 |
| Jupiter | 2053-04-01 | 2053-06-21 | 81 |
| Saturn | 2053-06-21 | 2053-09-25 | 96 |
| Mercury | 2053-09-25 | 2053-12-20 | 86 |
| Ketu | 2053-12-20 | 2054-01-25 | 36 |
| **Sun** | 2054-01-25 | 2054-07-27 | 183 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2054-01-25 | 2054-02-03 | 9 |
| Moon | 2054-02-03 | 2054-02-18 | 15 |
| Mars | 2054-02-18 | 2054-03-01 | 11 |
| Rahu | 2054-03-01 | 2054-03-28 | 27 |
| Jupiter | 2054-03-28 | 2054-04-22 | 24 |
| Saturn | 2054-04-22 | 2054-05-21 | 29 |
| Mercury | 2054-05-21 | 2054-06-15 | 26 |
| Ketu | 2054-06-15 | 2054-06-26 | 11 |
| Venus | 2054-06-26 | 2054-07-27 | 30 |

#### Mahadasha: Mars (2054-07-27 - 2061-07-26)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Mars** | 2054-07-27 | 2054-12-23 | 149 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2054-07-27 | 2054-08-04 | 9 |
| Rahu | 2054-08-04 | 2054-08-27 | 22 |
| Jupiter | 2054-08-27 | 2054-09-16 | 20 |
| Saturn | 2054-09-16 | 2054-10-09 | 24 |
| Mercury | 2054-10-09 | 2054-10-30 | 21 |
| Ketu | 2054-10-30 | 2054-11-08 | 9 |
| Venus | 2054-11-08 | 2054-12-03 | 25 |
| Sun | 2054-12-03 | 2054-12-10 | 7 |
| Moon | 2054-12-10 | 2054-12-23 | 12 |
| **Rahu** | 2054-12-23 | 2056-01-10 | 384 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2054-12-23 | 2055-02-18 | 58 |
| Jupiter | 2055-02-18 | 2055-04-10 | 51 |
| Saturn | 2055-04-10 | 2055-06-10 | 61 |
| Mercury | 2055-06-10 | 2055-08-03 | 54 |
| Ketu | 2055-08-03 | 2055-08-26 | 22 |
| Venus | 2055-08-26 | 2055-10-29 | 64 |
| Sun | 2055-10-29 | 2055-11-17 | 19 |
| Moon | 2055-11-17 | 2055-12-19 | 32 |
| Mars | 2055-12-19 | 2056-01-10 | 22 |
| **Jupiter** | 2056-01-10 | 2056-12-16 | 341 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2056-01-10 | 2056-02-25 | 45 |
| Saturn | 2056-02-25 | 2056-04-19 | 54 |
| Mercury | 2056-04-19 | 2056-06-06 | 48 |
| Ketu | 2056-06-06 | 2056-06-26 | 20 |
| Venus | 2056-06-26 | 2056-08-22 | 57 |
| Sun | 2056-08-22 | 2056-09-08 | 17 |
| Moon | 2056-09-08 | 2056-10-06 | 28 |
| Mars | 2056-10-06 | 2056-10-26 | 20 |
| Rahu | 2056-10-26 | 2056-12-16 | 51 |
| **Saturn** | 2056-12-16 | 2058-01-25 | 405 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2056-12-16 | 2057-02-18 | 64 |
| Mercury | 2057-02-18 | 2057-04-17 | 57 |
| Ketu | 2057-04-17 | 2057-05-10 | 24 |
| Venus | 2057-05-10 | 2057-07-17 | 67 |
| Sun | 2057-07-17 | 2057-08-06 | 20 |
| Moon | 2057-08-06 | 2057-09-09 | 34 |
| Mars | 2057-09-09 | 2057-10-02 | 24 |
| Rahu | 2057-10-02 | 2057-12-02 | 61 |
| Jupiter | 2057-12-02 | 2058-01-25 | 54 |
| **Mercury** | 2058-01-25 | 2059-01-22 | 362 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 2058-01-25 | 2058-03-17 | 51 |
| Ketu | 2058-03-17 | 2058-04-07 | 21 |
| Venus | 2058-04-07 | 2058-06-07 | 60 |
| Sun | 2058-06-07 | 2058-06-25 | 18 |
| Moon | 2058-06-25 | 2058-07-25 | 30 |
| Mars | 2058-07-25 | 2058-08-15 | 21 |
| Rahu | 2058-08-15 | 2058-10-08 | 54 |
| Jupiter | 2058-10-08 | 2058-11-26 | 48 |
| Saturn | 2058-11-26 | 2059-01-22 | 57 |
| **Ketu** | 2059-01-22 | 2059-06-20 | 149 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 2059-01-22 | 2059-01-31 | 9 |
| Venus | 2059-01-31 | 2059-02-25 | 25 |
| Sun | 2059-02-25 | 2059-03-04 | 7 |
| Moon | 2059-03-04 | 2059-03-17 | 12 |
| Mars | 2059-03-17 | 2059-03-25 | 9 |
| Rahu | 2059-03-25 | 2059-04-17 | 22 |
| Jupiter | 2059-04-17 | 2059-05-07 | 20 |
| Saturn | 2059-05-07 | 2059-05-30 | 24 |
| Mercury | 2059-05-30 | 2059-06-20 | 21 |
| **Venus** | 2059-06-20 | 2060-08-19 | 426 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 2059-06-20 | 2059-08-30 | 71 |
| Sun | 2059-08-30 | 2059-09-21 | 21 |
| Moon | 2059-09-21 | 2059-10-26 | 36 |
| Mars | 2059-10-26 | 2059-11-20 | 25 |
| Rahu | 2059-11-20 | 2060-01-23 | 64 |
| Jupiter | 2060-01-23 | 2060-03-20 | 57 |
| Saturn | 2060-03-20 | 2060-05-26 | 67 |
| Mercury | 2060-05-26 | 2060-07-26 | 60 |
| Ketu | 2060-07-26 | 2060-08-19 | 25 |
| **Sun** | 2060-08-19 | 2060-12-25 | 128 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2060-08-19 | 2060-08-26 | 6 |
| Moon | 2060-08-26 | 2060-09-05 | 11 |
| Mars | 2060-09-05 | 2060-09-13 | 7 |
| Rahu | 2060-09-13 | 2060-10-02 | 19 |
| Jupiter | 2060-10-02 | 2060-10-19 | 17 |
| Saturn | 2060-10-19 | 2060-11-08 | 20 |
| Mercury | 2060-11-08 | 2060-11-26 | 18 |
| Ketu | 2060-11-26 | 2060-12-04 | 7 |
| Venus | 2060-12-04 | 2060-12-25 | 21 |
| **Moon** | 2060-12-25 | 2061-07-26 | 213 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2060-12-25 | 2061-01-12 | 18 |
| Mars | 2061-01-12 | 2061-01-24 | 12 |
| Rahu | 2061-01-24 | 2061-02-25 | 32 |
| Jupiter | 2061-02-25 | 2061-03-26 | 28 |
| Saturn | 2061-03-26 | 2061-04-28 | 34 |
| Mercury | 2061-04-28 | 2061-05-29 | 30 |
| Ketu | 2061-05-29 | 2061-06-10 | 12 |
| Venus | 2061-06-10 | 2061-07-16 | 36 |
| Sun | 2061-07-16 | 2061-07-26 | 11 |

#### Mahadasha: Rahu (2061-07-26 - 2079-07-27)
| Antardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| **Rahu** | 2061-07-26 | 2064-04-07 | 986 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Rahu | 2061-07-26 | 2061-12-21 | 148 |
| Jupiter | 2061-12-21 | 2062-05-02 | 131 |
| Saturn | 2062-05-02 | 2062-10-05 | 156 |
| Mercury | 2062-10-05 | 2063-02-22 | 140 |
| Ketu | 2063-02-22 | 2063-04-20 | 58 |
| Venus | 2063-04-20 | 2063-10-01 | 164 |
| Sun | 2063-10-01 | 2063-11-20 | 49 |
| Moon | 2063-11-20 | 2064-02-10 | 82 |
| Mars | 2064-02-10 | 2064-04-07 | 58 |
| **Jupiter** | 2064-04-07 | 2066-09-01 | 877 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Jupiter | 2064-04-07 | 2064-08-02 | 117 |
| Saturn | 2064-08-02 | 2064-12-19 | 139 |
| Mercury | 2064-12-19 | 2065-04-22 | 124 |
| Ketu | 2065-04-22 | 2065-06-12 | 51 |
| Venus | 2065-06-12 | 2065-11-06 | 146 |
| Sun | 2065-11-06 | 2065-12-19 | 44 |
| Moon | 2065-12-19 | 2066-03-02 | 73 |
| Mars | 2066-03-02 | 2066-04-23 | 51 |
| Rahu | 2066-04-23 | 2066-09-01 | 131 |
| **Saturn** | 2066-09-01 | 2069-07-08 | 1041 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Saturn | 2066-09-01 | 2067-02-13 | 165 |
| Mercury | 2067-02-13 | 2067-07-10 | 147 |
| Ketu | 2067-07-10 | 2067-09-09 | 61 |
| Venus | 2067-09-09 | 2068-02-29 | 173 |
| Sun | 2068-02-29 | 2068-04-22 | 52 |
| Moon | 2068-04-22 | 2068-07-17 | 87 |
| Mars | 2068-07-17 | 2068-09-16 | 61 |
| Rahu | 2068-09-16 | 2069-02-19 | 156 |
| Jupiter | 2069-02-19 | 2069-07-08 | 139 |
| **Mercury** | 2069-07-08 | 2072-01-25 | 931 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mercury | 2069-07-08 | 2069-11-17 | 132 |
| Ketu | 2069-11-17 | 2070-01-10 | 54 |
| Venus | 2070-01-10 | 2070-06-14 | 155 |
| Sun | 2070-06-14 | 2070-07-31 | 47 |
| Moon | 2070-07-31 | 2070-10-17 | 78 |
| Mars | 2070-10-17 | 2070-12-10 | 54 |
| Rahu | 2070-12-10 | 2071-04-29 | 140 |
| Jupiter | 2071-04-29 | 2071-08-31 | 124 |
| Saturn | 2071-08-31 | 2072-01-25 | 147 |
| **Ketu** | 2072-01-25 | 2073-02-12 | 384 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Ketu | 2072-01-25 | 2072-02-17 | 22 |
| Venus | 2072-02-17 | 2072-04-21 | 64 |
| Sun | 2072-04-21 | 2072-05-10 | 19 |
| Moon | 2072-05-10 | 2072-06-11 | 32 |
| Mars | 2072-06-11 | 2072-07-03 | 22 |
| Rahu | 2072-07-03 | 2072-08-30 | 58 |
| Jupiter | 2072-08-30 | 2072-10-20 | 51 |
| Saturn | 2072-10-20 | 2072-12-19 | 61 |
| Mercury | 2072-12-19 | 2073-02-12 | 54 |
| **Venus** | 2073-02-12 | 2076-02-13 | 1096 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Venus | 2073-02-12 | 2073-08-13 | 183 |
| Sun | 2073-08-13 | 2073-10-07 | 55 |
| Moon | 2073-10-07 | 2074-01-07 | 91 |
| Mars | 2074-01-07 | 2074-03-11 | 64 |
| Rahu | 2074-03-11 | 2074-08-23 | 164 |
| Jupiter | 2074-08-23 | 2075-01-16 | 146 |
| Saturn | 2075-01-16 | 2075-07-08 | 173 |
| Mercury | 2075-07-08 | 2075-12-11 | 155 |
| Ketu | 2075-12-11 | 2076-02-13 | 64 |
| **Sun** | 2076-02-13 | 2077-01-06 | 329 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Sun | 2076-02-13 | 2076-02-29 | 16 |
| Moon | 2076-02-29 | 2076-03-27 | 27 |
| Mars | 2076-03-27 | 2076-04-16 | 19 |
| Rahu | 2076-04-16 | 2076-06-04 | 49 |
| Jupiter | 2076-06-04 | 2076-07-18 | 44 |
| Saturn | 2076-07-18 | 2076-09-08 | 52 |
| Mercury | 2076-09-08 | 2076-10-24 | 47 |
| Ketu | 2076-10-24 | 2076-11-12 | 19 |
| Venus | 2076-11-12 | 2077-01-06 | 55 |
| **Moon** | 2077-01-06 | 2078-07-08 | 548 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Moon | 2077-01-06 | 2077-02-21 | 46 |
| Mars | 2077-02-21 | 2077-03-25 | 32 |
| Rahu | 2077-03-25 | 2077-06-15 | 82 |
| Jupiter | 2077-06-15 | 2077-08-27 | 73 |
| Saturn | 2077-08-27 | 2077-11-22 | 87 |
| Mercury | 2077-11-22 | 2078-02-07 | 78 |
| Ketu | 2078-02-07 | 2078-03-11 | 32 |
| Venus | 2078-03-11 | 2078-06-11 | 91 |
| Sun | 2078-06-11 | 2078-07-08 | 27 |
| **Mars** | 2078-07-08 | 2079-07-27 | 384 |
| Pratyantardasha Lord | Start Date | End Date | Duration |
|---|---|---|---|
| Mars | 2078-07-08 | 2078-07-30 | 22 |
| Rahu | 2078-07-30 | 2078-09-26 | 58 |
| Jupiter | 2078-09-26 | 2078-11-16 | 51 |
| Saturn | 2078-11-16 | 2079-01-16 | 61 |
| Mercury | 2079-01-16 | 2079-03-11 | 54 |
| Ketu | 2079-03-11 | 2079-04-03 | 22 |
| Venus | 2079-04-03 | 2079-06-05 | 64 |
| Sun | 2079-06-05 | 2079-06-25 | 19 |
| Moon | 2079-06-25 | 2079-07-27 | 32 |

```
