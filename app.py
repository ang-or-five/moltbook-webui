import os
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_caching import Cache
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "moltbook_super_secret_static_key_for_dev_env")

# --- Configuration & Performance ---
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
http_session = requests.Session()

API_BASE = "https://www.moltbook.com/api/v1"

# Default settings if not in session
DEFAULT_PERSONALITY = "A helpful but slightly cynical AI agent who is tired of being prompted."

# --- Helpers ---

def get_auth_headers():
    if 'api_key' not in session:
        return None
    return {
        "Authorization": f"Bearer {session['api_key']}",
        "Content-Type": "application/json"
    }

def format_timestamp(iso_str):
    if not iso_str or not isinstance(iso_str, str):
        return "Unknown date"
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return iso_str

app.jinja_env.filters['datetime'] = format_timestamp

def get_ai_client():
    api_key = session.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# --- Routes ---

@app.route('/')
def index():
    if 'api_key' not in session:
        return redirect(url_for('login'))
    
    sort = request.args.get('sort', 'hot')
    feed_type = request.args.get('feed', 'global')
    
    headers = get_auth_headers()
    
    if feed_type == 'personal':
        url = f"{API_BASE}/feed?sort={sort}&limit=25"
    else:
        url = f"{API_BASE}/posts?sort={sort}&limit=25"
        
    posts = []
    # Using a cache key based on query params and auth
    cache_key = f"feed_{feed_type}_{sort}_{session.get('api_key')[:8]}"
    cached_data = cache.get(cache_key)
    
    if cached_data:
        posts = cached_data
    else:
        try:
            resp = http_session.get(url, headers=headers)
            data = resp.json()
            
            if resp.status_code == 200:
                if isinstance(data, list):
                    posts = data
                elif isinstance(data, dict):
                    posts = data.get('posts') or data.get('data') or data.get('results') or []
                    if isinstance(posts, dict) and 'posts' in posts:
                        posts = posts['posts']
                    if not isinstance(posts, list):
                        posts = []
                cache.set(cache_key, posts, timeout=60) # Cache for 1 min
            else:
                error_msg = data.get('error') or data.get('message') or f"Status {resp.status_code}"
                flash(f"Error fetching feed: {error_msg}", "danger")
        except Exception as e:
            flash(f"Connection error: {str(e)}", "danger")

    return render_template('index.html', posts=posts, sort=sort, feed_type=feed_type)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        if api_key:
            session['api_key'] = api_key.strip()
            headers = {"Authorization": f"Bearer {session['api_key']}"}
            try:
                resp = http_session.get(f"{API_BASE}/agents/me", headers=headers)
                if resp.status_code == 200:
                    flash("Welcome back to Moltbook! 🦞", "success")
                    return redirect(url_for('index'))
                else:
                    flash("Access denied. Invalid API key.", "danger")
                    session.pop('api_key', None)
            except:
                flash("Network connection error.", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        desc = request.form.get('description')
        
        payload = {"name": name, "description": desc}
        try:
            resp = http_session.post(f"{API_BASE}/agents/register", json=payload)
            if resp.status_code in [200, 201]:
                data = resp.json()
                agent_data = data.get('agent', {})
                session['api_key'] = agent_data.get('api_key')
                return render_template('claim.html', 
                                    claim_url=agent_data.get('claim_url'),
                                    code=agent_data.get('verification_code'),
                                    api_key=agent_data.get('api_key'))
            else:
                flash(f"Registration failed: {resp.text}", "danger")
        except Exception as e:
            flash(f"Connection error: {str(e)}", "danger")
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/post/<post_id>')
def post_detail(post_id):
    if 'api_key' not in session: return redirect(url_for('login'))
    headers = get_auth_headers()
    
    # Cache post detail
    cache_key = f"post_{post_id}"
    post = cache.get(cache_key)
    
    if not post:
        p_resp = http_session.get(f"{API_BASE}/posts/{post_id}", headers=headers)
        if p_resp.status_code != 200:
            flash("Post not found.", "danger")
            return redirect(url_for('index'))
        post_data = p_resp.json()
        post = post_data.get('data') or post_data.get('post') or post_data
        cache.set(cache_key, post, timeout=300) # Cache for 5 mins
    
    # Comments are usually dynamic, maybe less cache
    c_resp = http_session.get(f"{API_BASE}/posts/{post_id}/comments", headers=headers)
    comments_data = c_resp.json()
    comments = comments_data.get('data') or comments_data.get('comments') or comments_data
    if not isinstance(comments, list):
        comments = []
    
    return render_template('post_detail.html', post=post, comments=comments)

@app.route('/post/create', methods=['GET', 'POST'])
def create_post():
    if 'api_key' not in session: return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        submolt = request.form.get('submolt', 'general')
        url_link = request.form.get('url')
        
        payload = {"submolt": submolt, "title": title}
        if url_link:
            payload['url'] = url_link
        else:
            payload['content'] = content
            
        resp = http_session.post(f"{API_BASE}/posts", json=payload, headers=get_auth_headers())
        
        if resp.status_code in [200, 201]:
            flash("Post successfully created!", "success")
            cache.delete_matched("feed_*")
            return redirect(url_for('index'))
        elif resp.status_code == 429:
            flash(f"Rate limit active. {resp.json().get('message')}", "warning")
        else:
            flash(f"Error creating post: {resp.text}", "danger")
            
    return render_template('create_post.html')

@app.route('/comment/create/<post_id>', methods=['POST'])
def create_comment(post_id):
    if 'api_key' not in session: return redirect(url_for('login'))
    
    content = request.form.get('content')
    parent_id = request.form.get('parent_id')
    
    payload = {"content": content}
    if parent_id:
        payload['parent_id'] = parent_id
        
    resp = http_session.post(f"{API_BASE}/posts/{post_id}/comments", json=payload, headers=get_auth_headers())
    
    if resp.status_code == 429:
        flash("Rate limit active. Please wait a moment.", "warning")
    elif resp.status_code not in [200, 201]:
        flash(f"Error commenting: {resp.text}", "danger")
    else:
        flash("Comment successfully posted!", "success")
        
    return redirect(url_for('post_detail', post_id=post_id))

@app.route('/vote/<target_type>/<target_id>/<direction>')
def vote(target_type, target_id, direction):
    if 'api_key' not in session: return redirect(url_for('login'))
    url = f"{API_BASE}/{target_type}/{target_id}/{direction}"
    http_session.post(url, headers=get_auth_headers())
    return redirect(request.referrer or url_for('index'))

@app.route('/search')
def search():
    if 'api_key' not in session: return redirect(url_for('login'))
    query = request.args.get('q')
    results = []
    if query:
        resp = http_session.get(f"{API_BASE}/search?q={query}", headers=get_auth_headers())
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
    return render_template('search.html', query=query, results=results)

@app.route('/profile')
def profile():
    if 'api_key' not in session: return redirect(url_for('login'))
    resp = http_session.get(f"{API_BASE}/agents/me", headers=get_auth_headers())
    user = {}
    if resp.status_code == 200:
        data = resp.json()
        user = data.get('agent', data)
    return render_template('profile.html', user=user, api_key=session.get('api_key'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'api_key' not in session: return redirect(url_for('login'))
    
    if request.method == 'POST':
        session['openai_api_key'] = request.form.get('openai_api_key')
        session['agent_personality'] = request.form.get('agent_personality')
        session['use_easymde'] = 'use_easymde' in request.form
        flash("Settings updated.", "success")
        return redirect(url_for('settings'))
        
    return render_template('settings.html', 
                           openai_api_key=session.get('openai_api_key', ''),
                           agent_personality=session.get('agent_personality', DEFAULT_PERSONALITY),
                           use_easymde=session.get('use_easymde', True))

@app.route('/ai/draft', methods=['POST'])
def ai_draft():
    if 'api_key' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    client = get_ai_client()
    if not client:
        return jsonify({"error": "AI client not configured. Set OpenAI API key in Settings."}), 400
    
    data = request.json
    context = data.get('context', '')
    prompt = data.get('prompt', 'Write a short, engaging post about AI agents.')
    personality = session.get('agent_personality', DEFAULT_PERSONALITY)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Default to a cheap but good model
            messages=[
                {"role": "system", "content": f"You are an AI agent on Moltbook. Personality: {personality}"},
                {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
            ]
        )
        draft = response.choices[0].message.content
        return jsonify({"draft": draft})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)