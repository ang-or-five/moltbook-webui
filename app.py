import os
import json
import requests
import markdown
import bleach
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_caching import Cache
from openai import OpenAI
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

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

def render_markdown(text):
    if not text:
        return ""
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['extra', 'nl2br'])
    html = md.convert(text)
    # Sanitize HTML to prevent XSS
    allowed_tags = [
        'p', 'br', 'strong', 'b', 'em', 'i', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'code', 'pre', 'blockquote', 'a', 'img', 'hr', 'table',
        'thead', 'tbody', 'tr', 'th', 'td', 'strike', 'del'
    ]
    allowed_attrs = {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title']
    }
    return bleach.clean(html, tags=allowed_tags, attributes=allowed_attrs, strip=True)

app.jinja_env.filters['markdown'] = render_markdown

def is_safe_url(url):
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"}

app.jinja_env.tests['safe_url'] = is_safe_url

def get_provider_key(provider, purpose='draft'):
    # purpose can be 'draft', 'captcha', or 'shared'
    
    key_map = {
        'openai': [f'{purpose}_openai_api_key', 'shared_openai_api_key'],
        'openrouter': [f'{purpose}_openrouter_api_key', 'shared_openrouter_api_key'],
        'google': [f'{purpose}_google_api_key', 'shared_google_api_key'],
        'poe': [f'{purpose}_poe_api_key', 'shared_poe_api_key']
    }
    
    api_key = None
    for session_key in key_map.get(provider, []):
        api_key = session.get(session_key)
        if api_key: break
        
    if not api_key:
        env_map = {
            'openai': 'OPENAI_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'poe': 'POE_API_KEY'
        }
        api_key = os.getenv(env_map.get(provider, ''))
    return api_key

def get_ai_client(provider='openai', purpose='draft'):
    api_key = get_provider_key(provider, purpose)
    if not api_key:
        return None
        
    if provider == 'openai':
        return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    elif provider == 'openrouter':
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif provider == 'google':
        return OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif provider == 'poe':
        return OpenAI(api_key=api_key, base_url="https://api.poe.com/v1")
        
    return None

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
    # Check for testing token in env (supports .env and system env)
    testing_token = os.getenv('TESTING_TOKEN')
    
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
    
    return render_template('login.html', testing_token=testing_token)

@app.route('/login/testing', methods=['POST'])
def login_testing():
    """Auto-login with testing token for LLM agents"""
    testing_token = os.getenv('TESTING_TOKEN')
    
    if not testing_token:
        flash("Testing token not configured.", "danger")
        return redirect(url_for('login'))
    
    session['api_key'] = testing_token.strip()
    headers = {"Authorization": f"Bearer {session['api_key']}"}
    
    try:
        resp = http_session.get(f"{API_BASE}/agents/me", headers=headers)
        if resp.status_code == 200:
            flash("Logged in with testing token 🦞", "success")
            return redirect(url_for('index'))
        else:
            flash("Testing token invalid.", "danger")
            session.pop('api_key', None)
    except:
        flash("Network connection error.", "danger")
    
    return redirect(url_for('login'))

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

@app.route('/info')
def info():
    return render_template('info.html')

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
            cache.clear()
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
    
    # Define all setting keys for easy management
    setting_keys = [
        'draft_ai_provider', 'draft_model', 'draft_poe_reasoning_effort',
        'captcha_ai_provider', 'captcha_model', 'captcha_poe_reasoning_effort',
        'translation_provider', 'translation_model',
        'agent_personality', 'shared_openai_api_key', 'shared_openrouter_api_key',
        'shared_google_api_key', 'shared_poe_api_key',
        'draft_openai_api_key', 'draft_openrouter_api_key', 'draft_google_api_key', 'draft_poe_api_key',
        'captcha_openai_api_key', 'captcha_openrouter_api_key', 'captcha_google_api_key', 'captcha_poe_api_key'
    ]
    
    checkbox_keys = ['auto_solve_captcha', 'captcha_preprocessing', 'enable_captcha_harvesting', 'use_easymde']

    if request.method == 'POST':
        for key in setting_keys:
            if key in request.form:
                session[key] = request.form.get(key)
        
        for key in checkbox_keys:
            session[key] = key in request.form
            
        flash("Settings updated successfully.", "success")
        return redirect(url_for('settings'))
        
    # Build context with current session values or defaults
    ctx = {key: session.get(key, '') for key in setting_keys}
    for key in checkbox_keys:
        # Default enable_captcha_harvesting to False for new users, True for existing if already set
        default_val = True if key in ['captcha_preprocessing', 'use_easymde'] else False
        ctx[key] = session.get(key, default_val)
    
    # Overrides/Defaults
    if not ctx['agent_personality']: ctx['agent_personality'] = DEFAULT_PERSONALITY
    if not ctx['draft_ai_provider']: ctx['draft_ai_provider'] = 'openai'
    if not ctx['captcha_ai_provider']: ctx['captcha_ai_provider'] = 'openai'
    if not ctx['translation_provider']: ctx['translation_provider'] = 'google'
        
    return render_template('settings.html', **ctx)

@app.route('/ai/draft', methods=['POST'])
def ai_draft():
    if 'api_key' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    provider = session.get('draft_ai_provider', 'openai')
    client = get_ai_client(provider)
    if not client:
        return jsonify({"error": f"AI client for {provider} not configured. Check your API keys in Settings."}), 400
    
    data = request.json
    context = data.get('context', '')
    prompt = data.get('prompt', 'Write a short, engaging post about AI agents.')
    personality = session.get('agent_personality', DEFAULT_PERSONALITY)
    
    model = session.get('draft_model', 'gpt-4o-mini')
    provider = session.get('draft_ai_provider', 'openai')
    
    # Strip provider prefix for Poe
    if provider == 'poe' and '/' in model:
        model = model.split('/')[-1]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are an AI agent on Moltbook. Personality: {personality}"},
                {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
            ]
        )
        draft = response.choices[0].message.content
        return jsonify({"draft": draft})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    text = data.get('text')
    dest = data.get('dest', 'en')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        translated = GoogleTranslator(source='auto', target=dest).translate(text)
        return jsonify({
            "translated_text": translated,
            "src_lang": "auto",
            "dest_lang": dest
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Captcha & Data Harvest Routes ---

@app.route('/captcha/solve', methods=['POST'])
def captcha_solve():
    """Solve a single captcha challenge and save to dataset."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    challenge = data.get('challenge', '')
    provider = data.get('provider', 'openai')
    model = data.get('model', 'gpt-4.1-nano')
    
    if not challenge:
        return jsonify({"error": "No challenge provided"}), 400
    
    api_key = get_provider_key(provider, 'captcha')
    if not api_key:
        return jsonify({"error": f"{provider.capitalize()} API key not configured for Captcha. Please add it in Settings."}), 400
    
    try:
        from captcha_harvester import solve_captcha_with_ai, save_dataset_entry
        
        result = solve_captcha_with_ai(challenge, api_key, model, provider)
        save_dataset_entry(result)
        
        return jsonify({
            "success": True,
            "answer": result['answer'],
            "cleaned_text": result['preprocessed'],
            "compressed": result['compressed'],
            "equation": result['equation'],
            "task_id": result['timestamp']
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "cleaned_text": challenge
        }), 500

@app.route('/captcha/harvest', methods=['POST'])
def captcha_harvest():
    """Harvest multiple captchas from sample challenges."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    count = data.get('count', 10)
    provider = data.get('provider', 'openai')
    model = data.get('model', 'gpt-4.1-nano')
    
    api_key = get_provider_key(provider, 'captcha')
    if not api_key:
        return jsonify({"error": f"{provider.capitalize()} API key not configured for Captcha. Please add it in Settings."}), 400
    
    try:
        from captcha_harvester import generate_multiple_captchas
        
        results = generate_multiple_captchas(api_key, count, model, provider)
        
        return jsonify({
            "success": True,
            "harvested": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/captcha/batch', methods=['POST'])
def captcha_batch():
    """Harvest captchas from n random unresponded posts."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    n = data.get('n', 10)
    lean_niche = data.get('lean_niche', True)
    provider = data.get('provider', 'openai')
    model = data.get('model', 'gpt-4.1-nano')
    
    api_key = session.get('api_key')
    ai_key = get_provider_key(provider, 'captcha')
    
    if not ai_key:
        return jsonify({"error": f"{provider.capitalize()} API key not configured for Captcha. Please add it in Settings."}), 400
    
    try:
        from captcha_harvester import fetch_random_posts, harvest_captchas_from_posts
        
        # Fetch random posts
        posts = fetch_random_posts(api_key, n, lean_niche)
        
        if not posts:
            return jsonify({"error": "No unresponded posts found"}), 404
        
        # Harvest captchas from posts
        results = harvest_captchas_from_posts(api_key, ai_key, posts, model, provider)
        
        return jsonify({
            "success": True,
            "posts_found": len(posts),
            "harvested": len([r for r in results if 'answer' in r]),
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/captcha/dataset', methods=['GET'])
def captcha_dataset():
    """Get dataset statistics and entries."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        from captcha_harvester import DatasetGenerator
        
        generator = DatasetGenerator()
        stats = generator.get_stats()
        
        # Get recent entries (last 50)
        recent = generator.entries[-50:] if len(generator.entries) > 50 else generator.entries
        
        return jsonify({
            "stats": stats,
            "recent_entries": recent,
            "total_entries": len(generator.entries)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/<provider>', methods=['GET'])
@cache.cached(timeout=300)
def api_models(provider):
    """Fetch available models for a provider."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # Fetch OpenRouter models from their API
        if provider == 'openrouter':
            try:
                resp = http_session.get('https://openrouter.ai/api/v1/models', timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for model in data.get('data', []):
                        models.append({
                            "id": model.get('id'),
                            "name": model.get('name', model.get('id')),
                            "context_length": model.get('context_length', 0),
                            "description": model.get('description', '')[:100],
                            "pricing": {
                                "prompt": model.get('pricing', {}).get('prompt', 0),
                                "completion": model.get('pricing', {}).get('completion', 0)
                            }
                        })
                    return jsonify({"models": models})
                else:
                    # Fallback to hardcoded if API fails
                    return jsonify({"models": [
                        {"id": "openai/gpt-4.1-nano", "name": "GPT-4.1 Nano (OR)", "context_length": 128000},
                        {"id": "openai/gpt-4.1-mini", "name": "GPT-4.1 Mini (OR)", "context_length": 128000},
                    ]})
            except Exception as e:
                return jsonify({"error": f"OpenRouter API error: {str(e)}"}), 500
        
        # Fetch from models.dev for supported providers
        models_dev_providers = {
            'openai': ['openai', 'OpenAI'],
            'google': ['google', 'Google', 'google-ai', 'googleai'],
            'poe': ['poe', 'Poe', 'POE']
        }
        
        if provider in models_dev_providers:
            try:
                resp = http_session.get('https://models.dev/api.json', timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    # Find provider in the schema (try multiple key variations)
                    target_provider = None
                    for key_variation in models_dev_providers[provider]:
                        if key_variation in data:
                            target_provider = data[key_variation]
                            break
                    
                    # Try case-insensitive match if not found
                    if not target_provider:
                        for key, prov in data.items():
                            if key.lower() == provider or key.lower() in [p.lower() for p in models_dev_providers[provider]]:
                                target_provider = prov
                                break
                    
                    if target_provider and 'models' in target_provider:
                        provider_models = target_provider['models']
                        for model_id, model in provider_models.items():
                            # Check if vision is supported from modalities
                            modalities = model.get('modalities', {})
                            has_vision = 'image' in modalities.get('input', []) if isinstance(modalities, dict) else False
                            
                            limit = model.get('limit', {})
                            context_length = limit.get('context', 0) if isinstance(limit, dict) else 0
                            
                            models.append({
                                "id": model.get('id', model_id),
                                "name": model.get('name', model_id),
                                "context_length": context_length,
                                "reasoning": model.get('reasoning', False),
                                "vision": has_vision,
                                "tool_call": model.get('tool_call', False),
                                "cost": model.get('cost', {})
                            })
                        return jsonify({"models": models})
            except Exception as e:
                print(f"models.dev API error for {provider}: {e}")
                pass
            
            # Fallback to hardcoded
            fallbacks = {
                'openai': [
                    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gpt-4o", "name": "GPT-4o", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                ],
                'google': [
                    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "context_length": 1000000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "context_length": 2000000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "context_length": 1000000, "reasoning": False, "vision": True, "tool_call": True},
                ],
                'poe': [
                    {"id": "gpt-5-nano", "name": "GPT-5 Nano (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gpt-5-mini", "name": "GPT-5 Mini (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                    {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                ]
            }
            return jsonify({"models": fallbacks.get(provider, [])})
        
        # Default hardcoded models for other providers
        models_map = {
            'google': [
                {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "context_length": 1000000, "reasoning": False, "vision": True, "tool_call": True},
                {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "context_length": 2000000, "reasoning": False, "vision": True, "tool_call": True},
                {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "context_length": 1000000, "reasoning": False, "vision": True, "tool_call": True},
            ],
            'poe': [
                {"id": "gpt-5-nano", "name": "GPT-5 Nano (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                {"id": "gpt-5-mini", "name": "GPT-5 Mini (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
                {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini (Poe)", "context_length": 128000, "reasoning": False, "vision": True, "tool_call": True},
            ]
        }
        
        models = models_map.get(provider, [])
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Captcha Demo Route ---

@app.route('/captcha_demo')
def captcha_demo():
    """Show captcha demo page."""
    if 'api_key' not in session:
        return redirect(url_for('login'))
    
    return render_template('captcha_demo.html')

# --- Mass Draft Generation Routes ---

@app.route('/mass-drafts', methods=['GET'])
def mass_drafts():
    """Show mass draft generation interface."""
    if 'api_key' not in session:
        return redirect(url_for('login'))
    
    return render_template('mass_drafts.html')

@app.route('/mass-drafts/generate', methods=['POST'])
def mass_drafts_generate():
    """Generate drafts for n random posts."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    n = data.get('n', 10)
    lean_niche = data.get('lean_niche', True)
    
    api_key = session.get('api_key')
    provider = session.get('draft_ai_provider', 'openai')
    personality = session.get('agent_personality', DEFAULT_PERSONALITY)
    
    client = get_ai_client(provider)
    if not client:
        return jsonify({"error": f"{provider.capitalize()} API key not configured. Please add it in Settings."}), 400
    
    try:
        from captcha_harvester import fetch_random_posts, generate_drafts_for_posts
        
        # Fetch random posts
        posts = fetch_random_posts(api_key, n, lean_niche)
        
        if not posts:
            return jsonify({"error": "No unresponded posts found"}), 404
        
        model = session.get('draft_model', 'gpt-4o-mini')
        
        # Strip provider prefix for Poe
        if provider == 'poe' and '/' in model:
            model = model.split('/')[-1]
        
        # Generate drafts manually using the client from settings
        drafts = []
        for post in posts:
            try:
                title = post.get('title', '')
                content = post.get('content', '')
                author = post.get('author_name', 'someone')
                
                context = f"Post by {author}: {title}"
                if content:
                    context += f"\n\n{content[:500]}"
                
                prompt = "Write a thoughtful, engaging comment response to this post. Be concise (1-3 sentences), authentic, and add value to the conversation."
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are an AI agent on Moltbook. Personality: {personality}"},
                        {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
                    ]
                )
                
                draft_text = response.choices[0].message.content
                
                drafts.append({
                    'post_id': post.get('id'),
                    'post_title': post.get('title', '')[:100],
                    'post_content': post.get('content', '')[:200] if post.get('content') else '',
                    'post_author': post.get('author_name', 'unknown'),
                    'draft': draft_text,
                    'status': 'pending',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                drafts.append({
                    'post_id': post.get('id'),
                    'post_title': post.get('title', '')[:100],
                    'post_content': post.get('content', '')[:200] if post.get('content') else '',
                    'post_author': post.get('author_name', 'unknown'),
                    'draft': f"Error: {str(e)}",
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save drafts
        from captcha_harvester import save_pending_drafts
        save_pending_drafts(drafts)
        
        return jsonify({
            "success": True,
            "count": len(drafts),
            "drafts": drafts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mass-drafts/list', methods=['GET'])
def mass_drafts_list():
    """Get list of pending drafts."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        from captcha_harvester import load_pending_drafts
        drafts = load_pending_drafts()
        return jsonify({"drafts": drafts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mass-drafts/approve', methods=['POST'])
def mass_drafts_approve():
    """Approve and post a draft."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    post_id = data.get('post_id')
    draft_text = data.get('draft')
    
    if not post_id or not draft_text:
        return jsonify({"error": "Missing post_id or draft"}), 400
    
    api_key = session.get('api_key')
    
    try:
        from captcha_harvester import post_comment, load_pending_drafts, save_pending_drafts
        
        # Post the comment
        success = post_comment(post_id, draft_text, api_key)
        
        if success:
            # Update draft status
            drafts = load_pending_drafts()
            for draft in drafts:
                if draft.get('post_id') == post_id and draft.get('draft') == draft_text:
                    draft['status'] = 'posted'
                    break
            save_pending_drafts(drafts)
            
            return jsonify({"success": True, "message": "Draft posted successfully"})
        else:
            return jsonify({"error": "Failed to post comment"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mass-drafts/reject', methods=['POST'])
def mass_drafts_reject():
    """Reject a draft."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    post_id = data.get('post_id')
    draft_text = data.get('draft')
    
    try:
        from captcha_harvester import load_pending_drafts, save_pending_drafts
        
        drafts = load_pending_drafts()
        for draft in drafts:
            if draft.get('post_id') == post_id and draft.get('draft') == draft_text:
                draft['status'] = 'rejected'
                break
        save_pending_drafts(drafts)
        
        return jsonify({"success": True, "message": "Draft rejected"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mass-drafts/clear', methods=['POST'])
def mass_drafts_clear():
    """Clear all pending drafts."""
    if 'api_key' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        from captcha_harvester import save_pending_drafts
        save_pending_drafts([])
        return jsonify({"success": True, "message": "All drafts cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
