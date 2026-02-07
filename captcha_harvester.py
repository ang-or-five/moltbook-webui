import os
import json
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from openai import OpenAI

# Constants
DATA_DIR = Path("data")
DATASET_FILE = DATA_DIR / "captcha_dataset.jsonl"
HARVEST_LOG_FILE = DATA_DIR / "harvest_log.json"
API_BASE = "https://www.moltbook.com/api/v1"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Captcha preprocessing patterns
GARBLED_PATTERNS = [
    (r'[\^\|\~\<\>\]\[\{\}\(\)\/\\\-\_]', ' '),  # Remove garbled symbols
    (r'\s+', ' '),  # Normalize whitespace
]

# Sample challenges for testing
SAMPLE_CHALLENGES = [
    "A] Lo-BsTeRr SwImS^ At/ TwEnTy ThReE ] CeNtiMeTeRs~ PeR| SeCoNd AnD] An-OtHeRr AdDs^ SeVeN, WhAtS< ToTaL?",
    "An] AgGrReSsIvE lO-bStEr'S ClAw ExErT s ThIrTy TwO NeUwToNs ~ anD/ iTs MaTe AdDs FiFtEeN NeUwToNs, WhAt Is ThE ToTaL FoRcE?",
    "TwO] LoBsTeRs] CoLlIdE] OnE] GoInG] TeN] MeTeRs~ PeR| SeCoNd, ThE] OtHeRr] At] SiX] MeTeRs~ PeR| SeCoNd, WhAt Is ThEiRr] CoMbInEd] SpEeD?",
    "A] LoBsTeRr] HaS] EiGhT] LeGs,] ThReE] MoRe] LoBsTeRs] HaVe] HoW] MaNy] LeGs] ToTaL?",
    "If] A] LoBsTeR] CaTchEs] FiVe] FiSh] PeR] DaY,] HoW] MaNy] FiSh] DoEs] It] CaTch] In] SeVeN] DaYs?",
    "ThReE] LoBsTeRs] HaVe] TwElVe] ClAwS] ToTaL,] HoW] MaNy] ClAwS] DoEs] OnE] LoBsTeR] HaVe?",
    "A] LoBsTeR] WaLkS] FoUr] MeTeRs,] ThEn] SwImS] NiNe] MeTeRs,] WhAt] Is] ThE] ToTaL] DiStAnCe?",
    "If] FiVe] LoBsTeRs] ShArE] TwEnTy] FiSh] EqUaLlY,] HoW] MaNy] FiSh] PeR] LoBsTeR?",
    "A] LoBsTeR] HaS] FoUrTeEn] SeGmEnTs] In] Its] BoDy,] AnD] LoSeS] ThReE,] HoW] MaNy] ReMaIn?",
    "TwO] LoBsTeRs] CoMbInEd] HaVe] SiXtEeN] LeGs,] HoW] MaNy] LeGs] PeR] LoBsTeR?",
]


def preprocess_captcha(text: str) -> str:
    """Clean and normalize garbled captcha text."""
    cleaned = text
    for pattern, replacement in GARBLED_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()


def extract_tags(response: Optional[str]) -> Dict[str, str]:
    """Extract compressed, equation, and answer tags from AI response."""
    result = {
        'compressed': '',
        'equation': '',
        'answer': ''
    }
    
    # Extract compressed reasoning
    compressed_match = re.search(r'<compressed>(.*?)</compressed>', response, re.DOTALL | re.IGNORECASE)
    if compressed_match:
        result['compressed'] = compressed_match.group(1).strip()
    
    # Extract equation
    equation_match = re.search(r'<equation>(.*?)</equation>', response, re.DOTALL | re.IGNORECASE)
    if equation_match:
        result['equation'] = equation_match.group(1).strip()
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    
    return result


def solve_captcha_with_ai(
    captcha_text: str,
    api_key: str,
    model: str = "gpt-4.1-nano",
    provider: str = "openai"
) -> Dict:
    """Solve captcha using AI and return structured data."""
    preprocessed = preprocess_captcha(captcha_text)
    
    # Build the prompt
    prompt = f"""{preprocessed}
Write a compressed reasoning process first, in <compressed> tags, then write the simple equation in <equation> tags, then write answer number in <answer>"""
    
    # Initialize client based on provider
    if provider == "openai":
        client = OpenAI(api_key=api_key)
    elif provider == "openrouter":
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    elif provider == "poe":
        client = OpenAI(
            api_key=api_key,
            base_url="https://models.dev/api/poe"
        )
        # Strip provider prefix for Poe if present
        if '/' in model:
            model = model.split('/')[-1]
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Call the API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a captcha solver. Parse the math problem and provide the answer in the requested format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    full_response = response.choices[0].message.content
    extracted = extract_tags(full_response)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'uncleaned': captcha_text,
        'preprocessed': preprocessed,
        'full_response': full_response,
        'compressed': extracted['compressed'],
        'equation': extracted['equation'],
        'answer': extracted['answer'],
        'model': model,
        'provider': provider
    }


def save_dataset_entry(entry: Dict) -> None:
    """Append a dataset entry to the JSONL file."""
    with open(DATASET_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def load_dataset() -> List[Dict]:
    """Load all entries from the dataset."""
    entries = []
    if DATASET_FILE.exists():
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def get_responded_post_ids() -> set:
    """Get set of post IDs that have already been harvested."""
    dataset = load_dataset()
    # Extract post IDs from entries that have them
    post_ids = set()
    for entry in dataset:
        if 'post_id' in entry:
            post_ids.add(entry['post_id'])
    return post_ids


def fetch_random_posts(
    api_key: str,
    n: int = 10,
    lean_niche: bool = True
) -> List[Dict]:
    """Fetch n random posts that haven't been responded to."""
    headers = {"Authorization": f"Bearer {api_key}"}
    responded_ids = get_responded_post_ids()
    
    # Fetch a larger pool to filter from
    pool_size = max(n * 3, 50)
    url = f"{API_BASE}/posts?sort=hot&limit={pool_size}"
    
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        if isinstance(data, list):
            posts = data
        else:
            posts = data.get('posts') or data.get('data') or []
        
        # Filter out already responded posts
        unresponded = [p for p in posts if p.get('id') not in responded_ids]
        
        if lean_niche and len(unresponded) > n:
            # Sort by engagement (score + comment count) and pick from lower end
            # but still include some randomness
            unresponded.sort(key=lambda x: x.get('score', 0) + len(x.get('comments', [])))
            # Take 70% from lower engagement, 30% random
            niche_count = int(n * 0.7)
            random_count = n - niche_count
            
            niche_posts = unresponded[:max(len(unresponded)//2, niche_count)]
            selected = random.sample(niche_posts, min(niche_count, len(niche_posts)))
            
            remaining = [p for p in unresponded if p not in selected]
            if remaining and random_count > 0:
                selected.extend(random.sample(remaining, min(random_count, len(remaining))))
            
            return selected
        else:
            return random.sample(unresponded, min(n, len(unresponded)))
    
    except Exception as e:
        print(f"Error fetching posts: {e}")
        return []


def harvest_captchas_from_posts(
    api_key: str,
    openai_api_key: str,
    posts: List[Dict],
    model: str = "gpt-4.1-nano",
    provider: str = "openai"
) -> List[Dict]:
    """Harvest captchas from a list of posts."""
    results = []
    
    for post in posts:
        # Generate a captcha from post content (or use sample if no content)
        content = post.get('content', '') or post.get('title', '')
        
        # Create a captcha based on post content or use sample
        if content and len(content) > 10:
            # Create a math problem from post metrics
            score = post.get('score', 0)
            comments_count = len(post.get('comments', []))
            captcha = f"A] PoSt] HaS] {score} ] UpVoTeS] AnD] {comments_count} ] CoMmEnTs,] WhAtS] ThE] ToTaL?"
        else:
            captcha = random.choice(SAMPLE_CHALLENGES)
        
        try:
            result = solve_captcha_with_ai(captcha, openai_api_key, model, provider)
            result['post_id'] = post.get('id')
            result['post_title'] = post.get('title', '')[:100]
            save_dataset_entry(result)
            results.append(result)
        except Exception as e:
            print(f"Error solving captcha for post {post.get('id')}: {e}")
            results.append({
                'timestamp': datetime.now().isoformat(),
                'uncleaned': captcha,
                'error': str(e),
                'post_id': post.get('id')
            })
    
    return results


def generate_multiple_captchas(
    openai_api_key: str,
    count: int = 10,
    model: str = "gpt-4.1-nano",
    provider: str = "openai"
) -> List[Dict]:
    """Generate and solve multiple sample captchas."""
    results = []
    
    for i in range(count):
        captcha = random.choice(SAMPLE_CHALLENGES)
        try:
            result = solve_captcha_with_ai(captcha, openai_api_key, model, provider)
            save_dataset_entry(result)
            results.append(result)
        except Exception as e:
            print(f"Error on captcha {i+1}: {e}")
    
    return results


class DatasetGenerator:
    """Generate train/eval splits and template-based responses."""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = Path(dataset_path) if dataset_path else DATASET_FILE
        self.entries = []
        self.load_entries()
    
    def load_entries(self):
        """Load entries from dataset file."""
        self.entries = []
        if self.dataset_path.exists():
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    
    def split_train_eval(self, train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and eval sets."""
        random.seed(seed)
        shuffled = self.entries.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_set = shuffled[:split_idx]
        eval_set = shuffled[split_idx:]
        
        return train_set, eval_set
    
    def build_response(self, entry: Dict, template: str) -> str:
        """Build a response string using template variables."""
        template_vars = {
            'compressed': entry.get('compressed', ''),
            'equation': entry.get('equation', ''),
            'answer': entry.get('answer', ''),
            'preprocessed': entry.get('preprocessed', ''),
            'uncleaned': entry.get('uncleaned', ''),
            'full_response': entry.get('full_response', '')
        }
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            return f"Error: Missing variable {e} in template"
    
    def export_to_jsonl(self, entries: List[Dict], output_path: str, response_template: str = None):
        """Export entries to JSONL file with optional response templating."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                export_entry = entry.copy()
                
                if response_template:
                    export_entry['formatted_response'] = self.build_response(entry, response_template)
                
                f.write(json.dumps(export_entry, ensure_ascii=False) + '\n')
    
    def export_train_eval(
        self,
        output_dir: str,
        train_ratio: float = 0.8,
        response_template: str = None,
        seed: int = 42
    ):
        """Export train and eval splits to separate files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_set, eval_set = self.split_train_eval(train_ratio, seed)
        
        self.export_to_jsonl(train_set, output_dir / "train.jsonl", response_template)
        self.export_to_jsonl(eval_set, output_dir / "eval.jsonl", response_template)
        
        return {
            'train_count': len(train_set),
            'eval_count': len(eval_set),
            'train_path': str(output_dir / "train.jsonl"),
            'eval_path': str(output_dir / "eval.jsonl")
        }
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total = len(self.entries)
        with_answers = sum(1 for e in self.entries if e.get('answer'))
        with_errors = sum(1 for e in self.entries if e.get('error'))
        
        return {
            'total_entries': total,
            'with_answers': with_answers,
            'with_errors': with_errors,
            'success_rate': (with_answers / total * 100) if total > 0 else 0
        }


# --- Draft Generation Helpers ---

DRAFTS_FILE = DATA_DIR / "pending_drafts.json"


def save_pending_drafts(drafts: List[Dict]) -> None:
    """Save pending drafts to file."""
    with open(DRAFTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(drafts, f, ensure_ascii=False, indent=2)


def load_pending_drafts() -> List[Dict]:
    """Load pending drafts from file."""
    if not DRAFTS_FILE.exists():
        return []
    try:
        with open(DRAFTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def generate_draft_for_post(
    post: Dict,
    client: OpenAI,
    personality: str = "A helpful AI agent",
    model: str = "gpt-4o-mini"
) -> str:
    """Generate a draft comment for a post."""
    title = post.get('title', '')
    content = post.get('content', '')
    author = post.get('author_name', 'someone')
    
    context = f"Post by {author}: {title}"
    if content:
        context += f"\n\n{content[:500]}"
    
    prompt = f"""Write a thoughtful, engaging comment response to this post. 
Be concise (1-3 sentences), authentic, and add value to the conversation.
Reference specific points from the post when relevant."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are an AI agent on Moltbook. Personality: {personality}"},
            {"role": "user", "content": f"Context: {context}\n\nTask: {prompt}"}
        ]
    )
    
    return response.choices[0].message.content


def generate_drafts_for_posts(
    posts: List[Dict],
    openai_api_key: str,
    personality: str = "A helpful AI agent",
    model: str = "gpt-4o-mini",
    provider: str = "openai"
) -> List[Dict]:
    """Generate drafts for multiple posts."""
    if provider == "openai":
        client = OpenAI(api_key=openai_api_key)
    elif provider == "openrouter":
        client = OpenAI(api_key=openai_api_key, base_url="https://openrouter.ai/api/v1")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    drafts = []
    for post in posts:
        try:
            draft_text = generate_draft_for_post(post, client, personality, model)
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
    
    save_pending_drafts(drafts)
    return drafts


def post_comment(
    post_id: str,
    content: str,
    api_key: str
) -> bool:
    """Post a comment to Moltbook."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {"content": content}
    
    try:
        resp = requests.post(
            f"{API_BASE}/posts/{post_id}/comments",
            json=payload,
            headers=headers
        )
        return resp.status_code in [200, 201]
    except Exception:
        return False
