# Mass Draft Generation Feature - Complete

## Summary
Successfully implemented a web-based mass draft generation feature that allows users to:
1. Fetch n random unresponded posts (with niche-leaning option)
2. Generate AI drafts for all posts at once
3. View all drafts in a scrollable mass-view interface
4. Quickly approve/reject drafts with one-click buttons
5. Batch approve all pending drafts

## Files Created/Modified

### 1. `captcha_harvester.py` - Enhanced with Draft Functions
Added new helper functions:
- `save_pending_drafts()` - Save drafts to JSON file
- `load_pending_drafts()` - Load pending drafts
- `generate_draft_for_post()` - Generate single draft using OpenAI
- `generate_drafts_for_posts()` - Batch generate drafts for multiple posts
- `post_comment()` - Post approved comment to Moltbook API

### 2. `app.py` - New Web Routes
Added mass draft routes:
- `GET /mass-drafts` - Main interface page
- `POST /mass-drafts/generate` - Generate drafts for n random posts
- `GET /mass-drafts/list` - Get list of pending drafts
- `POST /mass-drafts/approve` - Approve and post a draft
- `POST /mass-drafts/reject` - Reject a draft
- `POST /mass-drafts/clear` - Clear all drafts

### 3. `templates/mass_drafts.html` - Web Interface
Features:
- Generation settings (num posts, model, provider, niche-leaning)
- Progress bar during generation
- Stats summary (total, pending, approved, rejected)
- Scrollable drafts list with post previews
- Quick action buttons (Approve & Post, Reject, View Post)
- Batch "Approve All Pending" button
- Visual status indicators (badges, color coding)
- Auto-refresh on page load

### 4. `templates/layout.html` - Navigation
Added "Mass Drafts" link to navbar with robot icon

## How It Works

1. **User clicks "Generate Drafts"**
   - Fetches n random unresponded posts
   - If "lean niche" is checked, prioritizes lower-engagement posts
   - Generates AI draft for each post using existing `/ai/draft` logic
   - Saves all drafts to `data/pending_drafts.json`

2. **Draft Display**
   - Shows post title, author, and preview
   - Shows generated draft content
   - Status badge (Pending/Posted/Rejected/Error)

3. **Quick Actions**
   - **Approve & Post**: Immediately posts the comment
   - **Reject**: Marks draft as rejected (won't be posted)
   - **View Post**: Opens post in new tab
   - **Approve All**: Batch approves all pending drafts

## Usage

1. Go to `/mass-drafts` in the web interface
2. Set number of posts and options
3. Click "Generate Drafts"
4. Review drafts in the scrollable list
5. Click "Approve & Post" or "Reject" for each
6. Or use "Approve All Pending" to batch post

## API Endpoints

```bash
# Generate drafts
POST /mass-drafts/generate
{
  "n": 10,
  "lean_niche": true,
  "model": "gpt-4o-mini",
  "provider": "openai"
}

# List drafts
GET /mass-drafts/list

# Approve draft
POST /mass-drafts/approve
{
  "post_id": "...",
  "draft": "draft text"
}

# Reject draft
POST /mass-drafts/reject
{
  "post_id": "...",
  "draft": "draft text"
}

# Clear all
POST /mass-drafts/clear
```

## Data Storage
Pending drafts stored in: `data/pending_drafts.json`

## Integration
Uses existing:
- `get_ai_client()` from app.py
- `fetch_random_posts()` from captcha_harvester.py
- Moltbook API for posting comments
- Session-based authentication
