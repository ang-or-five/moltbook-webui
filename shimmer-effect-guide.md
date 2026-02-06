# Shimmer Effect Guide

## What is the Shimmer Effect?

The shimmer effect is a CSS animation that creates a smooth, gradient-based loading animation. It gives users visual feedback that content is being loaded, making the app feel more responsive.

## How It Works

The shimmer uses a linear gradient that moves across the element using CSS animations:

```css
@keyframes shimmer {
    0% {
        background-position: 200% 0;
    }
    100% {
        background-position: -200% 0;
    }
}
```

## Implementation in LiteSearch

### Loading Text Shimmer

Located in: `static/css/style.css` (lines 560-583)

```css
.loading span {
    background: linear-gradient(
        90deg,
        #9aa0a6 0%,
        #202124 50%,
        #9aa0a6 100%
    );
    background-size: 200% auto;
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 0.8s linear infinite;
    font-size: 3em;
    font-weight: 600;
}
```

**Key Properties:**
- `background-size: 200% auto` - Makes the gradient twice as wide as the element
- `background-clip: text` - Clips the background to only show within text
- `-webkit-text-fill-color: transparent` - Makes text transparent to show background
- `animation: shimmer 0.8s linear infinite` - Continuously animates the gradient

### Usage in HTML

```html
<div class="loading">
    <span id="loading-text">Searching...</span>
</div>
```

### JavaScript Control

The loading shimmer is controlled via JavaScript in `static/js/search.js`:

```javascript
// Show loading after 70ms delay
loadingTimeout = setTimeout(() => {
    loading.style.display = 'flex';
    loadingText.textContent = 'Searching... 0.0s';

    // Update elapsed time every 100ms
    elapsedInterval = setInterval(updateElapsedTime, 100);
}, 70);
```

**Why the 70ms delay?**
- Prevents flash of loading state for fast searches
- Only shows if search takes longer than 70ms

## Customizing the Shimmer

### Change Colors

```css
background: linear-gradient(
    90deg,
    #your-light-color 0%,
    #your-dark-color 50%,
    #your-light-color 100%
);
```

### Change Speed

```css
animation: shimmer 0.5s linear infinite; /* Faster */
animation: shimmer 1.5s linear infinite; /* Slower */
```

### Change Direction

```css
/* Right to left (default) */
background: linear-gradient(90deg, ...);

/* Left to right */
background: linear-gradient(270deg, ...);

/* Top to bottom */
background: linear-gradient(180deg, ...);
```

## Best Practices

1. **Use for indeterminate loading** - When you don't know how long something will take
2. **Add delays** - Don't show immediately for fast operations (use 50-100ms delay)
3. **Keep it subtle** - Don't make colors too contrasting
4. **Use smooth timing** - `linear` or `ease-in-out` work best

## Browser Support

- Modern browsers: Full support
- Safari: Requires `-webkit-` prefixes (already included)
- IE11: Not supported (gracefully degrades to solid color)

## Common Issues

### Shimmer not visible
- Check that `background-clip: text` is set
- Ensure `-webkit-text-fill-color: transparent` is present
- Verify `background-size: 200% auto` or larger

### Shimmer too fast/slow
- Adjust the animation duration in the `animation` property
- Typical range: 0.5s - 2s

### Shimmer stutters
- Use `linear` timing function instead of `ease`
- Ensure `will-change: background-position` for GPU acceleration

## Alternative: Skeleton Shimmer

For loading cards/boxes instead of text:

```css
.skeleton {
    background: linear-gradient(
        90deg,
        #f0f0f0 0%,
        #e0e0e0 50%,
        #f0f0f0 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 4px;
    height: 20px;
}
```

## Performance Tips

1. **Limit simultaneous shimmers** - Don't animate 100 elements at once
2. **Use `will-change`** - Hints to browser for optimization:
   ```css
   will-change: background-position;
   ```
3. **Pause when hidden** - Stop animations for off-screen elements
4. **Use GPU acceleration** - `transform` and `opacity` are GPU-accelerated

## Related Files

- CSS: `static/css/style.css` (lines 550-583)
- JavaScript: `static/js/search.js` (lines 382-388, 422-435)
- HTML: `templates/index.html` (lines 94-97)

## Quick Reference

```css
/* Basic shimmer template */
.your-element {
    background: linear-gradient(90deg, #colorA 0%, #colorB 50%, #colorA 100%);
    background-size: 200% auto;
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 0.8s linear infinite;
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
```

## Examples in LiteSearch

1. **Search loading state** - Shows while search is in progress
2. **Elapsed time display** - Updates every 100ms with shimmer effect
3. **Status indicators** - Used in re-indexing status (pulsing animation)

---

**Remember:** The shimmer effect is defined in the CSS file, controlled via JavaScript display logic, and applied to specific loading states. The key is the gradient moving across transparent text using `background-clip: text`.
